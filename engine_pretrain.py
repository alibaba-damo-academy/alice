import argparse
import os
import sys
import datetime
import time
import math
import json
import numpy as np
import utils
import models
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm
from models.nnFormer import nnFormer
from interfaces import init_model, get_embedding, find_point_in_vol
import pickle

from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from tensorboardX import SummaryWriter

from models.head import AliceHead
from loader import get_loader
from loss import Loss
from CASA import CASA_Module

#Dp
from torch.multiprocessing import Process
import torch.utils.data.distributed
import torch.distributed as dist


def train_one_epoch(student, teacher, teacher_without_ddp, alice_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, sam_cfg, CASA, args):
    
    global memory_queue_patch
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    
    # common params
    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in student.module.named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in teacher_without_ddp.named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
    params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]

    pred_labels = []
    iters = 0
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        # image = batch['image'].cuda(non_blocking=True)
        image = batch['image'].to(args.local_rank, non_blocking=True)
        name1 = batch['name'][0]
        #emb1 = np.load(args.embed_dir+name1+'.npy', allow_pickle=True).item()
        emb_path_1 = args.embed_dir + 'Embeddings' + name1 + '.pkl'
        with open(emb_path_1, 'rb') as file:
            emb1 = pickle.load(file)
            #print(emb1[0].shape, emb1[1].shape, emb1[2].shape, emb1[3])
        
        if epoch == 0 and iters == 0:
            memory_queue_patch = batch
        
        memory_image = memory_queue_patch['image']
        name2 = memory_queue_patch['name'][0]
        #emb2 = np.load(args.embed_dir+name2+'.npy', allow_pickle=True).item()
        emb_path_2 = args.embed_dir + 'Embeddings' + name2 + '.pkl'
        with open(emb_path_2, 'rb') as file_2:
            emb2 = pickle.load(file_2)
        
        iter_points, scores = 0, 0
        while iter_points<=100 and scores<=0.7:
            pts = utils.select_random_points(2, image.transpose(2, 4))
            pts1, pts2 = pts[0], pts[1]
            pts1_pred, scores = find_point_in_vol(emb1, emb2, [pts1], sam_cfg)
            iter_points += 1
        pts1_pred = pts1_pred[0]
        query =  utils.crop_tensor_new(image, pts1, args.roi_x, args.roi_y, args.roi_z).to(device)
        anchor = utils.crop_tensor_new(memory_image, pts1_pred, args.roi_x, args.roi_y, args.roi_z).to(device)
        memory_queue_patch = batch
        
        query_aug, anchor_aug= utils.data_aug(args, query), utils.data_aug(args, anchor)
        images_normal = [query, anchor]
        images_aug = [query_aug, anchor_aug]
        masks = utils.random_mask(args, images_normal)
   
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images_normal)
            student_output = student(images_normal, mask=masks)
            
            feat1_ali, feat2_ali = CASA(student_output, teacher_output)
            
            all_loss = alice_loss(images_normal, student_output, teacher_output, feat1_ali, feat2_ali, masks, epoch)
            loss = all_loss.pop('loss')
            
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)
        
        # log statistics
        probs1 = teacher_output[0].chunk(args.global_crops_number)
        probs2 = student_output[0].chunk(args.global_crops_number)
        pred1 = utils.concat_all_gather(probs1[0].max(dim=1)[1]) 
        pred2 = utils.concat_all_gather(probs2[1].max(dim=1)[1])
        acc = (pred1 == pred2).sum() / pred1.size(0)
        pred_labels.append(pred1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        for key, value in all_loss.items():
            metric_logger.update(**{key: value.item()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(acc=acc)
        
        iters += 1

    pred_labels = torch.cat(pred_labels).cpu().detach().numpy()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    return return_dict