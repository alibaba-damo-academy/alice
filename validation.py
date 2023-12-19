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

from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from tensorboardX import SummaryWriter

from models.head import AliceHead
from loader import get_loader
from loss import Loss
from CASA import CASA_Module
from engine_pretrain import train_one_epoch
#Dp
from torch.multiprocessing import Process
import torch.utils.data.distributed
import torch.distributed as dist


def validation(args, student, teacher, alice_loss, test_loader, epoch, sam_cfg, CASA):
    global memory_queue_patch
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    student.eval()
    teacher.eval()
    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)

    with torch.no_grad():
        for it, batch in enumerate(test_loader):
            #image = batch['image'].cuda(non_blocking=True)
            image = batch['image'].to(args.local_rank, non_blocking=True)
            name1 = batch['name'][0]
            emb1 = np.load(args.embed_dir+name1+'.npy', allow_pickle=True).item()
        
            memory_image = memory_queue_patch['image']
            name2 = memory_queue_patch['name'][0]
            emb2 = np.load(args.embed_dir+name2+'.npy', allow_pickle=True).item()
        
            pts = utils.select_random_points(1, image.transpose(2, 4))
            pts1 = pts[0]

            pts1_pred, scores = find_point_in_vol(emb1, emb2, [pts1], sam_cfg)
            pts1_pred = pts1_pred[0]

            query =  utils.crop_tensor_new(image, pts1, args.roi_x, args.roi_y, args.roi_z).to(device)
            anchor = utils.crop_tensor_new(memory_image, pts1_pred, args.roi_x, args.roi_y, args.roi_z).to(device)


            query_aug, anchor_aug = utils.data_aug(args, query), utils.data_aug(args, anchor)
            images_normal = [query, anchor]
            images_aug = [query_aug, anchor_aug]
        
            masks = utils.random_mask(args, images_normal)
            
            teacher_output = teacher(images_normal)
            student_output = student(images_normal, mask=masks)
            feat1_ali, feat2_ali = CASA(student_output, teacher_output)
            
            all_loss = alice_loss(images_normal, student_output, teacher_output, feat1_ali, feat2_ali, masks, epoch)

            all_loss['val_cls'] = all_loss.pop('cls')
            all_loss['val_patch'] = all_loss.pop('patch')
            all_loss['val_recon'] = all_loss.pop('recon')
            all_loss['val_loss'] = all_loss.pop('loss')
                                    
            # logging
            torch.cuda.synchronize()
            metric_logger.update(val_loss=all_loss['val_loss'].item())
            for key, value in all_loss.items():
                metric_logger.update(**{key: value.item()})
        
        metric_logger.synchronize_between_processes()
        print("Averaged validation stats:", metric_logger)
        return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    return return_dict              

if __name__ == '__main__':
    #parser = argparse.ArgumentParser('Alice', parents=[get_args_parser()])
    #args = parser.parse_args()
    #torch.cuda.set_device(args.local_rank)
    #Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    #validation(args)
    print("validation done!")