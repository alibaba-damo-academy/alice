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


# from evaluation.unsupervised.unsup_cls import eval_pred
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


def get_args_parser():
    parser = argparse.ArgumentParser('Alice', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='nnformer', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'deit_tiny', 'deit_small',
                 'swin_tiny','swin_small', 'swin_base', 'swin_large'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--window_size', default=7, type=int, help="""Size of window - default 7.
        This config is only valid for Swin Transofmer and is ignoired for vanilla ViT architectures.""")
    parser.add_argument('--out_dim', default=512, type=int, help="""Dimensionality of
        output for [CLS] token.""")
    parser.add_argument('--patch_out_dim', default=512, type=int, help="""Dimensionality of
        output for patch tokens.""")
    parser.add_argument('--feature_size', default=48, type=int, help='feature size')
    parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
    parser.add_argument('--out_channels', default=1, type=int, help='number of output channels')
    
    parser.add_argument('--shared_head', default=False, type=utils.bool_flag, help="""Wether to share 
        the same head for [CLS] token output and patch tokens output. When set to false, patch_out_dim
        is ignored and enforced to be same with out_dim. (Default: False)""")
    parser.add_argument('--shared_head_teacher', default=True, type=utils.bool_flag, help="""See above.
        Only works for teacher model. (Defeault: True)""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--norm_in_head', default=None,
        help="Whether to use batch normalizations in projection head (Default: None)")
    parser.add_argument('--act_in_head', default='gelu',
        help="Whether to use batch normalizations in projection head (Default: gelu)")
    parser.add_argument('--use_masked_im_modeling', default=True, type=utils.bool_flag,
        help="Whether to use masked image modeling (mim) in backbone (Default: True)")
    parser.add_argument('--pred_ratio', default=0.3, type=float, nargs='+', help="""Ratio of partial prediction.
        If a list of ratio is specified, one of them will be randomly choosed for each patch.""")
    parser.add_argument('--pred_ratio_var', default=0, type=float, nargs='+', help="""Variance of partial prediction
        ratio. Length should be indentical to the length of pred_ratio. 0 for disabling. """)
    parser.add_argument('--pred_shape', default='block', type=str, help="""Shape of partial prediction.""")
    parser.add_argument('--pred_start_epoch', default=0, type=int, help="""Start epoch to perform masked
        image prediction. We typically set this to 50 for swin transformer. (Default: 0)""")
    parser.add_argument('--lambda1', default=1.0, type=float, help="""loss weight for contrastive
        loss over [CLS] tokens (Default: 1.0)""")
    parser.add_argument('--lambda2', default=1.0, type=float, help="""loss weight for contrastive 
        loss over patch token embeddings (Default: 1.0)""")
    parser.add_argument('--lambda3', default=1.0, type=float, help="""loss weight for MAE 
        loss over masked patch tokens (Default: 1.0)""")       
    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.008, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.01, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_patch_temp', default=0.008, type=float, help="""See 
        `--warmup_teacher_temp`""")
    parser.add_argument('--teacher_patch_temp', default=0.01, type=float, help=""""See 
        `--teacher_temp`""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--epochs', default=500, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--warmup_epochs", default=50, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument("--lr", default=5e-2, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument('--min_lr', type=float, default=1e-4, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--load_from', default=None, help="""Path to load checkpoints to resume training.""")
    parser.add_argument('--drop_path', type=float, default=0.1, help="""Drop path rate for student network.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_number', type=int, default=2, help="""Number of global
        views to generate. Default is to use two global crops. """)

    
    # Medical data process
    parser.add_argument('--a_min', default=-125, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=225, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--roi_x', default=64, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=64, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=64, type=int, help='roi size in z direction')
    parser.add_argument('--batch_size', default=1, type=int, help='number of batch size')
    parser.add_argument('--sw_batch_size', default=2, type=int, help='number of sliding window batch size')
    parser.add_argument('--normal_dataset', default=False, action='store_true', help='use monai Dataset class')
    parser.add_argument('--smartcache_dataset', default=True, action='store_true', help='use monai smartcache Dataset class')
    parser.add_argument('--distributed', action='store_true', default=True, help='enable distributed training')
    parser.add_argument("--threshold", type=float, default=0.6, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    parser.add_argument('--mask_ratio', default=0.75, help='mask ratio')
    
    # Misc
    parser.add_argument("--data_dir", default="/mnt/workspace/Flare/Dataset/", type=str, help="dataset directory")
    parser.add_argument("--json_list", default="pretrainset.json", type=str, help="dataset json file")
    parser.add_argument('--output_dir', default="./results/final-ddp/", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--embed_dir', default="/mnt/workspace/workgroup/Flare/embedding/", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--config_file', default="./configs/sam/sam_r18_i3d_fpn_1x_multisets_sgd_T_0.5_half_test.py", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--checkpoint_file', default="./PointMatch/iter_38000.pth", type=str, help='Path to save logs and checkpoints.')
    
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--eval_epoch', default=10, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=5, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--master_port", default='29501', type=str, help="Please ignore and do not set this argument.")
    parser.add_argument("--gpu", default=0, type=int, help="Please ignore and do not set this argument.")
    
    
    return parser

def train_Alice(args):
    utils.init_distributed_mode(args)
    # utils.fix_random_seeds(args.seed)
    
    # ============ preparing data ... ============
    pred_size = args.patch_size * 8 if 'swin' in args.arch else args.patch_size
    data_loader, test_loader = get_loader(args)

    print(f"Data loaded: there are {len(data_loader)} images.")

    # ============ building student and teacher networks ... ============
    student = nnFormer(
              img_size=(args.roi_x, args.roi_y, args.roi_z),
              input_channels=args.in_channels,
              output_channels=args.out_channels,
              embedding_dim=args.feature_size,
              )
    teacher = nnFormer(
              img_size=(args.roi_x, args.roi_y, args.roi_z),
              input_channels=args.in_channels,
              output_channels=args.out_channels,
              embedding_dim=args.feature_size,
              )
    embed_dim = args.feature_size * (2 ** 3)
    
    casa_module = CASA_Module.CASA(args.out_dim, 2048, 8, 64, 64).cuda()
    casa_module = nn.parallel.DistributedDataParallel(casa_module, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True)
    
    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(
        student, 
        AliceHead(
            embed_dim,
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            norm_last_layer=args.norm_last_layer,
            shared_head=args.shared_head,
        ),
    )
    teacher = utils.MultiCropWrapper(
        teacher,
        AliceHead(
            embed_dim, 
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            shared_head=args.shared_head_teacher,
        ),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True) if \
            'nnformer' in args.arch else nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], find_unused_parameters=True)
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True) if \
        'nnformer' in args.arch else nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=True)
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")
    
    
    # ============ building SAM model ... ============    
    sam_model, sam_cfg = init_model(args.config_file, args.checkpoint_file)

    # ============ preparing loss ... ============
    same_dim = args.shared_head or args.shared_head_teacher
    alice_loss = Loss(
        args.out_dim,
        args.out_dim if same_dim else args.patch_out_dim,
        args.global_crops_number,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_patch_temp,
        args.teacher_patch_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        mim_start_epoch=args.pred_start_epoch,
    ).cuda()

    if utils.is_main_process(): # Tensorboard configuration
        local_runs = os.path.join(args.output_dir, 'tf_logs')
        writer = SummaryWriter(logdir=local_runs)
        
    # ============ preparing optimizer ... ============
    # params_groups = utils.get_params_groups(student)
    params_groups = utils.get_params_groups_dual(student, casa_module)
    
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, 
        len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay, 
        args.weight_decay_end,
        args.epochs, 
        len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                            args.epochs, len(data_loader))
                  
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    if args.load_from:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            alice_loss=alice_loss,
        )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting training Alice!")
    
    best_val = 1e8
    global memory_queue_patch
    memory_queue_patch = 0
    
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        # data_loader.dataset.set_epoch(epoch)

        # ============ training one epoch of Alice ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, alice_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, sam_cfg, casa_module, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'alice_loss': alice_loss.state_dict(),
        }
        
        if args.eval_epoch and (epoch % args.eval_epoch == 0):
            val_stats = validation(args, student, teacher, alice_loss, test_loader, epoch, sam_cfg, casa_module)
            log_val_stats = {**{f'{k}': v for k, v in val_stats.items()},
                     'epoch': epoch}
            if utils.is_main_process():
                with (Path(args.output_dir) / "log_val.txt").open("a") as f:
                    f.write(json.dumps(log_val_stats) + "\n")
                    for k, v in val_stats.items():
                        writer.add_scalar(k, v, epoch)

            if val_stats['val_loss'] < best_val:
                best_val = val_stats['val_loss']
                utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint_bestval.pth'))
                print('Model was saved ! Best Val Loss: {}'.format(best_val))
            else:
                print('Model was not saved ! Best Val Loss: {}'.format(best_val))
        
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and (epoch % args.saveckp_freq == 0) and epoch:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                for k, v in train_stats.items():
                    writer.add_scalar(k, v, epoch)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Alice', parents=[get_args_parser()])
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_Alice(args)