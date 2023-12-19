import os
import sys
import time
import math
import json
import random
import datetime
import subprocess
import numpy as np
from random import choice
import torch
import torch.distributed as dist
import torch.nn as nn

from collections import defaultdict, deque
from pathlib import Path
from torch import nn
from PIL import ImageFilter, ImageOps, Image, ImageDraw

import cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class PermutePatch(object):
    """
    Apply Patch permutation to the PIL image.
    """
    def __init__(self, psz):
        self.psz = psz

    def __call__(self, img):
        imgs = []
        imgwidth, imgheight = img.size
        for i in range(0, imgheight, self.psz):
            for j in range(0, imgwidth, self.psz):
                box = (j, i, j+self.psz, i+self.psz)
                imgs.append(img.crop(box))
        random.shuffle(imgs)
        new_img = Image.new('RGB', (imgwidth, imgheight))
        k = 0
        for i in range(0, imgheight, self.psz):
            for j in range(0, imgwidth, self.psz):
                new_img.paste(imgs[k], (j, i))
                k += 1
        return new_img

class HideAndSeek(object):
    """
    Apply Patch permutation to the PIL image.
    """
    def __init__(self, ratio, psz):
        self.ratio = ratio
        self.psz = psz

    def __call__(self, img):
        imgwidth, imgheight = img.size 
        numw, numh = imgwidth // self.psz, imgheight // self.psz
        mask_num = int(numw * numh * self.ratio)
        mask_patch = np.random.choice(np.arange(numw * numh), mask_num, replace=False)
        mask_w, mask_h = mask_patch % numh, mask_patch // numh
        # img.save('test1.png')
        draw = ImageDraw.Draw(img)
        for mw, mh in zip(mask_w, mask_h):
            draw.rectangle((mw * self.psz, 
                            mh * self.psz,
                            (mw + 1) * self.psz,
                            (mh + 1) * self.psz), fill="black")
        # img.save('test2.png')
        return img

def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        return
    elif pretrained_weights == 'download':
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights are provided, we load the pretrained weights from {}.".format(url))
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
            return
    elif pretrained_weights == 'supervised':
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "deit_small_patch16_224-cd65a155.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "deit_base_patch16_224-b5f2ef4d.pth"
        if url is not None:
            print("Since no pretrained weights are provided, we load the pretrained weights from {}.".format(url))
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/deit/" + url)
            msg = model.load_state_dict(state_dict['model'], strict=False)
            print('Supervised weights found at {} and loaded with msg: {}'.format(url, msg))
            return
    print("There is no reference weights available for this model => We use random weights.")


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        # args.rank, args.gpu, args.world_size = 0, 0, 1
        args.rank, args.world_size = 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = args.master_port
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

def create_ds_config(args):
    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head=None):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        if head is None:
            self.head = nn.Identity()
        else:
            self.head = head

    def forward(self, x, mask=None, return_backbone_feat=False, 
                **kwargs):
        # convert to list
        if not isinstance(x, list):
            x = [x]
            mask = [mask] if mask is not None else None
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            inp_x = torch.cat(x[start_idx: end_idx])
            if mask is not None:
                inp_m = torch.cat(mask[start_idx: end_idx])
                kwargs.update(dict(mask=inp_m))

            _out = self.backbone(inp_x, **kwargs)
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        cls_token, feat_query, feat_anchor, de_out = output[0], output[1], output[2], output[3]

        clstoken, en_feat, de_feat = self.head(cls_token, feat_query, feat_anchor)
        
        return [clstoken, en_feat, de_feat, de_out]


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

def get_params_groups_dual(model1, model2):
    regularized = []
    not_regularized = []
    for name, param in model1.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
            
    for name, param in model2.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class PCA():
    """
    Class to  compute and apply PCA.
    """
    def __init__(self, dim=256, whit=0.5):
        self.dim = dim
        self.whit = whit
        self.mean = None

    def train_pca(self, cov):
        """
        Takes a covariance matrix (np.ndarray) as input.
        """
        d, v = np.linalg.eigh(cov)
        eps = d.max() * 1e-5
        n_0 = (d < eps).sum()
        if n_0 > 0:
            d[d < eps] = eps

        # total energy
        totenergy = d.sum()

        # sort eigenvectors with eigenvalues order
        idx = np.argsort(d)[::-1][:self.dim]
        d = d[idx]
        v = v[:, idx]

        print("keeping %.2f %% of the energy" % (d.sum() / totenergy * 100.0))

        # for the whitening
        d = np.diag(1. / d**self.whit)

        # principal components
        self.dvt = np.dot(d, v.T)

    def apply(self, x):
        # input is from numpy
        if isinstance(x, np.ndarray):
            if self.mean is not None:
                x -= self.mean
            return np.dot(self.dvt, x.T).T

        # input is from torch and is on GPU
        if x.is_cuda:
            if self.mean is not None:
                x -= torch.cuda.FloatTensor(self.mean)
            return torch.mm(torch.cuda.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)

        # input if from torch, on CPU
        if self.mean is not None:
            x -= torch.FloatTensor(self.mean)
        return torch.mm(torch.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.
         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only
           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj)]

        k = 0;
        ij = 0;
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1 # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]); 
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


def read_image(im_path):
    vol = load_sitk_vol(im_path)
    spacing = vol.GetSpacing()
    origin = vol.GetOrigin()
    direc = vol.GetDirection()
    vol = sitk.GetArrayFromImage(vol)
    if not np.all(np.array(direc) == np.eye(3).ravel()):
        vol, direc = adjust_direction(im_path, vol, direc)
    return vol, dict(im_path=im_path, spacing=spacing, origin=origin, direction=direc)


def load_sitk_vol(im_path):
    if im_path.endswith('.nii') or im_path.endswith('.nii.gz'):
        vol = sitk.ReadImage(im_path)
    else:
        reader = sitk.ImageSeriesReader()
        series = reader.GetGDCMSeriesFileNames(im_path)
        reader.SetFileNames(series)
        vol = reader.Execute()
    return vol


def adjust_direction(fn, vol, direc):
    direc = np.reshape(direc, (3,3))
    print(f"{fn} has direction {direc}.")
    flag = False
    assert np.max(np.abs(np.abs(direc) - np.eye(3))) < .1, f'unsupported direction!'
    for axis in range(3):
        if direc[axis, axis] == -1:
            vol = np.flip(vol, 2-axis)
            direc[axis, axis] = 1
            print(f"axis {axis} is flipped.")
            flag = True
    if flag:
        vol = vol.copy()
        print('Caution: The image has been flipped, the predicted boxes are based on the flipped coordinates (RAI direction)')
    return vol, direc


def windowing(im, win=[-200, 200]):
    """scale intensity from win[0]~win[1] to float numbers"""
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    # im1 *= 255
    return im1


def visualize(q_img, k_img, norm_ratio_1, norm_ratio_2, pt1, pts2, score, save_path=None):
    num_pred = len(score)
    fig, ax = plt.subplots(3, num_pred+1, figsize=(20, 30))
    slice = q_img[pt1[2], :, :]
    ax[0, 0].set_title('query')
    ax[0, 0].imshow(slice, cmap='gray')
    ax[0, 0].plot((pt1[0]), (pt1[1]), 'o', markerfacecolor='none',
                  markeredgecolor="red",
                  markersize=12, markeredgewidth=2)
    slice = q_img[:, pt1[1], :]
    slice = slice[::-1, :]
    ax[1, 0].set_title('query')
    ax[1, 0].imshow(slice, cmap='gray', aspect=norm_ratio_1[2] / norm_ratio_1[0])
    ax[1, 0].plot((pt1[0]), (q_img.shape[0] - pt1[2] - 1), 'o',
                  markerfacecolor='none', markeredgecolor="red",
                  markersize=12, markeredgewidth=2)

    slice = q_img[:, :, pt1[0]]
    slice = slice[::-1, :]
    ax[2, 0].set_title('query')
    ax[2, 0].imshow(slice, cmap='gray', aspect=norm_ratio_1[2] / norm_ratio_1[1])
    ax[2, 0].plot((pt1[1]), (q_img.shape[0] - pt1[2] - 1), 'o',
                  markerfacecolor='none', markeredgecolor="red",
                  markersize=12, markeredgewidth=2)

    for p in range(num_pred):
        slice = k_img[pts2[p, 2], :, :]
        ax[0, p+1].set_title('key')
        ax[0, p+1].imshow(slice, cmap='gray')
        ax[0, p+1].plot((pts2[p, 0]), (pts2[p, 1]), 'o', markerfacecolor='none',
                      markeredgecolor="red",
                      markersize=12, markeredgewidth=2)
    
        slice = k_img[:, pts2[p, 1], :]
        slice = slice[::-1, :]
    
        ax[1, p+1].set_title('key')
        ax[1, p+1].imshow(slice, cmap='gray', aspect=norm_ratio_2[2] / norm_ratio_2[0])
        ax[1, p+1].plot((pts2[p, 0]), (k_img.shape[0] - pts2[p, 2] - 1), 'o',
                      markerfacecolor='none',
                      markeredgecolor="red",
                      markersize=12, markeredgewidth=2)
        slice = k_img[:, :, pts2[p, 0]]
        slice = slice[::-1, :]
        ax[2, p+1].set_title('key')
        ax[2, p+1].imshow(slice, cmap='gray', aspect=norm_ratio_2[2] / norm_ratio_2[1])
        ax[2, p+1].plot((pts2[p, 1]), (k_img.shape[0] - pts2[p, 2] - 1), 'o',
                      markerfacecolor='none',
                      markeredgecolor="red",
                      markersize=12, markeredgewidth=2)
        plt.suptitle(f'score:{score}')
    plt.tight_layout()
    if save_path is None:
        plt.show()  # may be slow
    else:
        plt.savefig(save_path)


def draw_patch(im, box, patch_size=100, color=(0, 255, 0), add_boxes=None, add_color=None):
    im_show = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(im_show, (int(box[0]), int(box[1])),
                  (int(box[2]), int(box[3])), color=color, thickness=1)
    if add_boxes is not None:
        for add_box in add_boxes:
            cv2.rectangle(im_show, (int(add_box[0]), int(add_box[1])),
                          (int(add_box[2]), int(add_box[3])), color=add_color, thickness=1)

    center = (box[:2]+box[2:4])/2
    m = int(patch_size / 2)
    im_show = cv2.copyMakeBorder(im_show, m, m, m, m, cv2.BORDER_CONSTANT, value=0)
    x, y = (center + m).astype(int)
    patch = im_show[y - m: y+m, x-m:x+m, :]
    return patch


def visualize_landmark(vol, spacing, pts, sims, HU_vals=None, save_path=None, sim_th=.6, col_num=1, plane='axial'):
    patches = []
    for p, pt in enumerate(pts):
        if plane == 'axial':
            im = vol[pt[2]]
            im = (windowing(im, [-175, 275]) * 255).astype('uint8')
            patch = draw_patch(im, np.array([pt[0] - 2, pt[1] - 2, pt[0] + 2, pt[1] + 2]), 200)
        else:
            im = vol[:, pt[1]]
            fy = spacing[2]/spacing[0]
            im = (windowing(im, [-175, 275]) * 255).astype('uint8')
            im = cv2.resize(im, None, None, fx=1, fy=fy, interpolation=cv2.INTER_LINEAR)
            patch = draw_patch(im, np.array([pt[0]-2, pt[2]*fy-2, pt[0]+2, pt[2]*fy+2]), 200)
            patch = patch[::-1].copy()  # head-to-foot

        txt = f"{sims[p]:.2f}"
        if HU_vals is not None:
            txt += f" {HU_vals[p]:.0f}"

        color = [0,255,0] if sims[p] > sim_th else [0,0,255]
        patch = cv2.putText(patch, txt, (0, 20), cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=color, thickness=2)
        patches.append(patch)
    if len(patches) % col_num != 0:
        patches.extend([patches[0]*0] * (col_num-len(patches) % col_num))
    im_show = np.vstack([np.hstack(patches[p*col_num:p*col_num+col_num]) for p in range(len(patches)//col_num)])
    im_show = cv2.resize(im_show, None, None, fx=.5, fy=.5, interpolation=cv2.INTER_LINEAR)
    if save_path is not None:
        cv2.imwrite(save_path, im_show)

    return im_show



class MaskGenerator:
    def __init__(self, input_size, mask_patch_size=16, model_patch_size=2, mask_ratio=0.75):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size[0] % self.mask_patch_size == 0 and self.input_size[1] % self.mask_patch_size == 0 and self.input_size[2] % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size_h = self.input_size[0] // self.mask_patch_size
        self.rand_size_w = self.input_size[1] // self.mask_patch_size
        self.rand_size_d = self.input_size[2] // self.mask_patch_size
        
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size_h * self.rand_size_w
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size_h, self.rand_size_w))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        mask = torch.repeat_interleave(mask.unsqueeze(-1), repeats=self.rand_size_d*self.scale, dim=-1)
        
        return mask


def random_mask(args, img_list):
    device = torch.device(f"cuda:{args.local_rank}")
    
    mask_list = []
    mask_patch_size = [8, 16, 32]
    for img in img_list:
        bs, c, h, w, z  = img.size()
            
        mask_generator = MaskGenerator(
        input_size=(h, w, z),
        mask_patch_size=args.patch_size,
        # mask_patch_size=choice(mask_patch_size),
        model_patch_size=2,
        mask_ratio=args.mask_ratio,
        )
            
        mask_volume = mask_generator().type(torch.FloatTensor)
        mask_list.append(mask_volume.unsqueeze(0).unsqueeze(0))
    return mask_list


def data_aug(args, x_s):
    img_n, c, h, w, d = x_s.size()
    x_aug = x_s.detach().clone()
    device = torch.device(f"cuda:{args.local_rank}")
    degree = 30
    random_rotate = transforms.RandomRotation(degree)
    for i in range(img_n):
        x = x_s[i]
        x = x.permute(0, 3, 1, 2)
        x = random_rotate(x).permute(0, 2, 3, 1)
        # x_aug[i] = x.to(device)
        x_aug[i] = x.to(device) + (0.01**0.5)*torch.randn(c, h, w, d).to(device)
    return x_aug


def select_random_points(num_pts, im):
    im = im[0, 0]
    d, w, h = im.shape[0], im.shape[1], im.shape[2]
    pts = []
    iters = 0
    while len(pts) < num_pts:
        pt1 = (np.random.rand(3) * np.array(im.shape)[::-1]).astype(int)
        if im[pt1[2], pt1[1], pt1[0]] > 0 and pt1[0] < h-10 and pt1[1] < w-10 and pt1[2] < d-10:
            pts.append(pt1)
        elif pt1[0] < h-10 and pt1[1] < w-10 and pt1[2] < d-10 and iters >= 10000:
            pts.append(pt1)
        iters += 1
    return np.vstack(pts)


def crop_tensor_new(img, pts, roi_x, roi_y, roi_z):
    img_n, c, h, w, d = img.size()
    pts_x, pts_y, pts_z = pts[0], pts[1], pts[2]
    assert pts_x <= h and pts_y <= w and pts_z <= d
    
    hl, hr = max(0, pts_x-roi_x//2), min(pts_x+roi_x//2, h)
    wl, wr = max(0, pts_y-roi_y//2), min(pts_y+roi_y//2, w)
    dl, dr = max(0, pts_z-roi_z//2), min(pts_z+roi_z//2, d)
    
    pad_hl, pad_hr = max(0, -pts_x+roi_x//2), max(0, pts_x+roi_x//2-h)
    pad_wl, pad_wr = max(0, -pts_y+roi_y//2), max(0, pts_y+roi_y//2-w)
    pad_dl, pad_dr = max(0, -pts_z+roi_z//2), max(0, pts_z+roi_z//2-d)
    
    padding = nn.ConstantPad3d((pad_dl, pad_dr, pad_wl, pad_wr, pad_hl, pad_hr), 0)
    
    img_crop = img[:,:, hl:hr, wl:wr, dl:dr]
    img_pad = padding(img_crop)

    return img_pad


def data_aug(args, x_s):
    img_n, c, h, w, d = x_s.size()
    x_aug = x_s.detach().clone()
    device = torch.device(f"cuda:{args.local_rank}")
    degree = 15
    random_rotate = transforms.RandomRotation(degree)
    for i in range(img_n):
        x = x_s[i]
        x = x.permute(0, 3, 1, 2)
        x = random_rotate(x).permute(0, 2, 3, 1)
        x_aug[i] = x.to(device) + (0.01**0.5)*torch.randn(c, h, w, d).to(device)
    return x_aug