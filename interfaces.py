from time import time

import numpy as np
import matplotlib.pyplot as plt

import torch
from mmcv import Config
from mmcv.runner import (load_checkpoint, wrap_fp16_model)
from mmdet.models import build_detector
import torch.nn.functional as F
import torchio as tio
from mmcv.parallel import MMDataParallel

from sam.datasets.pipelines import Resample, RescaleIntensity, GenerateMetaInfo, Collect3d
from sam.datasets.collect import collate

def read_info(data_name, norm_spacing=(2., 2., 2.), crop_loc=None, is_MRI=False, to_canonical=True,
               remove_bed=False):
    data_path = '/mnt/data/oss_beijing/jiangyankai/AbdomenAtlas_wEmb/' + data_name + '/ct.nii.gz'
    img_tio = tio.ScalarImage(data_path)
    if to_canonical:
        ToCanonical = tio.ToCanonical()
        img_tio = ToCanonical(img_tio)
    # print(img_tio.spacing)
    if img_tio.orientation == ('R', 'A', 'S'):
        img_data = img_tio.data
        img_tio.data = torch.flip(img_data, (1, 2))
        img_tio.affine = np.array(
            [-img_tio.affine[0, :], -img_tio.affine[1, :], img_tio.affine[2, :], img_tio.affine[3, :]])
    assert img_tio.orientation == ('L', 'P', 'S'), print('right now the image orientation need to be LPS+ ')
    img_data = img_tio.data
    img_tio.data = img_data.permute(0, 2, 1, 3)
    img_tio.affine = np.array(
        [img_tio.affine[1, :], img_tio.affine[0, :], img_tio.affine[2, :], img_tio.affine[3, :]])

    img_tio_shape = img_tio.data.shape
    img_tio_spacing = img_tio.spacing
    #img_tio_affine = img_tio.affine
    norm_ratio = np.array(img_tio_spacing) / np.array(norm_spacing)
    return norm_ratio, img_tio_shape

def init_model(config, checkpoint):
    print('Initializing SAM model ...')
    cfg = Config.fromfile(config)
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    model.eval()
    return model, cfg


def proc_image(im, im_info, cfg):
    # assert np.all(np.reshape(im_info['direction'], (3, 3)) == np.eye(3)), f'unsupported direction!'
    assert np.max(np.abs(np.abs(np.reshape(im_info['direction'], (3, 3))) - np.eye(3))) < .1, f'unsupported direction!'

    if cfg.spacing_drop:
        r = np.floor(np.array(cfg.norm_spacing) / np.array(im_info['spacing']))
        r = np.maximum(1, r).astype(int)
        im = im[r[2] // 2::r[2], r[1] // 2::r[1], r[0] // 2::r[0]]
    else:
        r = np.ones((3,))
    img_data = torch.from_numpy(im).permute(1, 2, 0)[None]
    tio_affine = np.hstack((np.diag(im_info['spacing']) * r, np.array(im_info['origin'])[:, None]))
    tio_affine = np.vstack((tio_affine, [0, 0, 0, 1]))
    norm_ratio = np.array(im_info['spacing']) / np.array(cfg.norm_spacing)
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=img_data, affine=tio_affine),
    )
    data = {}
    data['image_fn'] = im_info['im_path']
    data['subject'] = subject
    resample = Resample()
    data = resample(data)
    rescale = RescaleIntensity()
    data = rescale(data)
    meta_collect = GenerateMetaInfo()
    data = meta_collect(data)
    collects = Collect3d(keys=['img'])
    input = collects(data)
    batch = collate([input])
    return batch, norm_ratio


def get_embedding(model, im, im_info, cfg):
    t = time()
    batched_data, im_norm_ratio = proc_image(im, im_info, cfg)
    print(f"Normalization takes {time() - t:.2f} s,", end=' ')
    t = time()
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **batched_data)
    print(f"embedding generation takes {time() - t:.2f} s")
    if 'semantic' in cfg:
        result = dict(coarse_emb=result[1], fine_emb=result[0], sem_emb=result[2], im_norm_ratio=im_norm_ratio,
                      im_shape=im.shape)
    else:
        result = dict(coarse_emb=result[1], fine_emb=result[0], im_norm_ratio=im_norm_ratio, im_shape=im.shape)
    return result


def find_point_in_vol(query_data, key_data, query_points, cfg):
    if 'semantic' in cfg:
        coarse_query_vec, fine_query_vec, sem_query_vec = extract_point_emb(query_data, query_points, cfg)
        return match_vec_in_vol(coarse_query_vec, fine_query_vec, key_data, cfg, sem_query_vec=sem_query_vec)
    else:
        coarse_query_vec, fine_query_vec = extract_point_emb(query_data, query_points, cfg)
        return match_vec_in_vol(coarse_query_vec, fine_query_vec, key_data, cfg)


def extract_point_emb(query_data, query_points, cfg):
    im_norm_ratio, _ = read_info(query_data[3])
    #query_points = np.array(query_points) * query_data['im_norm_ratio']
    query_points = np.array(query_points) * im_norm_ratio
    query_points = np.floor(query_points / cfg.local_emb_stride).astype(int)
    #coarse_query_vol = query_data['coarse_emb']
    coarse_query_vol = query_data[1]
    #fine_query_vol = query_data['fine_emb']
    fine_query_vol = query_data[0]
    coarse_query_vol = F.interpolate(coarse_query_vol, fine_query_vol.shape[2:], mode='trilinear')
    coarse_query_vol = F.normalize(coarse_query_vol, dim=1)
    if 'semantic' in cfg:
        sem_query_vol = query_data['sem_emb']
    if cfg.multi_pt_infer == 0:
        coarse_query_vec = coarse_query_vol[0, :, query_points[:, 2], query_points[:, 1], query_points[:, 0]].T
        fine_query_vec = fine_query_vol[0, :, query_points[:, 2], query_points[:, 1], query_points[:, 0]].T
        if 'semantic' in cfg:
            sem_query_vec = sem_query_vol[0, :, query_points[:, 2], query_points[:, 1], query_points[:, 0]].T
    else:
        shape = coarse_query_vol.shape[2:]
        coarse_query_vec, fine_query_vec = [], []
        if 'semantic' in cfg:
            sem_query_vec = []

        for i in range(-cfg.multi_pt_infer, cfg.multi_pt_infer + 1):
            zs = np.maximum(0, np.minimum(shape[0] - 1, query_points[:, 2] + i))
            for j in range(-cfg.multi_pt_infer, cfg.multi_pt_infer + 1):
                ys = np.maximum(0, np.minimum(shape[1] - 1, query_points[:, 1] + j))
                for k in range(-cfg.multi_pt_infer, cfg.multi_pt_infer + 1):
                    # if abs(i)+abs(j)+abs(k) > 1:
                    #     continue  # ignore some points, save memory
                    xs = np.maximum(0, np.minimum(shape[2] - 1, query_points[:, 0] + k))
                    coarse_query_vec += [coarse_query_vol[0, :, zs, ys, xs].T]
                    fine_query_vec += [fine_query_vol[0, :, zs, ys, xs].T]
                    if 'semantic' in cfg:
                        sem_query_vec += [sem_query_vol[0, :, zs, ys, xs].T]
        coarse_query_vec = torch.stack(coarse_query_vec)
        fine_query_vec = torch.stack(fine_query_vec)
        if 'semantic' in cfg:
            sem_query_vec = torch.stack(sem_query_vec)
    if 'semantic' in cfg:
        return coarse_query_vec, fine_query_vec, sem_query_vec
    else:
        return coarse_query_vec, fine_query_vec


def match_vec_in_vol(coarse_query_vec, fine_query_vec, key_data, cfg, sem_query_vec=None):
    #coarse_key_vol, fine_key_vol = key_data['coarse_emb'], key_data['fine_emb']
    coarse_key_vol, fine_key_vol = key_data[1], key_data[0]
    sem_key_vol = key_data['sem_emb'] if not sem_query_vec is None else None

    # is it correct to interpolate embeddings? Will it mix neighboring pixels?
    coarse_key_vol = F.interpolate(coarse_key_vol, fine_key_vol.shape[2:], mode='trilinear', align_corners=False)
    coarse_key_vol = F.normalize(coarse_key_vol, dim=1)
    if cfg.multi_pt_infer == 0:
        return match_vec_in_vol_single(coarse_key_vol, fine_key_vol, coarse_query_vec, fine_query_vec, key_data, cfg,
                                       sem_query_vec=sem_query_vec, sem_key_vol=sem_key_vol)
    else:
        return match_vec_in_vol_ensemble(coarse_key_vol, fine_key_vol, coarse_query_vec, fine_query_vec, key_data, cfg,
                                         sem_query_vec=sem_query_vec, sem_key_vol=sem_key_vol)


def match_vec_in_vol_single(coarse_key_vol, fine_key_vol, coarse_query_vec, fine_query_vec, key_data, cfg,
                            sem_query_vec=None, sem_key_vol=None):
    # change to convolution operator in GPU, similar speed w mat mul
    sim_fine = F.conv3d(fine_key_vol, fine_query_vec[:, :, None, None, None])
    sim_coarse = F.conv3d(coarse_key_vol, coarse_query_vec[:, :, None, None, None])
    if not sem_query_vec is None:
        sim_sem = F.conv3d(sem_key_vol, sem_query_vec[:, :, None, None, None])
        sim = (sim_fine[0] + sim_coarse[0] + sim_sem[0]) / 3
    else:
        sim = (sim_fine[0] + sim_coarse[0]) / 2

    # instead of interp emb, interp sim. Its speed and accuracy is similar to interp emb, but has lower sim scores
    # sim_coarse = F.interpolate(sim_coarse, sim_fine.shape[2:], mode='trilinear')
    sim = sim.view(sim.shape[0], -1)

    # compute sim by mat mul
    # dim = coarse_query_vec.shape[1]
    # fine_key_vec = fine_key_vol[0, :, :, :, :].reshape(dim, -1)
    # coarse_key_vec = coarse_key_vol[0, :, :, :, :].reshape(dim, -1)
    # sim_fine = torch.einsum("nc,ck->nk", fine_query_vec, fine_key_vec)
    # sim_coarse = torch.einsum("nc,ck->nk", coarse_query_vec, coarse_key_vec)
    # sim = (sim_fine + sim_coarse)/2

    # don't interp sim to ori image size, but rescale matched points
    ind = torch.argmax(sim, dim=1).cpu().numpy()
    zyx = np.unravel_index(ind, fine_key_vol.shape[2:])
    xyz = np.vstack(zyx)[::-1] * cfg.local_emb_stride + .5  # add .5 to closer to stride center
    im_norm_ratio, _ = read_info(key_data[3])
    #xyz = xyz.T / key_data['im_norm_ratio']
    xyz = xyz.T / im_norm_ratio
    _, im_shape = read_info(key_data[3])
    #xyz = np.minimum(np.round(xyz.astype(int)), np.array(key_data['im_shape'])[::-1] - 1)
    xyz = np.minimum(np.round(xyz.astype(int)), im_shape[1:])

    # interp sim to ori image size, no need to rescale points, maybe more accurate, similar speed, more memory
    # sim = (sim_fine + sim_coarse)/2
    # sim = F.interpolate(sim, key_data['im_shape'], mode='trilinear', align_corners=False)[0]
    # sim = sim.view(sim.shape[0], -1)
    # ind = torch.argmax(sim, dim=1).cpu().numpy()
    # zyx = np.unravel_index(ind, key_data['im_shape'])
    # xyz = np.vstack(zyx)[::-1].T

    max_sims = sim.max(dim=1)[0].cpu().numpy()

    return xyz, max_sims


def match_vec_in_vol_ensemble(coarse_key_vol, fine_key_vol, coarse_query_vec, fine_query_vec, key_data, cfg):
    """Use multiple points around each query point to match, then aggregate matched points"""
    num_ensemble, num_pt, ft_dim = coarse_query_vec.shape

    # fine_query_vec = torch.reshape(fine_query_vec, (-1, ft_dim))
    # coarse_query_vec = torch.reshape(coarse_query_vec, (-1, ft_dim))
    # ensemble each point separately to save memory
    xyzs, max_sims = [], []
    for p in range(num_pt):
        sim = (F.conv3d(fine_key_vol, fine_query_vec[:, p][:, :, None, None, None])
               + F.conv3d(coarse_key_vol, coarse_query_vec[:, p][:, :, None, None, None]))[0] / 2

        # sim = (sim_fine[0] + sim_coarse[0])/2
        sim = sim.view(sim.shape[0], -1)

        ind = torch.argmax(sim, dim=1).cpu().numpy()
        zyx = np.unravel_index(ind, fine_key_vol.shape[2:])
        xyz = np.vstack(zyx)[::-1] * cfg.local_emb_stride + .5  # add .5 to closer to stride center
        #xyz = xyz.T / key_data['im_norm_ratio']
        im_norm_ratio, _ = read_info(key_data[3])
        xyz = xyz.T / im_norm_ratio
        max_sim = sim.max(dim=1)[0].cpu().numpy()

        # average ensemble
        xyz = xyz.mean(axis=0)
        xyzs.append(xyz)
        max_sim = max_sim.mean(axis=0)
        max_sims.append(max_sim)
    xyzs = np.vstack(xyzs)
    #xyzs = np.minimum(np.round(xyzs.astype(int)), np.array(key_data['im_shape'])[::-1] - 1)
    _, im_shape = read_info(key_data[3])
    xyzs = np.minimum(np.round(xyzs.astype(int)), im_shape[1:])
    max_sims = np.hstack(max_sims)

    return xyzs, max_sims
