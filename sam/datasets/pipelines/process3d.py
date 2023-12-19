# Xiaoyu Bai, alibaba-inc

from mmdet.datasets.builder import PIPELINES
import torchio as tio
import numpy as np
from mmdet.datasets.pipelines import Compose
import copy
import torch
import os
import nibabel as nib


class Crop_mod(tio.Crop):
    def apply_transform(self, sample):
        low = self.bounds_parameters[::2]
        high = self.bounds_parameters[1::2]
        index_ini = low
        index_fin = high
        # index_fin = np.array(sample.spatial_shape) - high
        for image in self.get_images(sample):
            new_origin = nib.affines.apply_affine(image.affine, index_ini)
            new_affine = image.affine.copy()
            new_affine[:3, 3] = new_origin
            i0, j0, k0 = index_ini
            i1, j1, k1 = index_fin
            image.set_data(image.data[:, i0:i1, j0:j1, k0:k1].clone())
            image.affine = new_affine
        return sample


def get_range(mask, margin=0):
    """Get up, down, left, right extreme coordinates of a binary mask"""
    idx = np.nonzero(mask)
    u = max(0, idx[0].min() - margin)
    d = min(mask.shape[0] - 1, idx[0].max() + margin)
    l = max(0, idx[1].min() - margin)
    r = min(mask.shape[1] - 1, idx[1].max() + margin)
    return [u, d, l, r]


def meshgrid3d(shape, spacing):
    y_ = np.linspace(0., (shape[1] - 1) * spacing[0], shape[1])
    x_ = np.linspace(0., (shape[2] - 1) * spacing[1], shape[2])
    z_ = np.linspace(0., (shape[3] - 1) * spacing[2], shape[3])

    y, x, z = np.meshgrid(x_, y_, z_, indexing='ij')
    return [y, x, z]


def makelabel(shape):
    all_ = np.ones((1, shape[1] * shape[2] * shape[3]))
    all = all_.reshape(-1, shape[1], shape[2], shape[3])
    return all


@PIPELINES.register_module()
class RandomElasticDeformation():
    def __call__(self, data):
        subject = data['subject']
        random_elastic_transform = tio.RandomElasticDeformation()
        subject = random_elastic_transform(subject)
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class MultiBranch(object):
    def __init__(self, **transform_group):
        self.transform_group = {k: Compose(v) for k, v in transform_group.items()}

    def __call__(self, data):
        multi_results = []
        for k, v in self.transform_group.items():
            res = v(copy.deepcopy(data))
            if res is None:
                return None
            # res["img_metas"]["tag"] = k
            multi_results.append(res)
        return multi_results


@PIPELINES.register_module()
class LoadTestTioImage:
    """Load image as torchio subject.
    """

    def __init__(self, re_orient=True, landmark=False):
        self.re_orient = re_orient
        self.landmark = landmark

    def __call__(self, data):
        # time1 = time.time()
        data_path = data['data_path']
        image_fn = data['image_fn']
        if self.landmark:  # the fold structure of ChestCT landmark dataset is /fold/casefold/casenii
            img_tio = tio.ScalarImage(os.path.join(data_path, image_fn.split('.', 1)[0], image_fn))
        else:
            img_tio = tio.ScalarImage(data_path + image_fn)
        if self.re_orient:
            img_data = img_tio.data
            img_tio.data = img_data.permute(0, 2, 1, 3)
            img_tio.affine = np.array(
                [img_tio.affine[1, :], img_tio.affine[0, :], img_tio.affine[2, :], img_tio.affine[3, :]])
        img_tio_shape = img_tio.data.shape  # tio use lazy loading, so we read image shape from data.shape to force it load image here
        img_tio_spacing = img_tio.spacing
        subject = tio.Subject(
            image=img_tio
        )
        data['subject'] = subject
        # time2 = time.time()
        # print('load image info time cost:',time2-time1)
        return data


@PIPELINES.register_module()
class LoadTioImage:
    """Load image as torchio subject.
    """

    def __init__(self, re_orient=True):
        self.re_orient = re_orient

    def __call__(self, data):
        # time1 = time.time()
        data_path = data['data_path']
        image_fn = data['image_fn']
        img_tio = tio.ScalarImage(os.path.join(data_path, image_fn))
        # for NIH lymph node dataset covert orientation to PLS+
        if self.re_orient:
            img_data = img_tio.data
            img_tio.data = img_data.permute(0, 2, 1, 3)
            img_tio.affine = np.array(
                [img_tio.affine[1, :], img_tio.affine[0, :], img_tio.affine[2, :], img_tio.affine[3, :]])
        img_tio_shape = img_tio.data.shape  # tio use lazy loading, so we read image shape from data.shape to force it load image here
        img_tio_spacing = img_tio.spacing
        meshs = meshgrid3d(img_tio_shape, img_tio_spacing)
        labels = makelabel(img_tio_shape)
        subject = tio.Subject(
            image=img_tio,
            label_tio_y=tio.ScalarImage(
                tensor=meshs[0].reshape(1, img_tio_shape[1], img_tio_shape[2], img_tio_shape[3]),
                affine=img_tio.affine),
            label_tio_x=tio.ScalarImage(
                tensor=meshs[1].reshape(1, img_tio_shape[1], img_tio_shape[2], img_tio_shape[3]),
                affine=img_tio.affine),
            label_tio_z=tio.ScalarImage(
                tensor=meshs[2].reshape(1, img_tio_shape[1], img_tio_shape[2], img_tio_shape[3]),
                affine=img_tio.affine),
            labels_map=tio.LabelMap(tensor=labels, affine=img_tio.affine)
        )
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class ExtraAttrs(object):
    def __init__(self, **attrs):
        self.attrs = attrs

    def __call__(self, data):
        for k, v in self.attrs.items():
            assert k not in data
            data[k] = v
        return data


@PIPELINES.register_module()
class RandomBlur3d():
    def __init__(self, std=(0, 4)):
        self.std = std

    def __call__(self, data):
        subject = data['subject']
        image = subject.image
        randomblur_transform = tio.RandomBlur(std=self.std)
        image = randomblur_transform(image)
        subject.image.data = image.data
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class RandomNoise3d():
    def __init__(self, std=(0, 0.25)):
        self.std = std

    def __call__(self, data):
        subject = data['subject']
        image = subject.image
        randomnoise_transform = tio.RandomNoise(std=self.std)
        image = randomnoise_transform(image)
        subject.image.data = image.data
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class Resample():
    def __init__(self, norm_spacing=(2., 2., 2.)):
        self.norm_spacing = norm_spacing

    def __call__(self, data):
        subject = data['subject']
        keys = list(data.keys())
        if 'tag' in keys:
            tag = data['tag']
            spacing_matrix = data['spacing_matrix']
            if tag == 'view1':
                spacing = spacing_matrix[0]
            else:
                spacing = spacing_matrix[1]
        else:
            spacing = np.array(self.norm_spacing)
        resample_transform = tio.Resample(target=spacing)
        # subject = resample_transform(subject)
        # img_data = subject['image'].data
        # if (img_data[0, :, :, -1] == 0).all():
        #     crop_info = np.array(subject.shape[1:]).astype(int)
        #     crop_info[-1] = crop_info[-1] - 1
        #     crop_transform = Crop_mod([0, crop_info[0], 0, crop_info[1], 0, crop_info[2]])  # y1y2x1x2z1z2
        #     subject = crop_transform(subject)
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class GenerateMeshGrid():
    def __call__(self, data):
        sub = data['subject']
        image = sub.image.data
        y_sub = sub.label_tio_y.data
        x_sub = sub.label_tio_x.data
        z_sub = sub.label_tio_z.data
        pos_mm = torch.cat((y_sub, x_sub, z_sub), axis=0)
        # pos1_mm = pos1_mm.transpose((1, 2, 3, 0))
        sub_valid = sub.labels_map.data
        data['img'] = image.permute(0, 3, 1, 2)
        data['meshgrid'] = pos_mm.permute(0, 3, 1, 2)
        data['valid'] = sub_valid.permute(0, 3, 1, 2)
        data['spacing'] = sub.image.spacing
        return data


@PIPELINES.register_module()
class GenerateMetaInfo():
    def __call__(self, data):
        data['filename'] = data['image_fn']
        keys = list(data.keys())
        if 'tag' in keys:
            tag = data['tag']
            if tag == 'view1':
                data['crop_info'] = data['crop_matrix_origin'][0:2, :]
            else:
                data['crop_info'] = data['crop_matrix_origin'][2:, :]
        if 'img' not in keys:
            sub = data['subject']
            image = sub.image.data
            data['img'] = [image.permute(0, 3, 1, 2)]

        return data


@PIPELINES.register_module()
class ComputeCorrespond():
    def __init__(self, shape=(96, 96, 24)):
        self.shape = shape

    def __call__(self, data):
        subject = data['subject']
        resize_transform = tio.Resize(self.shape)
        subject = resize_transform(subject)
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class Resize3d():
    def __init__(self, shape=(96, 96, 24)):
        self.shape = shape

    def __call__(self, data):
        subject = data['subject']
        resize_transform = tio.Resize(self.shape)
        subject = resize_transform(subject)
        data['subject'] = subject

        return data


@PIPELINES.register_module()
class Crop100():
    def __init__(self, margin=5, z_half=False):
        self.margin = margin
        self.z_half = z_half

    def __call__(self, data):
        subject = data['subject']
        shape = subject.image.shape
        if self.z_half:
            crop_info = (0, shape[1],
                         0, shape[2],
                         shape[3] / 2, shape[3])
        else:
            crop_info = (0, shape[1],
                         0, shape[2],
                         self.margin, shape[3] - self.margin)
        # print(data['image_fn'],shape)
        crop_info = np.array(crop_info).astype(int)
        crop_transform = Crop_mod(crop_info)  # y1y2x1x2z1z2
        subject = crop_transform(subject)
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class CropBackground():
    def __init__(self, thres=-500):
        self.thres = thres

    def __call__(self, data):
        subject = data['subject']
        img = subject.image.data
        # img = img[:,:,:,:-1] # tio resample may generate allzero slice at the end
        mask = img > self.thres
        foreground_region = torch.where(mask == 1)
        crop_info = (foreground_region[1].min(), foreground_region[1].max(),
                     foreground_region[2].min(), foreground_region[2].max(),
                     foreground_region[3].min(),
                     foreground_region[3].max())
        crop_info = np.array(crop_info).astype(int)
        crop_transform = Crop_mod(crop_info)  # y1y2x1x2z1z2
        subject = crop_transform(subject)
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class Crop():
    def __init__(self, switch='origin', fix_size=(96, 96, 32)):
        self.switch = switch
        self.fix_size = fix_size
        assert self.switch in ['origin', 'fix']

    def __call__(self, data):
        # time1 = time.time()
        subject = data['subject']
        crop_matrix_origin = data['crop_matrix_origin']

        tag = data['tag']
        if tag == 'view1':
            view = 0
        else:
            view = 2
        if self.switch == 'origin':
            crop_info = (crop_matrix_origin[view][0],
                         crop_matrix_origin[view + 1][0],
                         crop_matrix_origin[view][1],
                         crop_matrix_origin[view + 1][1],
                         crop_matrix_origin[view][2],
                         crop_matrix_origin[view + 1][2])
        else:
            subject_shape = subject.shape
            assert subject_shape[1] >= 96, print(data['image_fn'])

            crop_info = (int(subject_shape[1] / 2) - self.fix_size[0] / 2,
                         int(subject_shape[1] / 2) + self.fix_size[0] / 2,
                         int(subject_shape[2] / 2) - self.fix_size[1] / 2,
                         int(subject_shape[2] / 2) + self.fix_size[1] / 2,
                         int(subject_shape[3] / 2) - self.fix_size[2] / 2,
                         int(subject_shape[3] / 2) + self.fix_size[2] / 2)
            crop_info = np.array(crop_info).astype(int)
        crop_transform = Crop_mod(crop_info)  # y1y2x1x2z1z2
        subject = crop_transform(subject)
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class RandomAffine3d():
    def __init__(self, scales=0, degrees=(0, 0, 0, 0, -3, 3), translation=0):
        self.scales = scales
        self.degrees = degrees
        self.translation = translation

    def __call__(self, data):
        subject = data['subject']
        random_affine_transform = tio.RandomAffine(scales=self.scales, degrees=self.degrees,
                                                   translation=self.translation)
        subject = random_affine_transform(subject)
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class RescaleIntensity():
    def __init__(self, out_min_max=(0, 255), in_min_max=(-1024, 3071)):
        self.out_min_max = out_min_max
        self.in_min_max = in_min_max

    def __call__(self, data):
        subject = data['subject']
        img = subject.image.data
        img[img < -1024] = -1024
        img[img > 3071] = 3071
        subject.image.data = img
        rescale_transform = tio.RescaleIntensity(out_min_max=self.out_min_max, in_min_max=self.in_min_max)
        image = rescale_transform(subject.image)
        subject.image.data = image.data - 50.
        data['subject'] = subject
        return data


@PIPELINES.register_module()
class ComputeAugParam():
    def __init__(self, scale=(0.8, 1.2), standard_spacing=(2., 2., 2.), patch_size=(96, 96, 32)):
        self.scale = scale
        self.standard_spacing = standard_spacing
        self.patch_size = patch_size

    def __call__(self, data):
        view1_scale = np.random.uniform(self.scale[0], self.scale[1])
        view2_scale = np.random.uniform(self.scale[0], self.scale[1])
        # view2_scale = view1_scale
        origin_spacing = data['subject'].image.spacing
        origin_size = data['subject'].image.shape
        # print(origin_size)
        standard_size = [np.floor(origin_spacing[0] / self.standard_spacing[0] * origin_size[1]).astype(int),
                         np.floor(origin_spacing[1] / self.standard_spacing[1] * origin_size[2]).astype(int),
                         np.floor(origin_spacing[2] / self.standard_spacing[2] * origin_size[3]).astype(int)]
        view1_patch_size = np.ceil(np.array(self.patch_size) * view1_scale).astype(int)  # y,x,z
        view2_patch_size = np.ceil(np.array(self.patch_size) * view2_scale).astype(int)
        affine_standard_to_origin = np.array([self.standard_spacing[0] / origin_spacing[0], 0, 0, 0,
                                              0, self.standard_spacing[1] / origin_spacing[1], 0, 0,
                                              0, 0, self.standard_spacing[2] / origin_spacing[2], 0,
                                              0, 0, 0, 1]).reshape(4, 4).transpose()
        view1_crop_size = np.ceil(view1_patch_size * 1.05).astype(int)
        view2_crop_size = np.ceil(view2_patch_size * 1.05).astype(int)

        margin_x = 1
        margin_y = 1
        margin_z = 1
        rangex_min = margin_x
        rangex_max = standard_size[1] - margin_x
        rangey_min = margin_y
        rangey_max = standard_size[0] - margin_y
        rangez_min = margin_z
        rangez_max = standard_size[2] - margin_z

        y_interval = rangey_max - rangey_min - (view1_crop_size[0] + view2_crop_size[0])
        x_interval = rangex_max - rangex_min - (view1_crop_size[1] + view2_crop_size[1])
        z_interval = rangez_max - rangez_min - (view1_crop_size[2] + view2_crop_size[2])

        if y_interval >= 2:
            p = np.random.rand(1)
            if p <= 0.8:
                intersection_y = np.random.randint(20, min(view1_crop_size[0], view2_crop_size[0]) - 2)
            else:
                intersection_y = np.random.randint(-y_interval + 1, -y_interval + 10)
        else:
            intersection_y = np.random.randint(-y_interval + 1, min(view1_crop_size[0], view2_crop_size[0]) - 1)

        if x_interval >= 2:
            p = np.random.rand(1)
            if p <= 0.8:
                intersection_x = np.random.randint(20, min(view1_crop_size[1], view2_crop_size[1]) - 2)
            else:
                intersection_x = np.random.randint(-x_interval + 1, -x_interval + 10)
        else:
            intersection_x = np.random.randint(-x_interval + 5, min(view1_crop_size[1], view2_crop_size[1]) - 2)

        if z_interval > 2:
            p = np.random.rand(1)
            if p <= 0.8:
                intersection_z = np.random.randint(5, min(view1_crop_size[2], view2_crop_size[2]) - 2)
            else:
                intersection_z = np.random.randint(-z_interval + 1, -1)
        else:
            intersection_z = np.random.randint(-z_interval + 1, min(view1_crop_size[2], view2_crop_size[2]))

        overall_crop_size_y = (view1_crop_size[0] + view2_crop_size[0]) - intersection_y
        overall_crop_size_x = (view1_crop_size[1] + view2_crop_size[1]) - intersection_x
        overall_crop_size_z = (view1_crop_size[2] + view2_crop_size[2]) - intersection_z

        overall_x_min_left_up_corner = rangex_min
        overall_x_max_left_up_corner = rangex_max - overall_crop_size_x
        overall_y_min_left_up_corner = rangey_min
        overall_y_max_left_up_corner = rangey_max - overall_crop_size_y
        overall_z_min_left_up_corner = rangez_min
        overall_z_max_left_up_corner = rangez_max - overall_crop_size_z
        assert overall_x_min_left_up_corner <= overall_x_max_left_up_corner and \
               overall_y_min_left_up_corner <= overall_y_max_left_up_corner and \
               overall_z_min_left_up_corner <= overall_z_max_left_up_corner, \
            data['image_fn']

        overall_crop_left_up = (np.random.randint(overall_y_min_left_up_corner, overall_y_max_left_up_corner),
                                np.random.randint(overall_x_min_left_up_corner, overall_x_max_left_up_corner),
                                np.random.randint(overall_z_min_left_up_corner, overall_z_max_left_up_corner))
        overall_crop_left_up = np.asarray(overall_crop_left_up)

        view1_x_min = 0
        view1_x_max = overall_crop_size_x - view1_crop_size[1]
        view1_y_min = 0
        view1_y_max = overall_crop_size_y - view1_crop_size[0]
        view1_z_min = 0
        view1_z_max = overall_crop_size_z - view1_crop_size[2]

        assert view1_z_min < view1_z_max and view1_y_min < view1_y_max and view1_x_min < view1_x_max, data['image_fn']

        view2_x_min = 0
        view2_x_max = overall_crop_size_x - view2_crop_size[1]
        view2_y_min = 0
        view2_y_max = overall_crop_size_y - view2_crop_size[0]
        view2_z_min = 0
        view2_z_max = overall_crop_size_z - view2_crop_size[2]

        assert view2_z_min < view2_z_max and view2_y_min < view2_y_max and view2_x_min < view2_x_max, data['image_fn']

        view1_left_up = (np.random.randint(view1_y_min, view1_y_max),
                         np.random.randint(view1_x_min, view1_x_max),
                         np.random.randint(view1_z_min, view1_z_max))
        view1_left_up = np.asarray(view1_left_up)

        view2_left_up = (np.random.randint(view2_y_min, view2_y_max),
                         np.random.randint(view2_x_min, view2_x_max),
                         np.random.randint(view2_z_min, view2_z_max))
        view2_left_up = np.asarray(view2_left_up)
        view1_y1x1z1_crop = view1_left_up + overall_crop_left_up
        view1_y2x2z2_crop = view1_left_up + overall_crop_left_up + view1_crop_size
        view2_y1x1z1_crop = view2_left_up + overall_crop_left_up
        view2_y2x2z2_crop = view2_left_up + overall_crop_left_up + view2_crop_size

        crop_matrix = np.array([view1_y1x1z1_crop, view1_y2x2z2_crop, view2_y1x1z1_crop, view2_y2x2z2_crop])

        crop_matrix_origin = np.dot(crop_matrix, affine_standard_to_origin[:3, :3])
        crop_matrix_origin[::2, :] = np.floor(crop_matrix_origin[::2, :])
        crop_matrix_origin[1::2, :] = np.ceil(crop_matrix_origin[1::2, :])
        crop_matrix_origin = crop_matrix_origin.astype(int)
        view1_spacing = np.array(self.standard_spacing) * view1_scale
        view2_spacing = np.array(self.standard_spacing) * view2_scale
        spacing_matrix = np.array([view1_spacing, view2_spacing])

        data['crop_matrix_origin'] = crop_matrix_origin
        data['spacing_matrix'] = spacing_matrix
        return data
