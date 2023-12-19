# Xiaoyu Bai, NWPU. rearrange the ignore computation into useindex, save computation cost
import os.path
import warnings

import torch
from mmdet.models.builder import DETECTORS, build_backbone, build_neck
from mmdet.models.detectors.base import BaseDetector
import torch.nn.functional as F
import time
import pickle


class nodes():
    def __init__(self,
                 source,
                 target,
                 view='view1'):
        self.source = source
        self.target = target
        self.view = view
        self.ignore_list = []
        self.source_ignore_list = []

    def add_ignore(self, target_list):
        if self.target in target_list:
            target_list.remove(self.target)
            self.ignore_list = target_list

    def add_source_ignore(self, source_list):
        if self.source in source_list:
            source_list.remove(self.source)
        self.source_ignore_list = source_list


@DETECTORS.register_module()
class Sam(BaseDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 read_out_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(Sam, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.backbone.init_weights()
        if neck is not None:
            self.neck = build_neck(neck)
        self.criterion = torch.nn.CrossEntropyLoss().cuda()

        self.read_out_head = build_neck(read_out_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        out1 = self.neck(x[:3])[0]
        out2 = self.read_out_head([x[-1]])[0]
        out1 = F.normalize(out1, dim=1)
        out2 = F.normalize(out2, dim=1)
        out1 = out1.type(torch.half)
        out2 = out2.type(torch.half)
        return [out1, out2]

    def meshgrid3d(self, shape):
        z_ = torch.linspace(0., shape[0] - 1, shape[0])
        y_ = torch.linspace(0., shape[1] - 1, shape[1])
        x_ = torch.linspace(0., shape[2] - 1, shape[2])
        z, y, x = torch.meshgrid(z_, y_, x_)
        return torch.stack((z, y, x), 3)

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        x = self.extract_feat(img)
        outs = []
        out1 = x[0]#.data.cpu().numpy()
        out2 = x[1]#.data.cpu().numpy()
        outs = [out1, out2]
        if not self.test_cfg.output_embedding:
            if not os.path.exists(self.test_cfg.save_path):
                os.mkdir(self.test_cfg.save_path)
            outfilename = self.test_cfg.save_path + \
                          img_metas[0]['filename'].split('.', 1)[0] + '.pkl'
            f = open(outfilename, 'wb')
            pickle.dump(outs, f)
            return [
                x[0][0, 0, 0, 0, 0].data.cpu()]  # we have saved the data into harddisk, this is just for fit the code
        else:
            return outs

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.simple_test(imgs, img_metas, **kwargs)
