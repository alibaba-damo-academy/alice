import numpy as np
import utils
import models
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn

def get_sim(target, behaviored):
    attention_distribution = []
    for i in range(behaviored.size(0)):
        attention_score = torch.cosine_similarity(target, behaviored[i].view(1, -1))
        attention_distribution.append(attention_score)
    attention_distribution = torch.Tensor(attention_distribution)
  
    return attention_distribution    


class KLD(nn.Module):
    def forward(self, inputs, targets):
        inputs = F.log_softmax(inputs, dim=1)
        targets = F.softmax(targets, dim=1)
        return F.kl_div(inputs, targets, reduction='batchmean')


class Loss(nn.Module):
    def __init__(self, out_dim, patch_out_dim, ncrops, warmup_teacher_temp, 
                 teacher_temp, warmup_teacher_temp2, teacher_temp2, 
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, 
                 center_momentum=0.9, center_momentum2=0.9,
                 lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0, mim_start_epoch=0):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.mae_loss = torch.nn.MSELoss()

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp2_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2
        )) if mim_start_epoch == 0 else np.concatenate((
            np.ones(mim_start_epoch) * warmup_teacher_temp2,
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch) * teacher_temp2
        ))

    def forward(self, images, student_output, teacher_output, feat1_ali, feat2_ali, masks, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_cls, _, _, student_decoder = student_output
        teacher_cls, _, _, teacher_decoder = teacher_output
        
        student_patch, teacher_patch = feat1_ali, feat2_ali
        # [CLS] and patch for global patches
        # student_cls1 = cls1_ali
        # teacher_cls1 = cls2_ali
        
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ncrops)
        
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ncrops)
        
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ncrops)
        
        # print(student_cls1_c[0][:,:10], teacher_cls1_c[0][:,:10], student_cls2_c[0][:,:10], teacher_cls2_c[0][:,:10])
        
        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
        for q in [0, 1]:
            for v in [0, 1]:
                if v == q:
                    loss2 = torch.sum(-teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1), dim=-1)
                    total_loss2 += loss2.mean()
                    n_loss_terms2 += 1
                else:
                    loss1 = torch.sum(-teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1), dim=-1)
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1
            
        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2
        
        bs, c, h, w, z = images[0].size()
        images_raw = torch.cat([images[0], images[1]], dim=0)
        mask_raw = torch.cat([nn.functional.interpolate(masks[0], size=(h, w, z), mode="nearest").cuda(non_blocking=True), nn.functional.interpolate(masks[1], size=(h, w, z), mode="nearest").cuda(non_blocking=True)], dim=0)

        total_loss3 = self.mae_loss(images_raw, student_decoder) / 2 * self.lambda3

        total_loss = dict(cls=total_loss1, patch=total_loss2, recon=total_loss3, loss=total_loss1 + total_loss2 + total_loss3)
        self.update_center(teacher_cls, teacher_patch)
        # self.update_center(teacher_cls1, teacher_cls2, teacher_patch)    
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
        self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (1 - self.center_momentum2)
#     @torch.no_grad()
#     def update_center(self, teacher_cls1, teacher_cls2, teacher_patch):
#         """
#         Update center used for teacher output.
#         """
#         cls1_center = torch.sum(teacher_cls1, dim=0, keepdim=True)
#         dist.all_reduce(cls1_center)
#         cls1_center = cls1_center / (len(teacher_cls1) * dist.get_world_size())
#         self.center1 = self.center1 * self.center_momentum + cls1_center * (1 - self.center_momentum)
        
#         cls2_center = torch.sum(teacher_cls2, dim=0, keepdim=True)
#         dist.all_reduce(cls2_center)
#         cls2_center = cls2_center / (len(teacher_cls2) * dist.get_world_size())
#         self.center2 = self.center2 * self.center_momentum + cls2_center * (1 - self.center_momentum)

#         patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
#         dist.all_reduce(patch_center)
#         patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
#         self.center3 = self.center3 * self.center_momentum2 + patch_center * (1 - self.center_momentum2)