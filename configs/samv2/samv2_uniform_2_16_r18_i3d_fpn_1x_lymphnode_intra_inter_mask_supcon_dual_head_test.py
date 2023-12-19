_base_ = './samv2_uniform_2_16_r18_i3d_fpn_1x_lymphnode_intra_inter_mask_supcon_dual_head.py'
gpu_ids = [0]
norm_spacing = [2., 2., 2.]
local_emb_stride = 2
model = dict(
    test_cfg=dict(
        output_embedding=True
    ))

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
)
semantic = True

fp16 = dict(loss_scale="dynamic")
spacing_drop = True  # when original spacing is smaller than 1/2 of norm_spacing, directly drop half pixels to save time
multi_pt_infer = 0
# use the surrounding points of each template point to assist matching.
# Seems not more accurate in phase classification. Set to 0 to disable.