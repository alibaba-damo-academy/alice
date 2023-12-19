_base_ = './sam_r18_i3d_fpn_1x_deeplesion.py'

view1_pipline = [
    dict(type="ExtraAttrs", tag="view1"),
    dict(type="Crop"),
    # dict(type="RandomAffine3d"),
    # dict(type="RandomElasticDeformation"),
    dict(type="Resample"),
    dict(type="Crop", switch='fix'),
    dict(type='RescaleIntensity'),
    dict(type="RandomNoise3d"),
    # dict(type="RandomBlur3d"),
    dict(type="GenerateMeshGrid"),
    dict(type="GenerateMetaInfo"),
    dict(type='DefaultFormatBundle3d'),
    dict(type='Collect3d',
         keys=['img', 'meshgrid', 'valid'],
         meta_keys=(
             "filename",
             "tag",
             "crop_info"
         ), )
]
view2_pipline = [
    dict(type="ExtraAttrs", tag="view2"),
    dict(type="Crop"),
    # dict(type="RandomAffine3d"),
    # dict(type="RandomElasticDeformation"),
    dict(type="Resample"),
    dict(type="Crop", switch='fix'),
    dict(type='RescaleIntensity'),
    dict(type="RandomNoise3d"),
    # dict(type="RandomBlur3d"),
    dict(type="GenerateMeshGrid"),
    dict(type="GenerateMetaInfo"),
    dict(type='DefaultFormatBundle3d'),
    dict(type='Collect3d',
         keys=['img', 'meshgrid', 'valid'],
         meta_keys=(
             "filename",
             "tag",
             "crop_info"

         ), )
]
train_pipeline = [
    dict(type='LoadTioImage'),
    # dict(type='RescaleIntensity'),
    # dict(type='Crop100'),
    dict(type='CropBackground'),
    dict(type='ComputeAugParam'),
    dict(
        type="MultiBranch", view1=view1_pipline, view2=view2_pipline
    ),
]
test_pipeline = [
    dict(type='LoadTestTioImage', landmark=True),  # when do evaluation on ChestCT landmark,set it to true
    dict(type='Resample'),
    dict(type='RescaleIntensity'),
    dict(type="GenerateMetaInfo"),
    # dict(type='DefaultFormatBundle3d'),
    dict(type='Collect3d',
         keys=['img'])
]

lymphnode_data_root = '/data/sdb/baixiaoyu/lymphnode/nii/'
anno_root = '/data/sdb/baixiaoyu/mmdet/data/'
dlt_eval_data_root = '/data/sdb/baixiaoyu/Deeplesion/Images_nifti/'
landmark_eval_data_root = '/data/sdc/baixiaoyu/lankmark/CT_nifti_ln_instance/'
deeplesion_data_root = '/data/sdb/baixiaoyu/Deeplesion/nii/'
luna_data_root = '/data/sdb/baixiaoyu/luna/nii/'
anno_root = '/data/sdb/baixiaoyu/mmdet/data/'

lymphnode_set = dict(
    type='Dataset3dsam',
    multisets=True,
    data_dir=lymphnode_data_root,
    index_file=anno_root + 'lymphnode_filename.csv',
    pipeline=train_pipeline,
)

deeplesion_set = dict(
    type='Dataset3dsam',
    multisets=True,
    data_dir=deeplesion_data_root,
    index_file=anno_root + 'deeplesion_filename.csv',
    pipeline=train_pipeline,
)

luna_set = dict(
    type='Dataset3dsam',
    multisets=True,
    data_dir=luna_data_root,
    index_file=anno_root + 'luna_filename.csv',
    pipeline=train_pipeline,
)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=12,
    train=dict(
        type='ConcatDataset',
        datasets=[lymphnode_set, deeplesion_set, luna_set]
    ),
    val=dict(
        data_dir=landmark_eval_data_root,
        index_file=anno_root + 'landmark.csv',
        pipeline=test_pipeline,
    ),
    test=dict(
        data_dir=landmark_eval_data_root,
        index_file=anno_root + 'landmark.csv',
        pipeline=test_pipeline,
    ), )
