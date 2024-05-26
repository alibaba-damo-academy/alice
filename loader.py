import random
import math
import numpy as np
import os

from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    RandSpatialCropSamplesd,
    SpatialPadd,
    NormalizeIntensityd,
    AsChannelFirstd,
    NormalizeIntensityd,
    RandFlipd,
    RandGaussianNoised,
    ThresholdIntensityd,
    Rand3DElastic,
    SpatialCropd,
)

from monai.data import (
    CacheDataset,
    SmartCacheDataset,
    load_decathlon_datalist,
    DataLoader,
    Dataset,
    DistributedSampler
)


def get_loader(args):
    datadir = args.data_dir
    #jsonlist = os.path.join(datadir, args.json_list)
    jsonlist = args.json_list
    num_workers = args.num_workers
    
    #num_none_list = [669, 966, 1567]
    
    new_datalist = []
    datalist = load_decathlon_datalist(jsonlist, False, "training", base_dir=datadir)
    for item in datalist:
        item_name = ''.join(item['image']).split('.')[0].split('/')[-2]
        item_num = int(''.join(item_name).split('_')[1])
        #if item_num in num_none_list:
            #continue
        
        item_dict = {'image': item['image'], 'name': item_name}
        new_datalist.append(item_dict)
    
    new_vallist = []
    vallist = load_decathlon_datalist(jsonlist, False, "validation", base_dir=datadir)
    for item in vallist:
        item_name = ''.join(item['image']).split('.')[0].split('/')[-2]
        item_num = int(''.join(item_name).split('_')[1])
        #if item_num in num_none_list:
            #continue
        
        item_dict = {'image': item['image'], 'name': item_name}
        new_vallist.append(item_dict)
    
    datalist = new_datalist
    val_files = new_vallist
    
    print('Dataset all training: number of data: {}'.format(len(datalist)))
    print('Dataset all validation: number of data: {}'.format(len(val_files)))

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAI", as_closest_canonical=True),
            ThresholdIntensityd(keys=["image"], 
                                threshold=args.a_max, 
                                above=False, 
                                cval=args.a_max, 
                                allow_missing_keys=False),
            ThresholdIntensityd(keys=["image"], 
                                threshold=args.a_min, 
                                above=True, 
                                cval=args.a_min, 
                                allow_missing_keys=False),
            
            ScaleIntensityRanged(keys=["image"],
                                 a_min=args.a_min,
                                 a_max=args.a_max,
                                 b_min=args.b_min,
                                 b_max=args.b_max,
                                 clip=True),
            # SpatialPadd(keys="image", spatial_size=[args.roi_x,
            #                                         args.roi_y,
            #                                         args.roi_z]),
            # CropForegroundd(keys=["image"], source_key="image", k_divisible=[args.roi_x,
            #                                                                  args.roi_y,
            #                                                                  args.roi_z]),
            # RandSpatialCropSamplesd(
            #     keys=["image"],
            #     roi_size=[args.roi_x,
            #               args.roi_y,
            #               args.roi_z],
            #     num_samples=args.global_crops_number,
            #     random_center=True,
            #     random_size=False
            # ),

            ToTensord(keys=["image"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAI", as_closest_canonical=True),
            ThresholdIntensityd(keys=["image"], 
                                threshold=args.a_max, 
                                above=False, 
                                cval=args.a_max, 
                                allow_missing_keys=False),
            ThresholdIntensityd(keys=["image"], 
                                threshold=args.a_min, 
                                above=True, 
                                cval=args.a_min, 
                                allow_missing_keys=False),
            ScaleIntensityRanged(keys=["image"],
                                 a_min=args.a_min,
                                 a_max=args.a_max,
                                 b_min=args.b_min,
                                 b_max=args.b_max,
                                 clip=True),
            # SpatialPadd(keys="image", spatial_size=[args.roi_x,
            #                                         args.roi_y,
            #                                         args.roi_z]),
            # CropForegroundd(keys=["image"], source_key="image", k_divisible=[args.roi_x,
            #                                                                  args.roi_y,
            #                                                                  args.roi_z]),
            # RandSpatialCropSamplesd(
            #     keys=["image"],
            #     roi_size=[args.roi_x,
            #               args.roi_y,
            #               args.roi_z],
            #     num_samples=args.global_crops_number,
            #     random_center=True,
            #     random_size=False
            # ),
            ToTensord(keys=["image"]),
        ]
    )

    if args.normal_dataset:
        print('Using Normal dataset')
        dataset = Dataset(data=datalist, transform=train_transforms)

    elif args.smartcache_dataset:
        print('Using SmartCacheDataset')
        dataset = SmartCacheDataset(data=datalist,
                                    transform=train_transforms,
                                    replace_rate=1,
                                    cache_rate=0.1)

    else:
        print('Using MONAI Cache Dataset')
        dataset = CacheDataset(data=datalist,
                               transform=train_transforms,
                               cache_rate=1,
                               num_workers=num_workers)
    
    if args.distributed:
        train_sampler = DistributedSampler(dataset=dataset,
                                           even_divisible=True,
                                           shuffle=True)
    else:
        train_sampler = None
    
    train_loader = DataLoader(dataset,
                              batch_size=args.batch_size,
                              num_workers=num_workers,
                              sampler=train_sampler,
                              drop_last=True)
    
    val_ds = SmartCacheDataset(data=val_files,
                               transform=val_transforms,
                               replace_rate=1,
                               cache_rate=0.1)

    val_loader = DataLoader(val_ds,
                            batch_size=args.batch_size,
                            num_workers=num_workers,
                            shuffle=False,
                            drop_last=True)

    return train_loader, val_loader