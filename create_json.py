import json
import os
from tqdm import tqdm
raw_path = '/mnt/data/oss_beijing/jiangyankai/AbdomenAtlas_wEmb/'
save_path = '/mnt/workspace/jiangyankai/Alice_code/datasets/pretrainset_test.json'

json_dict = {}
train_list = []
val_list = []

path = os.listdir(raw_path)
path[:].sort(key=lambda x: int(x.split('.')[0].split('_')[1]))

for data in tqdm(path[0:20]):
    item_dict = {}
    item_dict['image'] = 'AbdomenAtlas_wEmb/' + f'{data}' + '/ct.nii.gz'
    num_data = ''.join(data).split('_')[1]
    item_dict['label'] = item_dict['image']
    train_list.append(item_dict)

json_dict['training'] = train_list

# print(path)
for val_data in tqdm(path[20:41]):
    val_dict = {}
    val_dict['image'] = 'AbdomenAtlas_wEmb/' + f'{val_data}' + '/ct.nii.gz'
    num_val_data = ''.join(val_data).split('_')[1]
    val_dict['label'] = val_dict['image']
    val_list.append(val_dict)

json_dict['validation'] = val_list

with open(save_path, "w") as f:
    json.dump(json_dict, f)