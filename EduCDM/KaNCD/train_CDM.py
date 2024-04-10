'''
Author: wuyongyu wuyongyu@atomecho.xyz
Date: 2024-01-13 23:34:47
LastEditors: wuyongyu wuyongyu@atomecho.xyz
LastEditTime: 2024-02-02 23:48:19
FilePath: /zjq/zhangdacao/pisa/EduCDM/EduCDM/IRT/GD/train_CDM.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
import sys
sys.path.append('/zjq/zhangdacao/pisa/datasets')
from utils import *
from KaNCD_CDM import KaNCD

import json
from torch.utils.data import DataLoader
import random
import numpy as np
import argparse

seed = 2014
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--dataset_index', type=int, default=0, help='dataset index', required=True)
    parser.add_argument('--save_path', type=str, help='Path to save the model', required=True)
    parser.add_argument('--sensitive_name', type=str, default='escs', help='sensitive_type', required=True)
    parser.add_argument('--mode', type=str, default='ours', help='mode of train', required=True)

    return parser.parse_args()

args = parse_args()

with open('/zjq/zhangdacao/pisa/datasets/dataset_info.json', 'r', encoding='utf-8') as file:
    dataset_info = json.load(file)
print(dataset_info)
root = '/zjq/zhangdacao/pisa/datasets'
country = dataset_info[args.dataset_index]['country']
path = os.path.join(root, country)
print(path)
train_dataset = MyDataset(path, split='train', sensitive_name=args.sensitive_name)
val_dataset = MyDataset(path, split='val', sensitive_name=args.sensitive_name)
test_dataset = MyDataset(path, split='test', sensitive_name=args.sensitive_name)
print(len(train_dataset), len(val_dataset), len(test_dataset))
print(train_dataset[1])
user_num = dataset_info[args.dataset_index]['user_num']
item_num = dataset_info[args.dataset_index]['item_num']
print(user_num, item_num)

batch_size = 512
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

import os
import shutil
save_path = args.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)
if os.path.exists(save_path) and os.path.isdir(save_path) and os.listdir(save_path):
    shutil.rmtree(save_path)
    os.makedirs(save_path)
model = KaNCD(knowledge_n=68, exer_n=item_num, student_n=user_num, save_path=save_path, sensitive_name=args.sensitive_name, dataset_index=args.dataset_index, mode=args.mode)

model.train(train_data=train_loader, test_data=val_loader, epoch=100, device="cuda:0", lr=1.0)