import pandas as pd
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir.split('PSCRF')[0]
sys.path.append(project_root+'PSCRF/datasets')

from utils import *
from IRT_CDM import IRT

import json
from torch.utils.data import DataLoader
import random
import numpy as np
import argparse
import torch


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

with open('../datasets/dataset_info.json', 'r', encoding='utf-8') as file:
    dataset_info = json.load(file)
print(dataset_info)
root = '../datasets'
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
model = IRT(user_num=user_num, item_num=item_num, save_path=save_path, sensitive_name=args.sensitive_name, dataset_index=args.dataset_index, mode=args.mode)

model.train(train_data=train_loader, test_data=val_loader, epoch=100, device="cuda:0", lr=1.0)