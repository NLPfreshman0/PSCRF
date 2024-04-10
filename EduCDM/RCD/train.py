import pandas as pd
import sys
sys.path.append('/zjq/zhangdacao/pisa/datasets')
from utils import *
from RCD import RCD
import os

import json
from torch.utils.data import DataLoader
import random
import numpy as np
import argparse
from longling.ML.metrics import POrderedDict
device = 'cuda:0'

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
    # parser.add_argument('--method', type=str, default=None, help='Training method')
    parser.add_argument('--sensitive_name', type=str, default='escs', help='Training method')
    parser.add_argument('--student_n', type=int, default=0, help='student_n', required=False)
    parser.add_argument('--exer_n', type=int, default=0, help='exer_n', required=False)

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
print(train_dataset[0])
user_num = dataset_info[args.dataset_index]['user_num']
item_num = dataset_info[args.dataset_index]['item_num']
print(user_num, item_num)
args.student_n = user_num
args.exer_n = item_num
batch_size = 512
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

import os
import shutil
save_path = args.save_path + country
if not os.path.exists(save_path):
    os.makedirs(save_path)
if os.path.exists(save_path) and os.path.isdir(save_path) and os.listdir(save_path):
    shutil.rmtree(save_path)
    os.makedirs(save_path)
model = RCD(args, 68, item_num, user_num).to(device)

import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

def calc_escs_fair(ecsc, pred, true):
    data = list(zip(ecsc, pred, true))
    
    sorted_data = sorted(data, key=lambda x: x[0])
    
    n = len(sorted_data)
    q1_index = int(n / 4)
    q3_index = int(3 * n / 4)

    disadv_group = sorted_data[:q1_index]
    mid_group = sorted_data[q1_index:q3_index]
    adv_group = sorted_data[q3_index:]

    tpr_disadv = calculate_tpr(disadv_group)
    tpr_mid = calculate_tpr(mid_group)
    tpr_adv = calculate_tpr(adv_group)

    fpr_adv = calculate_fpr(adv_group)
    fpr_disadv = calculate_fpr(disadv_group)
    
    fnr_disadv = 1 - tpr_disadv
    fnr_adv = 1 - tpr_adv
    
    EO = np.std([tpr_disadv, tpr_mid, tpr_adv])
    Do = fpr_adv - fpr_disadv
    Du = fnr_disadv - fnr_adv
    
    beta = 2
    
    precision = tpr_disadv / (tpr_disadv + fpr_disadv) if tpr_disadv + fpr_disadv != 0 else 0
    recall = tpr_disadv / (tpr_disadv + fnr_disadv) if tpr_disadv + fnr_disadv != 0 else 0
    IR = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (beta**2 * precision + recall) != 0 else 0
    return EO, Do, Du, IR

def calc_fisced_fair(fisced, pred, true):
    data = list(zip(fisced, pred, true))

    disadv_group = [item for item in data if item[0] in [0, 1]]
    mid_group = [item for item in data if item[0] in [2, 3]]
    adv_group = [item for item in data if item[0] in [4, 5]]
    #print(len(disadv_group), len(mid_group), len(adv_group))

    tpr_disadv = calculate_tpr(disadv_group)
    tpr_mid = calculate_tpr(mid_group)
    tpr_adv = calculate_tpr(adv_group)

    fpr_adv = calculate_fpr(adv_group)
    fpr_disadv = calculate_fpr(disadv_group)

    fnr_disadv = 1 - tpr_disadv
    fnr_adv = 1 - tpr_adv

    EO = np.std([tpr_disadv, tpr_mid, tpr_adv])
    Do = fpr_adv - fpr_disadv
    Du = fnr_disadv - fnr_adv

    beta = 2

    precision = tpr_disadv / (tpr_disadv + fpr_disadv) if tpr_disadv + fpr_disadv != 0 else 0
    recall = tpr_disadv / (tpr_disadv + fnr_disadv) if tpr_disadv + fnr_disadv != 0 else 0
    IR = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (beta**2 * precision + recall) != 0 else 0

    return EO, Do, Du, IR

def calculate_tpr(group):
    true_positives = sum(1 for _, pred, true in group if pred == 1 and true == 1)
    actual_positives = sum(1 for _, _, true in group if true == 1)
    
    if actual_positives == 0:
        return 0.0
    
    tpr = true_positives / actual_positives
    return tpr

def calculate_fpr(group):
    false_positives = sum(1 for _, pred, true in group if pred == 1 and true == 0)
    actual_negatives = sum(1 for _, _, true in group if true == 0)

    if actual_negatives == 0:
        return 0.0

    fpr = false_positives / actual_negatives
    return fpr


def doa_report(user, item, know, score, theta):
    df = pd.DataFrame({
        "user_id": user,
        "item_id": item,
        "score": score,
        "theta": theta,
        "knowledge": know
    })
    ground_truth = []

    for _, group_df in tqdm(df.groupby("item_id"), "formatting item df"):
        ground_truth.append(group_df["score"].values)
        ground_truth.append(1 - group_df["score"].values)

    knowledges = []
    knowledge_item = []
    knowledge_user = []
    knowledge_truth = []
    knowledge_theta = []
    for user, item, score, theta, knowledge in tqdm(
            df[["user_id", "item_id", "score", "theta", "knowledge"]].values,
            "formatting knowledge df"):
        if isinstance(theta, list):
            for i, (theta_i, knowledge_i) in enumerate(zip(theta, knowledge)):
                if knowledge_i == 1:
                    knowledges.append(i)
                    knowledge_item.append(item)
                    knowledge_user.append(user)
                    knowledge_truth.append(score)
                    knowledge_theta.append(theta_i)
        else:  # pragma: no cover
            for i, knowledge_i in enumerate(knowledge):
                if knowledge_i == 1:
                    knowledges.append(i)
                    knowledge_item.append(item)
                    knowledge_user.append(user)
                    knowledge_truth.append(score)
                    knowledge_theta.append(theta)

    knowledge_df = pd.DataFrame({
        "knowledge": knowledges,
        "user_id": knowledge_user,
        "item_id": knowledge_item,
        "score": knowledge_truth,
        "theta": knowledge_theta
    })
    knowledge_ground_truth = []
    knowledge_prediction = []
    for _, group_df in knowledge_df.groupby("knowledge"):
        _knowledge_ground_truth = []
        _knowledge_prediction = []
        for _, item_group_df in group_df.groupby("item_id"):
            _knowledge_ground_truth.append(item_group_df["score"].values)
            _knowledge_prediction.append(item_group_df["theta"].values)
        knowledge_ground_truth.append(_knowledge_ground_truth)
        knowledge_prediction.append(_knowledge_prediction)

    return POrderedDict(doa_eval(knowledge_ground_truth, knowledge_prediction))


def doa_eval(y_true, y_pred):
    """
    >>> import numpy as np
    >>> y_true = [
    ...     [np.array([1, 0, 1])],
    ...     [np.array([0, 1, 1])]
    ... ]
    >>> y_pred = [
    ...     [np.array([.5, .4, .6])],
    ...     [np.array([.2, .3, .5])]
    ... ]
    >>> doa_eval(y_true, y_pred)['doa']
    1.0
    >>> y_pred = [
    ...     [np.array([.4, .5, .6])],
    ...     [np.array([.3, .2, .5])]
    ... ]
    >>> doa_eval(y_true, y_pred)['doa']
    0.5
    """
    doa = []
    doa_support = 0
    z_support = 0
    for knowledge_label, knowledge_pred in tqdm(zip(y_true, y_pred),
                                                "doa metrics"):
        _doa = 0
        _z = 0
        for label, pred in zip(knowledge_label, knowledge_pred):
            if sum(label) == len(label) or sum(label) == 0:
                continue
            pos_idx = []
            neg_idx = []
            for i, _label in enumerate(label):
                if _label == 1:
                    pos_idx.append(i)
                else:
                    neg_idx.append(i)
            pos_pred = pred[pos_idx]
            neg_pred = pred[neg_idx]
            invalid = 0
            for _pos_pred in pos_pred:
                _doa += len(neg_pred[neg_pred < _pos_pred])
                invalid += len(neg_pred[neg_pred == _pos_pred])
            _z += (len(pos_pred) * len(neg_pred)) - invalid
        if _z > 0:
            doa.append(_doa / _z)
            z_support += _z
            doa_support += 1
    return {
        "doa": np.mean(doa),
        "doa_know_support": doa_support,
        "doa_z_support": z_support,
    }
        
def do_eval(loader, device=device):
    model.eval()
    y_pred = []
    y_true = []
    escs_list = []
    fisced_list = []
    binary_pred = []
    binary_acc = []
    user_list = []
    item_list = []
    know_list = []
    theta_list = []
    for batch_data in tqdm(loader, "evaluating", ncols=100):
        user_id, item_id, response, fisced, escs, cls_labels, knowledge_list = batch_data['user_id'], batch_data['item_id'], batch_data['response'], batch_data['FISCED'], batch_data['ESCS'], batch_data['cls_labels'], batch_data['knowledge_list']
        user_id: torch.Tensor = user_id.to(device)
        item_id: torch.Tensor = item_id.to(device)
        escs: torch.Tensor = escs.to(device)
        labels: torch.Tensor = response.to(device)
        cls_labels: list = [labels.to(device) for labels in cls_labels]
        knowledge_list: torch.Tensor = knowledge_list.to(device)
        fisced: torch.Tensor = fisced.to(device)
        loss, pred, binary_outputs, theta, kw = model(user_id, item_id, knowledge_list, escs, labels, cls_labels)
        if binary_pred == []:
            binary_pred = [output.cpu().detach().tolist() for output in binary_outputs]
        else:
            for i in range(len(binary_outputs)):
                binary_pred[i].extend(binary_outputs[i].cpu().detach().tolist())
        
        user_list.extend(user_id.cpu().tolist())
        item_list.extend(item_id.cpu().tolist())
        theta_list.extend(theta.cpu().tolist())
        know_list.extend(kw.cpu().tolist())
        y_pred.extend(pred.tolist())
        y_true.extend(response.tolist())
        escs_list.extend(escs.cpu().tolist())
        fisced_list.extend(fisced.cpu().tolist())
        
    if args.sensitive_name == 'escs':
        EO, Do, Du, IR = calc_escs_fair(escs_list, np.array(y_pred) >= 0.5, y_true)
    elif args.sensitive_name == 'fisced':
        EO, Do, Du, IR = calc_fisced_fair(fisced_list, np.array(y_pred) >= 0.5, y_true)
    
    doa = doa_report(user_list, item_list, know_list, y_true, theta_list)['doa']
    
    for task_labels, task_outputs in zip(cls_labels, binary_pred):
        #print(task_outputs)
        #predicted_labels = [1 if output > 0.5 else 0 for output in task_outputs]
        predicted_labels = task_outputs
        correct_predictions = sum([1 for pred, label in zip(predicted_labels, task_labels) if pred == label])
        accuracy = correct_predictions / len(task_labels)
        binary_acc.append(accuracy)

    model.train()
    return EO, Do, Du, IR, roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), doa, binary_acc
    
    # logging.info('eval ... ')
    # model.eval()
    # y_true, y_pred = [], []
    # for batch_data in tqdm(loader, "Evaluating"):
    #     user_id, item_id, response, fisced, escs, cls_labels, knowledge_list = batch_data['user_id'], batch_data['item_id'], batch_data['response'], batch_data['FISCED'], batch_data['ESCS'], batch_data['cls_labels'], batch_data['knowledge_list']
    #     user_id: torch.Tensor = user_id.to(device)
    #     item_id: torch.Tensor = item_id.to(device)
    #     knowledge_list: torch.Tensor = knowledge_list.to(device)
    #     pred, _, _, _: torch.Tensor = model(user_id, item_id, knowledge_list)
    #     y_pred.extend(pred.detach().cpu().tolist())
    #     y_true.extend(response.tolist())
        
    # model.train()
    # return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

loss_function = nn.BCELoss()
epoch = 100
from torch.optim.lr_scheduler import LambdaLR  
def lr_lambda(epochs):
    warmup_epochs = 10
    decay_epochs = epoch - warmup_epochs
    initial_lr = 0.007
    final_lr = 0.0001
    
    if epochs < warmup_epochs:
        # 在 warm-up 阶段使用线性增长
        return (epochs) / warmup_epochs * initial_lr
    else:
        # 在 warm-up 结束后，进行线性 decay
        return max(0.0, 1.0 - (epochs - warmup_epochs) / (decay_epochs - warmup_epochs)) * (initial_lr - final_lr) + final_lr
    
trainer = torch.optim.Adam(model.parameters(), lr=1.0)
scheduler = LambdaLR(trainer, lr_lambda=lr_lambda)

best_auc = 0
best_epoch = 0

for e in range(epoch):
    losses = []
    scheduler.step()
    for batch_data in tqdm(train_loader, "Epoch %s" % e):
        user_id, item_id, response, fisced, escs, cls_labels, knowledge_list = batch_data['user_id'], batch_data['item_id'], batch_data['response'], batch_data['FISCED'], batch_data['ESCS'], batch_data['cls_labels'], batch_data['knowledge_list']
        user_id: torch.Tensor = user_id.to(device)
        item_id: torch.Tensor = item_id.to(device)
        knowledge_list: torch.Tensor = knowledge_list.to(device)
        escs: torch.Tensor = escs.to(device)
        labels: torch.Tensor = response.to(device)
        cls_labels: list = [labels.to(device) for labels in cls_labels]
        loss, _, _, _, _ = model(user_id, item_id, knowledge_list, escs, labels, cls_labels)
        #response: torch.Tensor = response.to(device)
        #loss = loss_function(predicted_response, response)

        # back propagation
        trainer.zero_grad()
        loss.backward()
        trainer.step()

        losses.append(loss.mean().item())
    print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))

    EO, Do, Du, IR, auc, accuracy, doa, _ = do_eval(val_loader, device=device)
    print("[Epoch %d val] val_EO: %.6f, val_Du: %.6f, val_IR: %.6f, val_auc: %.6f, val_accuracy: %.6f, val_doa: %.6f" % (e, EO, Du, IR, auc, accuracy, doa))
    if auc > best_auc:
        best_auc = auc
        best_epoch = e
    EO, Do, Du, IR, auc, accuracy, doa, _ = do_eval(test_loader, device=device)
    print("[Epoch %d test] test_EO: %.6f, test_Du: %.6f, test_IR: %.6f, test_auc: %.6f, test_accuracy: %.6f, test_doa: %.6f" % (e, EO, Du, IR, auc, accuracy, doa))

print('best_epoch:', best_epoch, 'best_auc:', best_auc)