# coding: utf-8
# 2021/4/23 @ tongshiwei

import logging
import numpy as np
import torch
from EduCDM import CDM
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import sys
sys.path.append('/zjq/zhangdacao/pisa/EduCDM/EduCDM/IRT')
from utils import *
from irt import irt3pl
from sklearn.metrics import roc_auc_score, accuracy_score
from metrics import *



class IRTNet(nn.Module):
    def __init__(self, user_num, item_num, value_range=-1, a_range=-1, irf_kwargs=None):
        super(IRTNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}
        self.theta = nn.Embedding(self.user_num, 1)
        self.a = nn.Embedding(self.item_num, 1)
        self.b = nn.Embedding(self.item_num, 1)
        self.c = nn.Embedding(self.item_num, 1)
        self.value_range = value_range
        self.a_range = a_range
        self.enc = nn.Sequential(
            nn.Linear(1,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
        self.dis = nn.Sequential(
            nn.Linear(1,1)
        )

        for name, param in self.named_parameters():#很重要的。。
            if 'weight' in name:
                nn.init.xavier_normal_(param)
                
    def forward(self, user, item):
        theta = torch.squeeze(self.theta(user), dim=-1)
        a = torch.squeeze(self.a(item), dim=-1)
        b = torch.squeeze(self.b(item), dim=-1)
        c = torch.squeeze(self.c(item), dim=-1)
        c = torch.sigmoid(c)
        if self.value_range is not None:
            theta = self.value_range * (torch.sigmoid(theta) - 0.5)
            b = self.value_range * (torch.sigmoid(b) - 0.5)
        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The value_range or a_range is too large.')
        d_output = self.dis(self.enc(self.theta(user)))
        return self.irf(theta, a, b, c, **self.irf_kwargs), d_output

    @classmethod
    def irf(cls, theta, a, b, c, **kwargs):
        return irt3pl(theta, a, b, c, F=torch, **kwargs)


class IRT(CDM):
    def __init__(self, user_num, item_num, value_range=-1, a_range=-1, save_path=None):
        super(IRT, self).__init__()
        self.irt_net = IRTNet(user_num, item_num, value_range, a_range)
        self.save_path = save_path
        self.train_loss = []
        self.metrics = {'eval_auc':[],
                        'eval_acc':[],
                        'eval_EO':[],
                        'eval_Do':[],
                        'eval_Du':[],
                        'eval_IR':[]
                        }

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001, method='None', sensitive_name=None) -> ...:
        self.irt_net = self.irt_net.to(device)
        loss_function = nn.BCELoss()
        mse_function = nn.MSELoss()

        trainer = torch.optim.Adam(self.irt_net.parameters(), lr)
            
        best_epoch = 0
        best_auc = 0
        best_acc = 0
        for e in range(epoch):
            if e == 0 and method == 'adv':
                for param in self.irt_net.dis.parameters():
                    param.requires_grad = False
            if e == 10 and method == 'adv':
                for param in self.irt_net.dis.parameters():
                    param.requires_grad = True
                for param in self.irt_net.enc.parameters():
                    param.requires_grad = False
            losses = []
            for batch_data in tqdm(train_data, "Epoch %s" % e, ncols=100):
                user_id, item_id, response, fisced, escs = batch_data['user_id'], batch_data['item_id'], batch_data['response'], batch_data['FISCED'], batch_data['ESCS']
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                escs: torch.Tensor = escs.to(device)
                fisced: torch.Tensor = fisced.to(device)
                predicted_response, d_output = self.irt_net(user_id, item_id)
                response: torch.Tensor = response.to(device)
                loss = loss_function(predicted_response, response)
                #pred = torch.where(predicted_response > 0.5, torch.tensor(1.0, requires_grad=True), torch.tensor(0.0, requires_grad=True))
                if method == 'reg':
                    eo_loss = self.calc_eo(escs, predicted_response)
                    loss += 0.1 * eo_loss
                elif method == 'adv':
                    dis_loss = mse_function(torch.squeeze(d_output, dim=-1), escs)
                    #print(d_output, escs, dis_loss)
                    if e < 10:
                        loss += torch.log(dis_loss + 1)
                    else:
                        loss -= 0.01*torch.log(dis_loss + 1)

                # back propagation
                trainer.zero_grad()
                loss.backward()
                trainer.step()

                losses.append(loss.mean().item())
            #print(dis_loss)
            print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))
            self.train_loss.append(np.mean(losses))

            if test_data is not None:
                EO, Do, Du, IR, auc, accuracy = self.eval(test_data, device=device, sensitive_name=sensitive_name)
                self.metrics['eval_auc'].append(auc)
                self.metrics['eval_acc'].append(accuracy)
                self.metrics['eval_EO'].append(EO)
                self.metrics['eval_Do'].append(Do)
                self.metrics['eval_Du'].append(Du)
                self.metrics['eval_IR'].append(IR)
                
                if auc > best_auc:
                    best_epoch = e + 1
                    best_auc = auc
                    best_acc = accuracy
                    self.save(f'{self.save_path}/model-epoch{best_epoch}-auc{round(best_auc,3)}-acc{round(best_acc,3)}')
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, EO: %.6f, Do: %.6f, Du: %.6f, IR: %.6f" % (e, auc, accuracy, EO, Do, Du, IR))
            self.save_log()
            
        print('best_epoch:', best_epoch, 'best_auc:', best_auc, 'best_acc:', best_acc)
        
    def calc_eo(self, escs, pred):
        indices = torch.argsort(escs)

        n = len(indices)
        q1_index = int(n / 4)
        q3_index = int(3 * n / 4)

        # Extract indices for three groups
        disadv_indices = indices[:q1_index]
        mid_indices = indices[q1_index:q3_index]
        adv_indices = indices[q3_index:]

        # Use indices to get corresponding pred values
        tpr_disadv = torch.mean(pred[disadv_indices])
        tpr_mid = torch.mean(pred[mid_indices])
        tpr_adv = torch.mean(pred[adv_indices])
            
        #tpr_disadv, tpr_mid, tpr_adv = torch.tensor(tpr_disadv), torch.tensor(tpr_mid), torch.tensor(tpr_adv)
        #tpr_disadv, tpr_mid, tpr_adv = tpr_disadv.requires_grad_(), tpr_mid.requires_grad_(), tpr_adv.requires_grad_()
        EO = torch.std(torch.stack([tpr_disadv, tpr_mid, tpr_adv]))
        return EO

    def eval(self, test_data, device="cpu", sensitive_name=None) -> tuple:
        self.irt_net = self.irt_net.to(device)
        self.irt_net.eval()
        y_pred = []
        y_true = []
        escs_list = []
        fisced_list = []
        for batch_data in tqdm(test_data, "evaluating", ncols=100):
            user_id, item_id, response, fisced, escs = batch_data['user_id'], batch_data['item_id'], batch_data['response'], batch_data['FISCED'], batch_data['ESCS']
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            pred, _ = self.irt_net(user_id, item_id)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())
            escs_list.extend(escs.cpu().tolist())
            fisced_list.extend(fisced.cpu().tolist())
        
        #print(sensitive_name)
        if sensitive_name == 'escs':
            EO, Do, Du, IR = self.calc_escs_fair(escs_list, np.array(y_pred) >= 0.5, y_true)
        elif sensitive_name == 'fisced':
            EO, Do, Du, IR = self.calc_fisced_fair(fisced_list, np.array(y_pred) >= 0.5, y_true)

        self.irt_net.train()
        return EO, Do, Du, IR, roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.irt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.irt_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
    
    def save_log(self):
        with open(f'{self.save_path}/train_loss.pkl', 'wb') as file:
            pickle.dump(self.train_loss, file)
        with open(f'{self.save_path}/eval_auc.pkl', 'wb') as file:
            pickle.dump(self.metrics, file)

    def calc_escs_fair(self, escs, pred, true):
        data = list(zip(escs, pred, true))
        
        sorted_data = sorted(data, key=lambda x: x[0])
        
        n = len(sorted_data)
        q1_index = int(n / 4)
        q3_index = int(3 * n / 4)

        disadv_group = sorted_data[:q1_index]
        mid_group = sorted_data[q1_index:q3_index]
        adv_group = sorted_data[q3_index:]

        tpr_disadv = self.calculate_tpr(disadv_group)
        tpr_mid = self.calculate_tpr(mid_group)
        tpr_adv = self.calculate_tpr(adv_group)

        fpr_adv = self.calculate_fpr(adv_group)
        fpr_disadv = self.calculate_fpr(disadv_group)
        
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
    
    def calc_fisced_fair(self, fisced, pred, true):
        data = list(zip(fisced, pred, true))

        disadv_group = [item for item in data if item[0] in [0, 1]]
        mid_group = [item for item in data if item[0] in [2, 3]]
        adv_group = [item for item in data if item[0] in [4, 5]]
        #print(len(disadv_group), len(mid_group), len(adv_group))

        tpr_disadv = self.calculate_tpr(disadv_group)
        tpr_mid = self.calculate_tpr(mid_group)
        tpr_adv = self.calculate_tpr(adv_group)

        fpr_adv = self.calculate_fpr(adv_group)
        fpr_disadv = self.calculate_fpr(disadv_group)

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

    
    def calculate_tpr_train(self, group):
        true_positives = torch.sum(torch.tensor([1 for _, pred, true in group if pred == 1 and true == 1]))
        actual_positives = torch.sum(torch.tensor([1 for _, _, true in group if true == 1]))

        if actual_positives == 0:
            return torch.tensor(0.0)

        tpr = true_positives / actual_positives
        return tpr

    def calculate_tpr(self, group):
        true_positives = sum(1 for _, pred, true in group if pred == 1 and true == 1)
        actual_positives = sum(1 for _, _, true in group if true == 1)
        
        if actual_positives == 0:
            return 0.0
        
        tpr = true_positives / actual_positives
        return tpr
    
    def calculate_fpr(self, group):
        false_positives = sum(1 for _, pred, true in group if pred == 1 and true == 0)
        actual_negatives = sum(1 for _, _, true in group if true == 0)

        if actual_negatives == 0:
            return 0.0

        fpr = false_positives / actual_negatives
        return fpr
    

        
