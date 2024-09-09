import logging
import numpy as np
import torch
import sys
from EduCDM import CDM
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import pickle
from datasets.utils import *
from EduCDM.IRT.irt import irt3pl
from sklearn.metrics import roc_auc_score, accuracy_score
#from metrics import *
import json

with open('../datasets/dataset_info.json', 'r', encoding='utf-8') as file:
    dataset_info = json.load(file)

class IRTNet(nn.Module):
    def __init__(self, user_num, item_num, value_range=None, a_range=None, irf_kwargs=None, sensitive_name=None, dataset_index=None, mode=None):
        super(IRTNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}
        self.theta = nn.Embedding(self.user_num, 1)
        self.fisced = nn.Embedding(6, 1)
        self.a = nn.Embedding(self.item_num, 1)
        self.b = nn.Embedding(self.item_num, 1)
        self.c = nn.Embedding(self.item_num, 1)
        self.value_range = value_range
        self.a_range = a_range
        self.alpha = nn.Embedding(self.user_num, 1)
        self.beta = nn.Embedding(self.user_num, 1)
        self.hidden_size = 512
        self.sensitive_name = sensitive_name
        self.dataset_index = dataset_index
        self.mode = mode

        
        self.mlp_combine = nn.Sequential(
            nn.Linear(2, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, 1),
            nn.BatchNorm1d(1),
        )
        
        self.mlp_sensitive = nn.Sequential(
            nn.Linear(1, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, 1),
            nn.BatchNorm1d(1),
        )
        
        self.mlp_sensitive_dense = nn.Sequential(
            nn.Linear(1, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, 1),
            nn.BatchNorm1d(1),
        )
        
        
        self.binary_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 1),
                nn.Sigmoid()
            ) for i in range(5)
        ])
        
        if self.sensitive_name == 'escs':
            self.mlp_sensitive_reverse = nn.Sequential(
                nn.Linear(1, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size//2),
                nn.BatchNorm1d(self.hidden_size//2),
                nn.ReLU(),
                nn.Linear(self.hidden_size//2, 1),
                nn.BatchNorm1d(1)
            )
        else:
            self.mlp_sensitive_reverse = nn.Sequential(
                nn.Linear(1, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size//2),
                nn.BatchNorm1d(self.hidden_size//2),
                nn.ReLU(),
                nn.Linear(self.hidden_size//2, 6),
                nn.Softmax(dim=1)
            )
        
        self.last_combine = nn.Sequential(
            nn.Linear(2, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, 1),
            nn.BatchNorm1d(1),
        )
        
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.size()) >= 2:
                nn.init.xavier_normal_(param)

    def forward(self, user, item, sensitive, labels, cls_labels, train_mode=True):   
        if self.sensitive_name == 'escs':
            sensitive_feature = torch.unsqueeze(sensitive, 1)
        else:
            sensitive_feature = self.fisced(sensitive)
        
        sensitive_feature = self.mlp_sensitive(sensitive_feature)
        theta = self.theta(user)
        Uf_features = self.mlp_combine(torch.cat([theta, sensitive_feature], dim=-1))
        Ud_features = self.mlp_sensitive_dense(sensitive_feature)
        
        Uf_reverse_sensitive = self.mlp_sensitive_reverse(Uf_features)
        Ud_reverse_sensitive = self.mlp_sensitive_reverse(Ud_features)
        
        if self.sensitive_name == 'escs':
            sensitive_mean = torch.tensor([dataset_info[self.dataset_index]['escs_mean']]).to('cuda:0')
            con_sensitive = sensitive_mean.expand(sensitive.size(0), -1)
            criterion = nn.MSELoss()
            loss_Uf_reverse = criterion(Uf_reverse_sensitive, con_sensitive)
            sensitive = torch.unsqueeze(sensitive, dim=-1)
        else:
            criterion = nn.CrossEntropyLoss()
            loss_Uf_reverse = -torch.mean(torch.sum(Uf_reverse_sensitive * torch.log(Uf_reverse_sensitive + 1e-8), dim=1))
            
        loss_Ud_reverse = criterion(Ud_reverse_sensitive, sensitive)
        reverse_loss = loss_Uf_reverse + loss_Ud_reverse
        sensitive = torch.squeeze(sensitive, dim=-1)

        Uf = torch.squeeze(Uf_features, dim=-1)
        Ud = torch.squeeze(Ud_features, dim=-1)

        a = torch.squeeze(self.a(item), dim=-1)
        b = torch.squeeze(self.b(item), dim=-1)
        c = torch.squeeze(self.c(item), dim=-1)
        c = torch.sigmoid(c)
            
        if self.value_range is not None:
            theta = self.value_range * (torch.sigmoid(theta) - 0.5)
            Uf = self.value_range * (torch.sigmoid(Uf) - 0.5)
            Ud = self.value_range * (torch.sigmoid(Ud) - 0.5)
            b = self.value_range * (torch.sigmoid(b) - 0.5)
        else:
            Uf = torch.sigmoid(Uf)
            Ud = torch.sigmoid(Ud)
            b = torch.sigmoid(b)
        
        alpha = torch.squeeze(torch.sigmoid(self.alpha(user)), -1)
        theta = torch.sigmoid((1-alpha) * Uf + alpha * Ud)
        
        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)
        
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b): 
            raise ValueError('ValueError:theta,a,b may contains nan!  The value_range or a_range is too large.')
        
        con_theta = torch.mean(self.theta.weight.data, dim=0)
        con_theta = con_theta.expand(sensitive.size(0), -1)

        if self.sensitive_name == 'escs':
            sensitive_mean = torch.tensor([dataset_info[self.dataset_index]['escs_mean']]).to('cuda:0')
            con_sensitive = sensitive_mean.expand(sensitive.size(0), -1)
        else:
            con_sensitive = torch.mean(self.fisced.weight.data, dim=0)
        
        con_sensitive = con_sensitive.expand(sensitive.size(0), -1)
        con_sensitive = self.mlp_sensitive(con_sensitive)
        
        con_Uf = torch.squeeze(self.mlp_combine(torch.cat([con_theta, con_sensitive], dim=-1)), dim=-1)
        
        if self.value_range is not None:
            con_theta = self.value_range * (torch.sigmoid(con_theta) - 0.5)
            con_Uf = self.value_range * (torch.sigmoid(con_Uf) - 0.5)
        else:
            con_Uf = torch.sigmoid(con_Uf)
        
        con_theta = torch.sigmoid((1-alpha) * con_Uf +  alpha * Ud)
        
        binary_outputs = [torch.squeeze(classifier(Uf_features)) for classifier in self.binary_classifiers]
        cls_losses = [nn.BCEWithLogitsLoss()(output, cls_label.float()) for output, cls_label in zip(binary_outputs, cls_labels)]
        cls_loss = sum(cls_losses) / len(cls_losses)
        
        beta = torch.squeeze(torch.sigmoid(self.beta(user)), -1)
        debias_theta = torch.sigmoid(theta - beta * con_theta)
        
        if not train_mode:
            if self.mode == 'sensitive':
                return self.irf(theta, a, b, c, **self.irf_kwargs), binary_outputs
            else:
                return self.irf(debias_theta, a, b, c, **self.irf_kwargs), binary_outputs
        
        loss_function = nn.BCELoss()
        total_loss = 0
        theta_loss = loss_function(self.irf(theta, a, b, c, **self.irf_kwargs), labels)
        debias_theta_loss = loss_function(self.irf(debias_theta, a, b, c, **self.irf_kwargs), labels)
        Uf_loss = loss_function(self.irf(Uf, a, b, c, **self.irf_kwargs), labels)
        Ud_loss = loss_function(self.irf(Ud, a, b, c, **self.irf_kwargs), labels)
        if self.sensitive_name == 'escs':
            theta_eo_loss = self.calc_eo1(sensitive, self.irf(debias_theta, a, b, c, **self.irf_kwargs))
            Ud_eo_loss = self.calc_eo1(sensitive, self.irf(Ud, a, b, c, **self.irf_kwargs))
        else:
            theta_eo_loss = self.calc_eo2(sensitive, self.irf(debias_theta, a, b, c, **self.irf_kwargs))
            Ud_eo_loss = self.calc_eo2(sensitive, self.irf(Ud, a, b, c, **self.irf_kwargs))
        ce_loss = debias_theta_loss
        total_loss += ce_loss
        total_loss += theta_eo_loss
        total_loss -= Ud_eo_loss
        total_loss += 0.1 * cls_loss
        total_loss += 0.5 * reverse_loss
        if self.mode == 'ours':
            total_loss = total_loss
        return total_loss, theta_eo_loss, Ud_eo_loss

    @classmethod
    def irf(cls, theta, a, b, c, **kwargs):
        return irt3pl(theta, a, b, c, F=torch, **kwargs)

    def calc_eo1(self, escs, pred):
        indices = torch.argsort(escs)
        #print('index:', indices)

        n = len(indices)
        q1_index = int(n / 4)
        q3_index = int(3 * n / 4)

        # Extract indices for three groups
        disadv_indices = indices[:q1_index]
        mid_indices = indices[q1_index:q3_index]
        adv_indices = indices[q3_index:]
        #print(disadv_indices, mid_indices, adv_indices)

        # Use indices to get corresponding pred values
        tpr_disadv = torch.mean(pred[disadv_indices])
        tpr_mid = torch.mean(pred[mid_indices])
        tpr_adv = torch.mean(pred[adv_indices])
            
        #tpr_disadv, tpr_mid, tpr_adv = torch.tensor(tpr_disadv), torch.tensor(tpr_mid), torch.tensor(tpr_adv)
        #tpr_disadv, tpr_mid, tpr_adv = tpr_disadv.requires_grad_(), tpr_mid.requires_grad_(), tpr_adv.requires_grad_()
        ##print(pred[disadv_indices], pred[mid_indices], pred[adv_indices])
        EO = torch.std(torch.stack([tpr_disadv, tpr_mid, tpr_adv]))
        return EO
    
    def calc_eo2(self, escs, pred):
        # Map original escs to new groups
        disadv_indices = torch.nonzero((escs == 0) | (escs == 1)).squeeze()
        mid_indices = torch.nonzero((escs == 2) | (escs == 3)).squeeze()
        adv_indices = torch.nonzero((escs == 4) | (escs == 5)).squeeze()

        # Use indices to get corresponding pred values
        tpr_disadv = torch.mean(pred[disadv_indices])
        tpr_mid = torch.mean(pred[mid_indices])
        tpr_adv = torch.mean(pred[adv_indices])
            
        # Calculate Equal Opportunity
        EO = torch.std(torch.stack([tpr_disadv, tpr_mid, tpr_adv]))
        return EO


class IRT(CDM):
    def __init__(self, user_num, item_num, value_range=None, a_range=None, save_path=None, sensitive_name=None, dataset_index=None, mode=None):
        super(IRT, self).__init__()
        self.sensitive_name = sensitive_name
        self.irt_net = IRTNet(user_num, item_num, value_range, a_range, sensitive_name=sensitive_name, dataset_index=dataset_index, mode=mode)
        self.save_path = save_path
        self.train_loss = []
        self.metrics = {'eval_auc':[],
                        'eval_acc':[],
                        'eval_EO':[],
                        'eval_Do':[],
                        'eval_Du':[],
                        'eval_IR':[]
                        }

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001) -> ...:
        self.irt_net = self.irt_net.to(device)
        loss_function = nn.BCELoss()

        trainer = torch.optim.Adam(self.irt_net.parameters(), lr)
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(epochs):
            warmup_epochs = 10
            decay_epochs = epoch - warmup_epochs
            initial_lr = 0.01
            final_lr = 0.001
            
            if epochs < warmup_epochs:
                return (epochs) / warmup_epochs * initial_lr
            else:
                return max(0.0, 1.0 - (epochs - warmup_epochs) / (decay_epochs - warmup_epochs)) * (initial_lr - final_lr) + final_lr
            
        scheduler = LambdaLR(trainer, lr_lambda=lr_lambda)
        
        best_epoch = 0
        best_auc = 0
        best_acc = 0
        for e in range(epoch):
            losses = []
            scheduler.step()
            print(f'Epoch {e}/{epoch}, Learning Rate: {trainer.param_groups[0]["lr"]}')
            for batch_data in tqdm(train_data, "Epoch %s" % e, ncols=100):
                user_id, item_id, response, fisced, escs, cls_labels = batch_data['user_id'], batch_data['item_id'], batch_data['response'], batch_data['FISCED'], batch_data['ESCS'], batch_data['cls_labels']
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                escs: torch.Tensor = escs.to(device)
                labels: torch.Tensor = response.to(device)
                cls_labels: list = [labels.to(device) for labels in cls_labels]
                loss, cls_loss, reverse_loss = self.irt_net(user_id, item_id, escs, labels, cls_labels)
            
                # back propagation
                trainer.zero_grad()
                loss.backward()
                        
                trainer.step()

                losses.append(loss.mean().item())
        
            print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))
            print(cls_loss, reverse_loss)
            self.train_loss.append(np.mean(losses))

            if test_data is not None:
                EO, Do, Du, IR, auc, accuracy, binary_acc = self.eval(test_data, device=device)
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
            

    def eval(self, test_data, device="cpu") -> tuple:
        self.irt_net = self.irt_net.to(device)
        self.irt_net.eval()
        y_pred = []
        y_true = []
        escs_list = []
        fisced_list = []
        binary_pred = []
        binary_acc = []
        for batch_data in tqdm(test_data, "evaluating", ncols=100):
            user_id, item_id, response, fisced, escs, cls_labels = batch_data['user_id'], batch_data['item_id'], batch_data['response'], batch_data['FISCED'], batch_data['ESCS'], batch_data['cls_labels']
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            escs: torch.Tensor = escs.to(device)
            fisced: torch.Tensor = fisced.to(device)
            labels: torch.Tensor = response.to(device)
            cls_labels: list = [labels.to(device) for labels in cls_labels]
            pred, binary_outputs = self.irt_net(user_id, item_id, escs, labels, cls_labels, train_mode=False)
            if binary_pred == []:
                binary_pred = [output.cpu().detach().tolist() for output in binary_outputs]
            else:
                for i in range(len(binary_outputs)):
                    binary_pred[i].extend(binary_outputs[i].cpu().detach().tolist())
                    
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())
            escs_list.extend(escs.cpu().tolist())
            fisced_list.extend(fisced.cpu().tolist())
            
        if self.sensitive_name == 'escs':
            EO, Do, Du, IR = self.calc_escs_fair(escs_list, np.array(y_pred) >= 0.5, y_true)
        elif self.sensitive_name == 'fisced':
            EO, Do, Du, IR = self.calc_fisced_fair(escs_list, np.array(y_pred) >= 0.5, y_true)
        
        for task_labels, task_outputs in zip(cls_labels, binary_pred):
            predicted_labels = [1 if output > 0.5 else 0 for output in task_outputs]
            correct_predictions = sum([1 for pred, label in zip(predicted_labels, task_labels) if pred == label])
            accuracy = correct_predictions / len(task_labels)
            binary_acc.append(accuracy)

        self.irt_net.train()
        return EO, Do, Du, IR, roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), binary_acc

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

    def calc_escs_fair(self, ecsc, pred, true):
        data = list(zip(ecsc, pred, true))
        
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
        
        fnr_mid = 1 - tpr_mid
        fpr_mid = self.calculate_fpr(mid_group)
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
    

        
