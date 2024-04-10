# coding: utf-8
# 2021/4/1 @ WangFei

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import sys
sys.path.append('/zjq/zhangdacao/pisa/EduCDM')
from EduCDM import CDM
import pandas as pd
from longling.ML.metrics import POrderedDict


#sys.path.append('/zjq/zhangdacao/pisa/EduCDM/EduCDM/MIRT')
import json
with open('/zjq/zhangdacao/pisa/datasets/dataset_info.json', 'r', encoding='utf-8') as file:
    dataset_info = json.load(file)
import pickle



class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class Net(nn.Module):

    def __init__(self, knowledge_n, exer_n, student_n, sensitive_name=None, dataset_index=None, mode=None):
        self.dataset_index = dataset_index
        self.mode = mode
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)
        
        self.sensitive_name = sensitive_name
        self.fisced = nn.Embedding(6, 1)
        self.hidden_size = 512
        self.alpha = nn.Embedding(self.emb_num, 1)
        self.beta = nn.Embedding(self.emb_num, 1)
        
        self.mlp_combine = nn.Sequential(
            nn.Linear(2*self.stu_dim, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, self.stu_dim),
            nn.BatchNorm1d(self.stu_dim),
        )
        
        self.mlp_sensitive = nn.Sequential(
            nn.Linear(1, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, self.stu_dim),
            nn.BatchNorm1d(self.stu_dim),
        )
        
        self.mlp_sensitive_dense = nn.Sequential(
            nn.Linear(self.stu_dim, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, self.stu_dim),
            nn.BatchNorm1d(self.stu_dim),
        )
        
        # self.binary_classifiers = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(1, self.hidden_size),
        #         nn.BatchNorm1d(self.hidden_size),
        #         nn.ReLU(),
        #         nn.Linear(self.hidden_size, dataset_info[0]['s_num'][i]),
        #         nn.Softmax(dim=1)
        #     ) for i in range(5)
        # ])
        
        self.binary_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.stu_dim, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 1),
                nn.Sigmoid()
            ) for i in range(5)
        ])
        
        if self.sensitive_name == 'escs':
            self.mlp_sensitive_reverse = nn.Sequential(
                nn.Linear(self.stu_dim, self.hidden_size),
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
                nn.Linear(self.stu_dim, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size//2),
                nn.BatchNorm1d(self.hidden_size//2),
                nn.ReLU(),
                nn.Linear(self.hidden_size//2, 6),
                nn.Softmax(dim=1)
            )
        
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.size()) >= 2:
                nn.init.xavier_normal_(param)
        
        def initialize_layer(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)

        self.mlp_combine.apply(initialize_layer)
        self.mlp_sensitive.apply(initialize_layer)
        self.mlp_sensitive_dense.apply(initialize_layer)
        for model in self.binary_classifiers:
            model.apply(initialize_layer)
        self.mlp_sensitive_reverse.apply(initialize_layer)

        # # initialize
        # for name, param in self.named_parameters():
        #     if 'weight' in name:
        #         nn.init.xavier_normal_(param)

    def forward(self, user, item, sensitive, labels, cls_labels, input_knowledge_point, train_mode=True):
        batch_size = input_knowledge_point.size(0)
        total_dim = 68
        multi_hot = torch.zeros(batch_size, total_dim).to(input_knowledge_point.device)
        for i in range(batch_size):
            indices = input_knowledge_point[i]
            multi_hot[i][indices] = 1.0
        kw = multi_hot
        if self.sensitive_name == 'escs':
            sensitive_feature = torch.unsqueeze(sensitive, 1)
        else:
            sensitive_feature = self.fisced(sensitive)
            
        sensitive_feature = self.mlp_sensitive(sensitive_feature)
        theta1 = self.student_emb(user)
        
        Uf_features = self.mlp_combine(torch.cat([theta1, sensitive_feature], dim=-1))
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
        
        a = torch.sigmoid(self.k_difficulty(item))
        b = torch.sigmoid(self.e_difficulty(item)) 
        
        Uf = torch.sigmoid(Uf)
        Ud = torch.sigmoid(Ud)
        #b = torch.sigmoid(b)
        
        alpha = torch.squeeze(torch.sigmoid(self.alpha(user)), -1).unsqueeze(1)
        theta = torch.sigmoid((1-alpha) * Uf + alpha * Ud)
        
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The a_range is too large.')
        
        con_theta = torch.mean(self.student_emb.weight.data, dim=0)
        con_theta = con_theta.expand(sensitive.size(0), -1)
        #sensitive_mean = self.con_sens
        if self.sensitive_name == 'escs':
            sensitive_mean = torch.tensor([dataset_info[self.dataset_index]['escs_mean']]).to('cuda:0')
            con_sensitive = sensitive_mean.expand(sensitive.size(0), -1)
        else:
            con_sensitive = torch.mean(self.fisced.weight.data, dim=0)
        
        con_sensitive = con_sensitive.expand(sensitive.size(0), -1)
        con_sensitive = self.mlp_sensitive(con_sensitive)
        
        con_Uf = torch.squeeze(self.mlp_combine(torch.cat([con_theta, con_sensitive], dim=-1)), dim=-1)
        con_Uf = torch.sigmoid(con_Uf)
        con_theta = torch.sigmoid((1-alpha) * con_Uf +  alpha * Ud)
        
        binary_outputs = [torch.squeeze(classifier(Uf_features)) for classifier in self.binary_classifiers]
        cls_losses = [nn.BCEWithLogitsLoss()(output, cls_label.float()) for output, cls_label in zip(binary_outputs, cls_labels)]
        cls_loss = sum(cls_losses) / len(cls_losses)
        beta = torch.squeeze(torch.sigmoid(self.beta(user)), -1).unsqueeze(1)
        debias_theta = torch.sigmoid(theta - beta * con_theta)
        
        
        #return 1 / (1 + F.exp(- F.sum(F.multiply(a, theta), axis=-1) + b))
        #print(a, theta1)
        # print(- torch.sum(torch.multiply(a, theta), axis=-1))
        # print(self.irf(theta, a, b, **self.irf_kwargs))
        # print(((theta) * a).sum(dim=1, keepdim=True).shape)
        # print(b.shape)
        # print((((theta) * a).sum(dim=1, keepdim=True)+b).shape)
        if not train_mode:
            #print(self.irf(theta, a, b, **self.irf_kwargs))
            if self.mode == 'ours':
                return self.irf(debias_theta, a, b, kw), binary_outputs, debias_theta, kw#[torch.tensor([1] * 512) for i in range(5)]# #[torch.argmax(binary, dim=-1) for binary in binary_outputs]
            else:
                return self.irf(theta, a, b, kw), binary_outputs, theta, kw
        
        loss_function = nn.BCELoss()
        total_loss = 0
        theta_loss = loss_function(self.irf(theta, a, b, kw), labels)
        #print(theta[0], theta1[0])
        debias_theta_loss = loss_function(self.irf(debias_theta, a, b, kw), labels)
        Uf_loss = loss_function(self.irf(Uf, a, b, kw), labels)
        Ud_loss = loss_function(self.irf(Ud, a, b, kw), labels)
        if self.sensitive_name == 'escs':
            theta_eo_loss = self.calc_eo1(sensitive, self.irf(debias_theta, a, b, kw))
            Ud_eo_loss = self.calc_eo1(sensitive, self.irf(Ud, a, b, kw))
        else:
            theta_eo_loss = self.calc_eo2(sensitive, self.irf(debias_theta, a, b, kw))
            Ud_eo_loss = self.calc_eo2(sensitive, self.irf(Ud, a, b, kw))
        total_loss = 0.5 * theta_loss + debias_theta_loss + 0.5 * Uf_loss + 0.5 * Ud_loss
        total_loss += theta_eo_loss
        total_loss -= Ud_eo_loss
        total_loss += 0.1 * cls_loss
        total_loss += 0.5 * reverse_loss
        #total_loss = theta_loss
        
        if self.mode == 'ours':
            return total_loss, theta_eo_loss, Ud_eo_loss
        else:
            return theta_loss, theta_eo_loss, Ud_eo_loss
    
    
    def irf(self, theta, a, b, kw):
        input_x = b * (theta - a) * kw #[b, 1] * ([b, k] - [b, k]) * [b, k]
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x)) 
        return output_1.view(-1)
    
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


class NCDM(CDM):
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, knowledge_n, exer_n, student_n, save_path=None, sensitive_name=None, dataset_index=None, mode=None):
        super(NCDM, self).__init__()
        self.irt_net = Net(knowledge_n, exer_n, student_n, sensitive_name=sensitive_name, dataset_index=dataset_index, mode=mode)
        self.sensitive_name = sensitive_name
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
        # for name, param in self.irt_net.named_parameters():
        #     print(name)
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
            
        scheduler = LambdaLR(trainer, lr_lambda=lr_lambda)
        
        best_epoch = 0
        best_auc = 0
        best_acc = 0
        for e in range(epoch):
            losses = []
            scheduler.step()
            print(f'Epoch {e}/{epoch}, Learning Rate: {trainer.param_groups[0]["lr"]}')
            for batch_data in tqdm(train_data, "Epoch %s" % e, ncols=100):
                user_id, item_id, response, fisced, escs, cls_labels, knowledge_list = batch_data['user_id'], batch_data['item_id'], batch_data['response'], batch_data['FISCED'], batch_data['ESCS'], batch_data['cls_labels'], batch_data['knowledge_list']
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                escs: torch.Tensor = escs.to(device)
                labels: torch.Tensor = response.to(device)
                cls_labels: list = [labels.to(device) for labels in cls_labels]
                knowledge_list: torch.Tensor = knowledge_list.to(device)
                loss, cls_loss, reverse_loss = self.irt_net(user_id, item_id, escs, labels, cls_labels, knowledge_list)
            
                # back propagation
                trainer.zero_grad()
                loss.backward()
                        
                trainer.step()

                losses.append(loss.mean().item())
        
            print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))
            print(cls_loss, reverse_loss)
            self.train_loss.append(np.mean(losses))

            if test_data is not None:
                EO, Do, Du, IR, auc, accuracy, doa, binary_acc = self.eval(test_data, device=device)
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
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, EO: %.6f, Do: %.6f, Du: %.6f, IR: %.6f, doa: %.6f" % (e, auc, accuracy, EO, Do, Du, IR, doa))
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
        user_list = []
        item_list = []
        know_list = []
        theta_list = []
        for batch_data in tqdm(test_data, "evaluating", ncols=100):
            user_id, item_id, response, fisced, escs, cls_labels, knowledge_list = batch_data['user_id'], batch_data['item_id'], batch_data['response'], batch_data['FISCED'], batch_data['ESCS'], batch_data['cls_labels'], batch_data['knowledge_list']
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            escs: torch.Tensor = escs.to(device)
            labels: torch.Tensor = response.to(device)
            cls_labels: list = [labels.to(device) for labels in cls_labels]
            knowledge_list: torch.Tensor = knowledge_list.to(device)
            fisced: torch.Tensor = fisced.to(device)
            pred, binary_outputs, theta, kw = self.irt_net(user_id, item_id, escs, labels, cls_labels, knowledge_list, train_mode=False)
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
            
        if self.sensitive_name == 'escs':
            EO, Do, Du, IR = self.calc_escs_fair(escs_list, np.array(y_pred) >= 0.5, y_true)
        elif self.sensitive_name == 'fisced':
            EO, Do, Du, IR = self.calc_fisced_fair(fisced_list, np.array(y_pred) >= 0.5, y_true)
        
        doa = self.doa_report(user_list, item_list, know_list, y_true, theta_list)['doa']
        
        for task_labels, task_outputs in zip(cls_labels, binary_pred):
            #print(task_outputs)
            #predicted_labels = [1 if output > 0.5 else 0 for output in task_outputs]
            predicted_labels = task_outputs
            correct_predictions = sum([1 for pred, label in zip(predicted_labels, task_labels) if pred == label])
            accuracy = correct_predictions / len(task_labels)
            binary_acc.append(accuracy)

        self.irt_net.train()
        return EO, Do, Du, IR, roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), doa, binary_acc

    def save(self, filepath):
        torch.save(self.irt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.irt_net.load_state_dict(torch.load(filepath))  # , map_location=lambda s, loc: s
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
        
        pre_1_disadv = sum(1 for _, pred, _ in disadv_group if pred == 1) / len(disadv_group)
        pre_1_mid = sum(1 for _, pred, _ in mid_group if pred == 1) / len(mid_group)
        pre_1_adv = sum(1 for _, pred, _ in adv_group if pred == 1) / len(adv_group)
        pre_1_all = sum(1 for _, pred, _ in sorted_data if pred == 1) / len(sorted_data)
        print('pre_1:', pre_1_disadv, pre_1_mid, pre_1_adv, pre_1_all)
        
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


    def doa_report(self, user, item, know, score, theta):
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

        return POrderedDict(self.doa_eval(knowledge_ground_truth, knowledge_prediction))


    def doa_eval(self, y_true, y_pred):
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
