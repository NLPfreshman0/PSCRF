import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from collections import defaultdict


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)
    
class RCD(nn.Module):
    def __init__(self, args, knowledge_n, exer_n, student_n):
        
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(RCD, self).__init__()

        #self.sensitive_name = sensitive_name
        self.hidden_size = 512
        self.alpha = nn.Embedding(student_n, 1)
        self.beta = nn.Embedding(student_n, 1)
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
        
        self.binary_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.stu_dim, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 1),
                nn.Sigmoid()
            ) for i in range(5)
        ])
        
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

                
        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        latent_dim = knowledge_n
        d_i_train_1, d_j_train_1, sparse_u_i_1, sparse_i_u_1 = obtain_adjency_matrix2(args)

        for i in range(len(d_i_train_1)):
            d_i_train_1[i] = [d_i_train_1[i]]
        for i in range(len(d_j_train_1)):
            d_j_train_1[i] = [d_j_train_1[i]]

        self.d_i_train_1 = torch.cuda.FloatTensor(d_i_train_1)
        self.d_j_train_1 = torch.cuda.FloatTensor(d_j_train_1)

        self.d_i_train_1 = self.d_i_train_1.expand(-1, latent_dim)
        self.d_j_train_1 = self.d_j_train_1.expand(-1, latent_dim)
        self.user_item_matrix_1 = sparse_u_i_1
        self.item_user_matrix_1 = sparse_i_u_1

        # initialize
        # for name, param in self.named_parameters():
        #     if 'weight' in name:
        #         nn.init.xavier_normal_(param)
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.size()) >= 2:
                nn.init.xavier_normal_(param)
        
        def initialize_layer(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)

    def forward(self, stu_id, input_exercise, input_knowledge_point, sensitive, labels, cls_labels):
        # before prednet
        batch_size = input_knowledge_point.size(0)
        total_dim = 68
        multi_hot = torch.zeros(batch_size, total_dim).to(input_knowledge_point.device)
        for i in range(batch_size):
            indices = input_knowledge_point[i]
            multi_hot[i][indices] = 1.0
        input_knowledge_point = multi_hot
        
        stu_emb = self.student_emb.weight
        exer_emb = self.k_difficulty.weight

        gcn1_users_embedding_1 = (torch.sparse.mm(self.user_item_matrix_1, exer_emb) + stu_emb.mul(
            self.d_i_train_1))  # *2. #+ users_embedding
        gcn1_items_embedding_1 = (torch.sparse.mm(self.item_user_matrix_1, stu_emb) + exer_emb.mul(
            self.d_j_train_1))  # *2. #+ items_embedding
        gcn2_users_embedding_1 = (
                torch.sparse.mm(self.user_item_matrix_1, gcn1_items_embedding_1) + gcn1_users_embedding_1.mul(
            self.d_i_train_1))  # *2. #+ users_embedding
        gcn2_items_embedding_1 = (
                torch.sparse.mm(self.item_user_matrix_1, gcn2_users_embedding_1) + gcn1_items_embedding_1.mul(
            self.d_j_train_1))  # *2. #+ items_embedding

        gcn3_users_embedding_1 = (
                    torch.sparse.mm(self.user_item_matrix_1, gcn2_items_embedding_1) + gcn2_users_embedding_1.mul(
                self.d_i_train_1))  # *2. #+ users_embedding
        gcn3_items_embedding_1 = (
                    torch.sparse.mm(self.item_user_matrix_1, gcn2_users_embedding_1) + gcn2_items_embedding_1.mul(
                self.d_j_train_1))  # *2. #+ items_embedding
        gcn4_users_embedding_1 = (
                torch.sparse.mm(self.user_item_matrix_1, gcn3_items_embedding_1) + gcn3_users_embedding_1.mul(
            self.d_i_train_1))  # *2. #+ users_embedding
        gcn4_items_embedding_1 = (
                torch.sparse.mm(self.item_user_matrix_1, gcn3_users_embedding_1) + gcn3_items_embedding_1.mul(
            self.d_j_train_1))  # *2. #+ items_embedding

        stu_emb = stu_emb + gcn1_users_embedding_1 + gcn2_users_embedding_1 + gcn3_users_embedding_1 + gcn4_users_embedding_1
        exer_emb = exer_emb + gcn1_items_embedding_1 + gcn2_items_embedding_1 + gcn3_items_embedding_1 + gcn4_items_embedding_1

        #stat_emb = torch.sigmoid(stu_emb[stu_id])
        
        sensitive_feature = torch.unsqueeze(sensitive, 1)
        sensitive_feature = self.mlp_sensitive(sensitive_feature)
        Uf_features = self.mlp_combine(torch.cat([stu_emb[stu_id], sensitive_feature], dim=-1))
        Ud_features = self.mlp_sensitive_dense(sensitive_feature)
        
        Uf_reverse_sensitive = self.mlp_sensitive_reverse(Uf_features)
        Ud_reverse_sensitive = self.mlp_sensitive_reverse(Ud_features)
        sensitive_mean = torch.tensor([0.00014507418272432547]).to('cuda:0')
        con_sensitive = sensitive_mean.expand(sensitive.size(0), -1)
        criterion = nn.MSELoss()
        loss_Uf_reverse = criterion(Uf_reverse_sensitive, con_sensitive)
        sensitive = torch.unsqueeze(sensitive, dim=-1)
        loss_Ud_reverse = criterion(Ud_reverse_sensitive, sensitive)
        reverse_loss = loss_Uf_reverse + loss_Ud_reverse
        sensitive = torch.squeeze(sensitive, dim=-1)
        
        Uf = torch.squeeze(Uf_features, dim=-1)
        Ud = torch.squeeze(Ud_features, dim=-1)
        Uf = torch.sigmoid(Uf)
        Ud = torch.sigmoid(Ud)
        alpha = torch.squeeze(torch.sigmoid(self.alpha(stu_id)), -1).unsqueeze(1)
        stat_emb = torch.sigmoid((1-alpha) * Uf + alpha * Ud)
        
        con_stu_emb = torch.mean(stu_emb.data, dim=0)
        con_stu_emb = con_stu_emb.expand(sensitive.size(0), -1)
        sensitive_mean = torch.tensor([0.00014507418272432547]).to('cuda:0')
        con_sensitive = sensitive_mean.expand(sensitive.size(0), -1)
        con_sensitive = con_sensitive.expand(sensitive.size(0), -1)
        con_sensitive = self.mlp_sensitive(con_sensitive)
        con_Uf = torch.squeeze(self.mlp_combine(torch.cat([con_stu_emb, con_sensitive], dim=-1)), dim=-1)
        con_Uf = torch.sigmoid(con_Uf)
        con_stat_emb = torch.sigmoid((1-alpha) * con_Uf +  alpha * Ud)
        
        binary_outputs = [torch.squeeze(classifier(Uf_features)) for classifier in self.binary_classifiers]
        cls_losses = [nn.BCEWithLogitsLoss()(output, cls_label.float()) for output, cls_label in zip(binary_outputs, cls_labels)]
        cls_loss = sum(cls_losses) / len(cls_losses)
        beta = torch.squeeze(torch.sigmoid(self.beta(stu_id)), -1).unsqueeze(1)
        debias_theta = torch.sigmoid(stat_emb - beta * con_stat_emb)
        
        k_difficulty = torch.sigmoid(exer_emb[input_exercise])

        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))
        # input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        # input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        # input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        # output_1 = torch.sigmoid(self.prednet_full3(input_x))
        b = e_difficulty
        a = k_difficulty
        kw = input_knowledge_point
        loss_function = nn.BCELoss()
        total_loss = 0
        theta_loss = loss_function(self.irf(stat_emb, a, b, kw), labels)
        #print(theta[0], theta1[0])
        debias_theta_loss = loss_function(self.irf(debias_theta, a, b, kw), labels)
        Uf_loss = loss_function(self.irf(Uf, a, b, kw), labels)
        Ud_loss = loss_function(self.irf(Ud, a, b, kw), labels)
        theta_eo_loss = self.calc_eo1(sensitive, self.irf(debias_theta, a, b, kw))
        Ud_eo_loss = self.calc_eo1(sensitive, self.irf(Ud, a, b, kw))
        
        total_loss = 0.5 * theta_loss + debias_theta_loss + 0.5 * Uf_loss + 0.5 * Ud_loss
        total_loss += theta_eo_loss
        total_loss -= Ud_eo_loss
        total_loss += 0.1 * cls_loss
        total_loss += 0.5 * reverse_loss
        return total_loss, self.irf(debias_theta, a, b, kw), binary_outputs, stat_emb, kw
    
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
    
    def irf(self, stat_emb, a, b, kw):
        input_x = b * (stat_emb - a) * kw #[b, 1] * ([b, k] - [b, k]) * [b, k]
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x)) 
        return output_1.view(-1)

import pandas as pd
def obtain_adjency_matrix2(args):
    read_data = '/zjq/zhangdacao/pisa/datasets/Australia/train.csv'
    print(read_data)
    data = pd.read_csv(read_data)
    train_data_user_score1,train_data_user_score0 = defaultdict(set), defaultdict(set)
    train_data_item_score1,train_data_item_score0 = defaultdict(set), defaultdict(set)
    for i in range(len(data)):
        u_id = data['CNTSTUID'][i]
        i_id = data['exercise_id'][i]
        train_data_user_score1[u_id].add(int(i_id))
        train_data_item_score1[int(i_id)].add(u_id)
    u_d_1 = readD(args, train_data_user_score1, args.student_n)
    i_d_1 = readD(args, train_data_item_score1, args.exer_n)
    sparse_u_i_1 = readTrainSparseMatrix(args, train_data_user_score1,u_d_1, i_d_1,  True)
    sparse_i_u_1 = readTrainSparseMatrix(args, train_data_item_score1,u_d_1, i_d_1, False)
    return u_d_1,i_d_1,sparse_u_i_1,sparse_i_u_1
    
def readD(args, set_matrix,num_):
    user_d=[]
    for i in range(num_):
        len_set=1.0/(len(set_matrix[i])+1)
        user_d.append(len_set)
    return user_d


def readTrainSparseMatrix(args, set_matrix,u_d,i_d, is_user):
    user_items_matrix_i=[]
    user_items_matrix_v=[]
    exer_num = args.exer_n
    student_n = args.student_n
    if is_user:
        d_i=u_d
        d_j=i_d
        user_items_matrix_i.append([student_n-1, exer_num-1])
        user_items_matrix_v.append(0)
    else:
        d_i=i_d
        d_j=u_d
        user_items_matrix_i.append([exer_num - 1, student_n - 1])
        user_items_matrix_v.append(0)
    for i in set_matrix:
        len_set=len(set_matrix[i])
        for j in set_matrix[i]:
            user_items_matrix_i.append([i,j])
            d_i_j=np.sqrt(d_i[i]*d_j[j])
            user_items_matrix_v.append(d_i_j)
    user_items_matrix_i=torch.LongTensor(user_items_matrix_i).cuda()
    user_items_matrix_v=torch.FloatTensor(user_items_matrix_v).cuda()
    #print(user_items_matrix_i.shape, user_items_matrix_v.shape)
    return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)