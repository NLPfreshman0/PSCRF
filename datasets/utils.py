from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import torch

class MyDataset(Dataset):
    def __init__(self, path="default", split='train', sensitive_name=None):
        self.data = pd.read_csv(f'{path}/{split}.csv')
        self.sensitive_name = sensitive_name
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id = torch.tensor(self.data['CNTSTUID'][idx])
        exercise_id = torch.tensor(self.data['exercise_id'][idx])
        score = torch.tensor(self.data['score'][idx], dtype=torch.float32)
        FISCED = torch.tensor(self.data['FISCED'][idx], dtype=torch.long)
        if self.sensitive_name == 'escs':
            ESCS = torch.tensor(self.data['ESCS'][idx], dtype=torch.float32)
        elif self.sensitive_name == 'fisced':
            ESCS = torch.tensor(self.data['FISCED'][idx], dtype=torch.long)
        else:
            print('sensitive name error')
        knowledge_strings = self.data['knowledge_list'][idx].strip('[]').split()
        knowledge_list = torch.tensor([int(knowledge) for knowledge in knowledge_strings], dtype=torch.long)
        books = torch.tensor(self.data['ST013Q01TA'][idx], dtype=torch.long)
        table_computers = torch.tensor(self.data['ST012Q07NA'][idx], dtype=torch.long)
        internet = torch.tensor(self.data['ST011Q06TA'][idx], dtype=torch.long)
        computers = torch.tensor(self.data['ST011Q04TA'][idx], dtype=torch.long)
        ebooks = torch.tensor(self.data['ST012Q08NA'][idx], dtype=torch.long)
        sample = {
            'user_id': user_id,
            'item_id': exercise_id,
            'response': score,
            'FISCED': FISCED,
            'ESCS': ESCS,
            'knowledge_list': knowledge_list,
            'cls_labels': [books, table_computers, internet, computers, ebooks]
        }
        return sample

