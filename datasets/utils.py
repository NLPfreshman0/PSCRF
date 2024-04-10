'''
Author: wuyongyu wuyongyu@atomecho.xyz
Date: 2024-01-28 22:07:00
LastEditors: wuyongyu wuyongyu@atomecho.xyz
LastEditTime: 2024-02-03 03:43:28
FilePath: /zjq/zhangdacao/pisa/datasets/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import torch

class MyDataset(Dataset):
    def __init__(self, path='/zjq/zhangdacao/pisa/datasets/Australia', split='train', sensitive_name=None):
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
        # books = torch.tensor(self.data['ST013Q01TA'][idx], dtype=torch.float32)
        # table_computers = torch.tensor(self.data['ST012Q07NA'][idx], dtype=torch.float32)
        # internet = torch.tensor(self.data['ST011Q06TA'][idx], dtype=torch.float32)
        # computers = torch.tensor(self.data['ST011Q04TA'][idx], dtype=torch.float32)
        # ebooks = torch.tensor(self.data['ST012Q08NA'][idx], dtype=torch.float32)
        
        # Assuming you want to return these values as a dictionary
        sample = {
            'user_id': user_id,
            'item_id': exercise_id,
            'response': score,
            'FISCED': FISCED,
            'ESCS': ESCS,
            'knowledge_list': knowledge_list,
            'cls_labels': [books, table_computers, internet, computers, ebooks]
            # 'books': books,
            # 'table_computers': table_computers,
            # 'internet': internet,
            # 'computers': computers,
            # 'ebooks': ebooks
        }
        return sample
    
# my_dataset = MyDataset()
# print(my_dataset[0])

# my_dataloader = DataLoader(dataset=my_dataset, batch_size=4, shuffle=True)
# for batch in my_dataloader:
#     print(batch)
#     break

