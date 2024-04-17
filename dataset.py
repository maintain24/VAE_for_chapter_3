# -*- coding:utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Dataset

class VAE_Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
        # 返回数据和对应的标签
        # data_point = torch.tensor(self.data[index], dtype=torch.float32)  # 转换为浮点数张量
        # return data_point, self.labels[index]

