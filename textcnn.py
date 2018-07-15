# Copyright (c) 2018 Aria-K-Alethia@github.com
# Licence: MIT
# textCNN for text classification
# Any use of this code should display all the info above; the user agrees to assume all liability for the use of this code.
import torch
import torch.nn as nn
import torch.nn.functional as F

class textCNN(nn.Module):
    '''
        overview:
            textCNN for text classification
        params:
            filter_size: list, size of window
            filter_number: number of filter for each filter size
            dropout: dropout ratio
            embedding_size: embedding size
    '''
    def __init__(self, filter_size, filter_number, embedding_size = 300, dropout = 0.5, label_number = 19):
        self.filter_size = filter_size
        self.filter_number = filter_number
        self.dropout = nn.Dropout(dropout)
        self.embedding_size = embedding_size
        # conv
        self.conv = [nn.Conv2d(1, filter_number, (size, embedding_size)) for size in filter_size]
        self.fc = nn.Linear(len(filter_size) * filter_number, label_number)
        self.lsm = nn.LogSoftmax(-1)
    def forward(self, x):
        '''
            overview:
                the forward method of textCNN
            params:
                x: [#batch, 1, #length, #dim]
            return:
                out: [#batch, #label_number]
        '''
        temp = [conv(x) for conv in self.conv] # [batch, number, length - size + 1, 1]
        temp = [F.relu(item) for item in temp]
        temp = [F.max_pool2d(item, (item.shape[2], item.shape[3]).squeeze() for item in temp]
        temp = torch.cat(temp, dim = 1)
        temp = self.dropout(temp)
        temp = self.fc(temp)
        temp = self.lsm(temp)
        return temp
        





