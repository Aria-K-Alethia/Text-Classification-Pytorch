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
            label_number: number of classes in text
    '''
    def __init__(self, filter_size, filter_number, embedding, embedding_size = 300, dropout = 0.5, label_number = 19):
        super(textCNN, self).__init__()
        self.embedding = embedding
        self.filter_size = filter_size
        self.filter_number = filter_number
        self.dropout = nn.Dropout(dropout)
        self.embedding_size = embedding_size
        # conv
        self.conv = nn.ModuleList([nn.Conv2d(1, filter_number, (size, embedding_size)) for size in filter_size])
        self.fc = nn.Linear(len(filter_size) * filter_number, label_number)
        self.lsm = nn.LogSoftmax(-1)
    def forward(self, x):
        '''
            overview:
                the forward method of textCNN
            params:
                x: [#batch, 1, #length]
            return:
                out: [#batch, #label_number]
        '''
        # check the dimension
        if x.dim() == 2:
            x.unsqueeze_(1)
        xe   = self.embedding(x)
        temp = [conv(xe) for conv in self.conv] # [batch, number, length - size + 1, 1]
        temp = [F.relu(item) for item in temp]
        temp = [F.max_pool2d(item, (item.shape[2], item.shape[3])).squeeze() for item in temp]
        temp = torch.cat(temp, dim = 1)
        temp = self.dropout(temp)
        temp = self.fc(temp)
        temp = self.lsm(temp)
        return temp
    
    def clip_weight(self, max_norm = 3):
        '''
            overview:
                clip the weight of fc layer if its l2 norm is greater than max_norm
            params:
                max_norm: maximum l2 norm for the weight, default is 3
        '''        
        norm = float(self.fc.weight.data.norm(2))
        if norm > max_norm:
            coef = max_norm / (norm + 1e-6)
            self.fc.weight.data.mul_(coef)




