# Copyright (c) 2018 Aria-K-Alethia@github.com
# Licence: MIT
# Recurrent CNN for text classification
# Any use of this code should display all the info above; the user agrees to assume all liability for the use of this code

import torch
import torch.nn as nn
import torch.nn.functional as F

def build_rnn(rnn_type, **kwargs):
    return getattr(nn, rnn_type)(**kwargs)

class RCNN(nn.Module):
    '''
        overview:
            Recurrent CNN model
        params:
            rnn_type: 'LSTM' or 'GRU'
            rnn_size: hidden layer size
            rnn_layers: the layer of rnn
            embedding: the embedding matrix
            embedding_size: embedding size, default is 300
            dropout: dropout ratio, default is 0.5
            label_number: label number, default is 19
    '''
    def __init__(self, rnn_type, rnn_size, rnn_layers,  embedding, feature_number,
                embedding_size = 300, dropout = 0.5, label_number = 19, batch_first = True):
        super(RCNN, self).__init__()
        assert rnn_type in ['LSTM', 'GRU']
        self.name = 'RCNN'
        self.embedding = embedding
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        self.embedding_size = embedding_size
        self.batch_first = batch_first
        # the rnn in RCNN should be bidirectional
        self.rnn = build_rnn(rnn_type,
                        input_size = embedding_size, hidden_size = rnn_size,
                        num_layers = rnn_layers, batch_first = batch_first,
                        dropout = dropout, bidirectional = True)
        self.dropout = nn.Dropout(dropout)
        self.feature_number = feature_number
        self.fc1 = nn.Linear(embedding_size + rnn_size * rnn_layers, feature_number)         
        self.fc2 = nn.Linear(feature_number, label_number)
        self.lsm = nn.LogSoftmax(-1)
    def forward(self, x):
        '''
            overview:
                the forward method of RCNN
            params:
                x: [#batch, #length]
            return:
                lsm: [#batch, #label_number]
        '''
        xe = self.embedding(x)
        # rnn
        outputs, final = self.rnn(xe)
        # cat
        temp = torch.cat([outputs, xe], dim = -1)
        # fc1
        temp = self.fc1(temp)
        temp = self.dropout(temp)
        # non-linear
        temp = F.relu(temp)
        # max-pooling
        temp = F.max_pool2d(temp, (temp.shape[1], 1)).squeeze()
        # fc2
        temp = self.fc2(temp)
        # lsm        
        temp = self.lsm(temp)
        return temp

