# Copyright (c) 2018 Aria-K-Alethia@github.com
# Licence: MIT
# Attention RNN for text classification
# Any use of this code should display all the info above; the user agrees to assume all liability for the use of this code

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import build_rnn
from textCNN_Pytorch.attention import GlobalAttention


class AttnRNN(nn.Module):
    '''
        overview:
            Naive Attention RNN for textclassification
        params:
            rnn_type: 'LSTM' or 'GRU' or 'RNN'
            rnn_size: hidden layer size
            rnn_layers: the layer of rnn
            embedding: the embedding matrix
            bidirectional: bool, number of direction of rnn
            embedding_size: embedding size, default is 300
            dropout: dropout ratio, default is 0.5
            label_number: label number, default is 19
            attn_type: attention type, should be 'general', 'dot'
            batch_first: batch first for the input 
            device: torch.device
    '''
    def __init__(self, rnn_type, rnn_size, rnn_layers, embedding, bidirectional, device,
        embedding_size = 300, dropout = 0.5, label_number = 19, attn_type = 'general',
        batch_first = True):
        super(AttnRNN, self).__init__()
        assert rnn_type in ['LSTM', 'GRU', 'RNN']
        self.name = 'AttnRNN'
        self.rnn_type = rnn_type    
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.embedding = embedding
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.label_number = label_number
        self.attn_type = attn_type
        self.batch_first = batch_first
        self.direction = 2 if bidirectional else 1
        self.device = device
        # parts
        self.rnn = build_rnn(rnn_type,
                        input_size = embedding_size, hidden_size = rnn_size,
                        num_layers = rnn_layers, batch_first = batch_first,
                        dropout = dropout, bidirectional = bidirectional)
        self.dummy = nn.Parameter(torch.ones(1, rnn_size * self.direction)) 
        self.attn = GlobalAttention(rnn_size * self.direction, device, attn_type = attn_type)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(rnn_size * self.direction, label_number)
        self.lsm = nn.LogSoftmax(-1)
    
    def forward(self, x):
        '''
            overview:
                forward method of Attention RNN
            params:
                x: [#batch, #length]
            return:
                [#batch, #label_number]
        '''
        # get the length
        lengths = (x != 0).sum(1).long()
        # embedding
        xe = self.embedding(x)
        # rnn
        outputs, final = self.rnn(xe)
        # attn
        outputs, attn_dis = self.attn(outputs, self.dummy.expand(outputs.shape[0], -1), lengths)
        # fc
        outputs = self.dropout(outputs)
        outputs = self.fc1(outputs)
        # output
        outputs = self.lsm(outputs)
        return outputs
