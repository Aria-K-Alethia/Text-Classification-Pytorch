# Copyright (c) 2018 Aria-K-Alethia@github.com
# Licence: MIT
# Attention RNN for text classification
# Any use of this code should display all the info above; the user agrees to assume all liability for the use of this code

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import build_rnn
from textCNN_Pytorch.attention import GlobalAttention

class MemoryNetwork(nn.Module):
    '''
        overview:
            Memory Network for text classification
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
            episode: number of episode 
    '''
    def __init__(self, rnn_type, rnn_size, rnn_layers, embedding, bidirectional, device,
        embedding_size = 300, dropout = 0.5, label_number = 19, attn_type = 'general',
        batch_first = True, episode = 2):
        super(AttnRNN, self).__init__()
        assert rnn_type in ['LSTM', 'GRU', 'RNN']
        self.name = 'MemoryNetwork'
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
        self.query = nn.Parameter(torch.ones(1, rnn_size * self.direction))
        self.enc_linear = nn.Linear(rnn_size * self.direction, rnn_size * self.direction)
        self.query_linear = nn.Linear(rnn_size * self.direction, rnn_size * self.direction)
        self.attn = GlobalAttention(rnn_size * self.direction, device, attn_type = attn_type)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(rnn_size * self.direction, label_number)
        self.lsm = nn.LogSoftmax(-1)
    def update_memory(self, enc_state):
        def _transform(state):
            if self.direction == 2:
                state = torch.cat([state[0:state.shape[0]:2], state[1:state.shape[0]:2]], 2)
            return state
        if isinstance(enc_state, tuple):
            temp = [_transform(item) for item in enc_state]
        else:
            temp = [_transform(enc_state)]
        temp = [self.enc_linear(item) + self.query_linear(self.query) for item in temp]
        temp = [F.relu(item) for item in temp]
        return tuple(temp) if len(temp) == 2 else temp[0]
    
    def forward(self, x):
        '''
            overview:
                forward method of Memory Network
            params:
                x: [#batch, #length]
            return:
                [#batch, #label_number]
        '''
        # get the length
        lengths = (x != 0).sum(1).long() 
        # embedding
        xe = self.embedding(x)
        # episodic rnn
        enc_state = None
        for i in range(self.episode):
            outputs, enc_state = self.rnn(xe, enc_state)
            enc_state = self.update_memory(enc_state)
        # attn
        outputs, attn_dis = self.attn(outputs, self.dummy.expand(outputs.shape[0], -1), lengths)
        # fc
        outputs = self.dropout(outputs)
        outputs = self.fc1(outputs)
        # output
        outputs = self.lsm(outputs)
        return outputs
