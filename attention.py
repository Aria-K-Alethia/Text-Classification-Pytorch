# Copyright (c) 2018 Aria-K-Alethia@github.com
# Licence: MIT
# Global Attention
# Any use of this code should display all the info above; the user agrees to assume all liability for the use of this code

import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import mask_sequence


class GlobalAttention(nn.Module):
    """
        overview:
            Global Attention
            Using the global info in the input sequence
            The complexity would increase with the increasing
            of sequence length
        params:
            hidden_size: the hidden size of decoder
            attn_type: attention type, should be one of dot,general,mlp
            device: torch.device
    """
    def __init__(self, hidden_size, device, attn_type = 'general'):
        super(GlobalAttention, self).__init__()
        assert attn_type in ['general', 'dot', 'mlp']
        self.hidden_size = hidden_size
        self.attn_type = attn_type
        self.device = device
        # score layer
        if(attn_type == 'general'):
            # Doesn't need bias in this layer
            self.general_linear = nn.Linear(hidden_size, hidden_size, bias = False)
        elif(attn_type == 'mlp'):
            self.in_linear = nn.Linear(hidden_size, hidden_size, bias = False)
            self.query_linear = nn.Linear(hidden_size, hidden_size, bias = True)
            self.v = nn.Linear(hidden_size, 1, bias = False)
        # softmax
        self.softmax = nn.Softmax(dim=-1)
        # mlp needs to have bias in output layer
        with_bias = (attn_type == 'mlp')
        self.linear_out = nn.Linear(hidden_size*2, hidden_size, bias = with_bias)
        # non-linear function
        self.tanh = nn.Tanh()

    def score(self, decoder_outputs, encoder_outputs):
        '''
            overview:
                calc the attention score between encoder_outputs
                and decoder_outputs
            params:
                decoder_outputs: [#batch_size, #outlen, #hidden_size]
                encoder_outputs: [#batch_size, #inlen, #hidden_size]
            return:
                attn_score: [#batch_size, #outlen, #inlen]
        '''
        _batch1, _len1, _dim1 = encoder_outputs.shape
        _batch2, _len2, _dim2 = decoder_outputs.shape
        if(self.attn_type == 'dot'):
            _encoder_outputs = encoder_outputs.transpose(2, 1)
            return torch.bmm(decoder_outputs, _encoder_outputs)
        elif(self.attn_type == 'general'):
            _decoder_outputs = self.general_linear(decoder_outputs) #[#batch_size,#outlen,#hidden_size]
            _encoder_outputs = encoder_outputs.transpose(2, 1)
            return torch.bmm(_decoder_outputs, _encoder_outputs)
        elif(self.attn_type == 'mlp'): # mlp
            _encoder_outputs = self.in_linear(encoder_outputs)
            _decoder_outputs = self.query_linear(decoder_outputs)
            _encoder_outputs = _encoder_outputs.view(_batch1, 1, _len1, _dim1)
            _encoder_outputs = _encoder_outputs.expand(_batch1, _len2, _len1, _dim1)
            _decoder_outputs = _decoder_outputs.view(_batch2, _len2, 1, _dim2)
            _decoder_outputs = _decoder_outputs.expand(_batch2, _len2, _len1, _dim2)
            output = _encoder_outputs + _decoder_outputs
            del _encoder_outputs, _decoder_outputs
            output = self.tanh(output)
            output = self.v(output).view(_batch2, _len2, _len1)
            return output
            

    def forward(self, encoder_outputs, decoder_outputs, lengths = None,
                context_only = False):
        '''
            overview:
                forward method for GlobalAttention
            params:
                encoder_outputs: [#batch_size, #inlen, #hidden_size]
                decoder_outputs: [#batch_size, #outlen, #hidden_size]
                lengths: the length of input sequence
                context_only: if set , only return context vector and
                              attention distribution, default false
            return:
                new_decoder_outputs: [#outlen, #batch_size, #hidden_size]
                attn_dis: [#outlen, #batch_size, #inlen]
                or if context_only is set
                context_vector: [#outlen, #batch_size, #hidden_size]
                attn_dis: [#outlen, #batch_size, #inlen]
            NOTE:
                1. The two input state may be discontiguous
                2. This function also supports one step query, i.e. [#batch_size, #hidden_size]
                   In this case the context_vector would be [#batch_size, #inlen]
        '''
        one_step = False
        if(decoder_outputs.dim() == 2):
            one_step = True
            decoder_outputs = decoder_outputs.unsqueeze(1) # [batch_size, 1, #hidden_size]
        
        #[#batch_size, #outlen, #inlen]
        attn_score = self.score(decoder_outputs, encoder_outputs)
        if(lengths is not None):
            mask = mask_sequence(lengths, max_length = max_length, device = self.device) # [#batch_size, #inlen]
            mask = mask.unsqueeze(1)
            attn_score.data.masked_fill_(1 - mask, -float('inf'))
        # softmax
        attn_dis = self.softmax(attn_score)
        # get the context vector
        context_vector = torch.bmm(attn_dis, encoder_outputs)
        # if context_only is set, we only want to get the context vector
        # and attention distribution 
        if(context_only):
            if(one_step):
                # [#batch_size, #hidden_size]
                context_vector = context_vector.squeeze(1)
            else:
                # [#outlen, #batch_size, #hidden_size]
                context_vector = context_vector.transpose(1, 0).contiguous()
            return context_vector, attn_dis
        # concat the context and hidden state
        concat = torch.cat([context_vector, decoder_outputs], 2)
        new_decoder_outputs = self.linear_out(concat)
        # general and dot need to have a non-linear
        if(self.attn_type in ['general', 'dot']):
            new_decoder_outputs = self.tanh(new_decoder_outputs)
        # if one step squeeze the two output vector
        if(one_step):
            context_vector = context_vector.squeeze(1)
            new_decoder_outputs = new_decoder_outputs.squeeze(1)
        # if multi-step transpose the vectors
        else:
            context_vector = context_vector.transpose(1, 0).contiguous()
            new_decoder_outputs = new_decoder_outputs.transpose(1, 0).contiguous()
        # return
        return new_decoder_outputs, attn_dis

