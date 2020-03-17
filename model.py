import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.fc = nn.Linear(n_head * d_v, d_model)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, x, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size, seq_len = x.size(0), x.size(1)

        original = x
        x = self.layer_norm(x)
        q = self.w_qs(x).view(batch_size, seq_len, n_head, d_k)
        k = self.w_ks(x).view(batch_size, seq_len, n_head, d_k)
        v = self.w_vs(x).view(batch_size, seq_len, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        output, attn = self.attention(q, k, v, mask=mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.dropout(self.fc(output))
        output += original

        return output, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_inner)
        self.w_2 = nn.Linear(d_inner, d_model)
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        original = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += original

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, x.size(1)].clone().detach()

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn

class Encoder(nn.Module):
    def __init__(self, d_feature, subsample, d_model, d_inner, n_head, d_k, d_v, n_block, dropout=0.1):
        super(Encoder, self).__init__()
        self.subsample = subsample

        self.fc = nn.Linear(d_feature * subsample, d_model)
        self.pe = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList(
                [EncoderLayer(d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout) 
                    for _ in range(n_block)])
        self.layer_norm = LayerNorm(d_model)

    def forward(self, x, return_attn=False):
        # x = [batch_size, seq_len, feature]
        new_seq_len = x.size(1) // self.subsample * self.subsample
        # discard last few frames
        x = x[:, :new_seq_len]
        # transpose and reshape to [batch_size, new_seq_len, features]
        x = x.reshape(x.size(0), new_seq_len // self.subsample, -1)
        enc_output = self.dropout(self.pe(self.fc(x)))

        enc_slf_attn_list = []
        for layer in self.encoder_layers:
            enc_output, enc_slf_attn = layer(enc_output)
            enc_slf_attn_list += [enc_slf_attn] if return_attn else []

        enc_output = self.layer_norm(enc_output)
        if return_attn:
            return enc_output, enc_slf_attn_list
        else:
            return enc_output, 

