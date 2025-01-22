import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
import nobuco
from nobuco import ChannelOrderingStrategy
from nobuco.converters.node_converter import converter
import math
import keras

# def _no_grad_uniform_(tensor, a, b):
#     with torch.no_grad():
#         return tensor.uniform_(a, b)
    
# class CustomGRU(nn.Module):
#     """
#     Custom GRU implementation
#     """
#     def __init__(self, in_channels, hidden_size, batch_first=True, bidirectional=False,**kwargs):
#         super(CustomGRU, self).__init__()
#         self.in_channels = in_channels
#         self.hidden_size = hidden_size
#         # Parameters
#         self.weight_ih_l0 = torch.nn.Parameter(torch.empty((3*hidden_size, in_channels),dtype=torch.float32), requires_grad=True)
#         self.weight_hh_l0 = nn.Parameter(torch.empty((3*hidden_size, hidden_size),dtype=torch.float32), requires_grad=True)
#         self.bias_ih_l0 = nn.Parameter(torch.empty((3*hidden_size),dtype=torch.float32), requires_grad=True)
#         self.bias_hh_l0 = nn.Parameter(torch.empty((3*hidden_size),dtype=torch.float32), requires_grad=True)
#         self.reset_parameters()
#         self.batch_first = batch_first
#         self.bidirectional = bidirectional

#     def reset_parameters(self) -> None:
#         stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
#         for weight in self.parameters():
#             _no_grad_uniform_(weight, -stdv, stdv)

#     def forward(self, x, h=None):
#         batch_size, seq_len, _ = x.size()
#         hidden_seq = []

#         if h is None:
#             hidden = torch.zeros(batch_size,1,self.hidden_size).to(x.device).transpose(0, 1)
#         else:
#             hidden = h.transpose(0, 1)
#         w_ir, w_iz, w_in = self.weight_ih_l0.chunk(3, 0)
#         w_hr, w_hz, w_hn = self.weight_hh_l0.chunk(3, 0)
#         b_ir, b_iz, b_in = self.bias_ih_l0.chunk(3, 0)
#         b_hr, b_hz, b_hn = self.bias_hh_l0.chunk(3, 0)
#         '''unroll the sequence
#         for t in range(seq_len):
#             x_t = x[:, t, :].unsqueeze(1)
#             resetgate = torch.sigmoid(x_t @ w_ir.t() + b_ir + hidden @ w_hr.t() + b_hr)
#             updategate = torch.sigmoid(x_t @ w_iz.t() + b_iz + hidden @ w_hz.t() + b_hz)
#             newgate = torch.tanh(x_t @ w_in.t() + b_in + resetgate * (hidden @ w_hn.t() + b_hn))
            
#             hidden = (1 - updategate) * newgate + updategate * hidden
#             hidden_seq.append(hidden)
#         '''
#         # unroll the sequence
#         for t in range(seq_len):
#             x_t = x[:, t, :].unsqueeze(1)
#             r = torch.sigmoid(x_t @ w_ir.t() + b_ir + hidden @ w_hr.t() + b_hr)
#             z = torch.sigmoid(x_t @ w_iz.t() + b_iz + hidden @ w_hz.t() + b_hz)
#             n = torch.tanh(x_t @ w_in.t() + b_in + r * (hidden @ w_hn.t() + b_hn))
#             hidden = (1 - z) * n + z * hidden
#             hidden_seq.append(hidden)
#         # stack hidden_seq
#         hidden_seq = torch.cat(hidden_seq, dim=0)
#         hidden = hidden.transpose(0, 1).contiguous()

#         return hidden_seq, hidden
import numpy as np

from nobuco.commons import ChannelOrder, ChannelOrderingStrategy
from nobuco.converters.node_converter import converter


class Bidirectional:
    def __init__(self, layer, backward_layer):
        self.layer = layer
        self.backward_layer = backward_layer

    def __call__(self, x, initial_state=None):
        if initial_state is not None:
            half = len(initial_state) // 2
            state_f = initial_state[:half]
            state_b = initial_state[half:]
        else:
            state_f = None
            state_b = None

        ret_f = self.layer(x, state_f)
        ret_b = self.backward_layer(x, state_b)
        y_f, h_f = ret_f[0], ret_f[1:]
        y_b, h_b = ret_b[0], ret_b[1:]
        if self.layer.time_major:
            reverse_axis = 0
        else:
            reverse_axis = 1
        y_b = tf.reverse(y_b, axis=(reverse_axis,))
        y_cat = tf.concat([y_f, y_b], axis=-1)
        return y_cat, *h_f, *h_b

@converter(nn.GRU, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_GRU(self: nn.GRU, input, hx=None):
    bidirectional = self.bidirectional
    num_layers = self.num_layers

    def create_layer(i, reverse):

        def reorder(param):
            assert param.shape[-1] % 3 == 0
            p1, p2, p3 = np.split(param, 3, axis=-1)
            return np.concatenate([p2, p1, p3], axis=-1)

        suffix = '_reverse' if reverse else ''
        weight_ih = self.__getattr__(f'weight_ih_l{i}{suffix}').cpu().detach().numpy().transpose((1, 0))
        weight_hh = self.__getattr__(f'weight_hh_l{i}{suffix}').cpu().detach().numpy().transpose((1, 0))
        weight_ih = reorder(weight_ih)
        weight_hh = reorder(weight_hh)
        weights = [weight_ih, weight_hh]

        if self.bias:
            bias_ih = self.__getattr__(f'bias_ih_l{i}{suffix}').cpu().detach().numpy()
            bias_hh = self.__getattr__(f'bias_hh_l{i}{suffix}').cpu().detach().numpy()
            bias_ih = reorder(bias_ih)
            bias_hh = reorder(bias_hh)
            weights += [np.stack([bias_ih, bias_hh], axis=0)]

        gru = keras.layers.GRU(
            units=self.hidden_size,
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=self.bias,
            dropout=self.dropout,
            return_sequences=True,
            return_state=True,
            time_major=not self.batch_first,
            reset_after=True,
            unroll=True,
            go_backwards=reverse,
            weights=weights,
        )
        return gru

    def convert_initial_states(hx):
        if hx is not None:
            h0 = tf.reshape(hx, (num_layers, -1, *hx.shape[1:]))
            initial_states = []
            for i in range(num_layers):
                if bidirectional:
                    state = (h0[i][0], h0[i][1])
                else:
                    state = h0[i][0]
                initial_states.append(state)
            return initial_states
        else:
            return None

    layers = []
    for i in range(num_layers):
        layer = create_layer(i, reverse=False)
        if bidirectional:
            layer_reverse = create_layer(i, reverse=True)
            # layer = keras.layers.Bidirectional(layer=layer, backward_layer=layer_reverse)
            layer = Bidirectional(layer=layer, backward_layer=layer_reverse)
        layers.append(layer)

    no_batch = input.dim() == 2

    def func(input, hx=None):
        x = input

        if no_batch:
            x = x[None, :, :]
            if hx is not None:
                hx = hx[:, None, :]

        initial_states = convert_initial_states(hx)

        hxs = []
        for i in range(num_layers):
            state = initial_states[i] if initial_states else None
            ret = layers[i](x, initial_state=state)
            x, hxo = ret[0], ret[1:]
            hxs += hxo
        hxs = tf.stack(hxs, axis=0)

        if no_batch:
            x = x[0, :, :]
            hxs = hxs[:, 0, :]

        return x, hxs
    return func


class FGRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels):
        super(FGRUBlock, self).__init__()
        self.GRU = nn.GRU(
            in_channels, hidden_size, batch_first=True, bidirectional=True
        )
        # the GRU is bidirectional -> multiply hidden_size by 2
        self.conv = nn.Conv2d(hidden_size * 2, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()

    def forward(self, x):
        """x has shape (batch * timesteps, number of features, feature_size)"""
        # We want the FGRU to consider the F axis as the one that unrolls the sequence,
        #  C the input_size, and BT the effective batch size --> x_.shape == (B,C,T,F)
        B, C, T, F = x.shape
        x_ = x.permute(0, 2, 3, 1)  # x_.shape == (B,T,F,C)
        x_ = x_.reshape(B * T, F, C)
        y, h = self.GRU(x_)  # x_.shape == (BT,F,C)
        y = y.reshape(B, T, F, self.hidden_size * 2)
        output = y.permute(0, 3, 1, 2)  # output.shape == (B,C,T,F)
        output = self.conv(output)
        output = self.bn(output)
        return self.relu(output)
    
class FLSTMBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels):
        super(FLSTMBlock, self).__init__()
        self.GRU = nn.LSTM(
            in_channels, hidden_size, batch_first=True, bidirectional=True
        )
        # the GRU is bidirectional -> multiply hidden_size by 2
        self.conv = nn.Conv2d(hidden_size * 2, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()

    def forward(self, x):
        """x has shape (batch * timesteps, number of features, feature_size)"""
        # We want the FGRU to consider the F axis as the one that unrolls the sequence,
        #  C the input_size, and BT the effective batch size --> x_.shape == (B,C,T,F)
        B, C, T, F = x.shape
        x_ = x.permute(0, 2, 3, 1)  # x_.shape == (B,T,F,C)
        x_ = x_.reshape(B * T, F, C)
        y, h = self.GRU(x_)  # x_.shape == (BT,F,C)
        y = y.reshape(B, T, F, self.hidden_size * 2)
        output = y.permute(0, 3, 1, 2)  # output.shape == (B,C,T,F)
        output = self.conv(output)
        output = self.bn(output)
        return self.relu(output)

class TGRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels,num_layers=1 ,skipGRU=False,**kwargs):
        super(TGRUBlock, self).__init__()

        if not skipGRU : 
            self.GRU = nn.GRU(in_channels, hidden_size, batch_first=True, num_layers=num_layers)
            # self.GRU = CustomGRU(in_channels, hidden_size, batch_first=True, bidirectional=True)
        else : 
            self.GRU = SkipGRU(in_channels, hidden_size, batch_first=True)
        self.conv = nn.Conv2d(hidden_size, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()

    def forward(self, x, rnn_state):
        # We want the GRU to consider the T axis as the one that unrolls the sequence,
        #  C the input_size, and BS the effective batch size --> x_.shape == (BS,T,C)
        # B = batch_size, T = time_steps, C = num_channels, S = feature_size
        #  (using S for feature_size because F is taken by nn.functional)
        B, C, T, F = x.shape  # x.shape == (B, C, T, F)

        # unpack, permute, and repack
        x1 = x.permute(0, 3, 2, 1)  # x2.shape == (B,F,T,C)
        x_ = x1.reshape(B * F, T, C)  # x_.shape == (BF,T,C)

        # run GRU
        y_, rnn_state = self.GRU(x_, rnn_state)  # y_.shape == (BF,T,C)
        # unpack, permute, and repack
        y1 = y_.reshape(B, F, T, self.hidden_size)  # y1.shape == (B,F,T,C)
        y2 = y1.permute(0, 3, 2, 1)  # y2.shape == (B,C,T,F)

        output = self.conv(y2)
        output = self.bn(output)
        output = self.relu(output)
        return output, rnn_state
    
# Temporal Fullband GRU
# Need to modify ONNX related codes
class TFGRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, state_size=7, **kwargs):
        super(TFGRUBlock, self).__init__()

        self.GRU = nn.GRU(in_channels*state_size, hidden_size*state_size, batch_first=True)
        self.conv = nn.Conv2d(hidden_size, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.relu = nn.ReLU()

    def forward(self, x, rnn_state):
        # We want the GRU to consider the T axis as the one that unrolls the sequence,
        #  C the input_size, and BS the effective batch size --> x_.shape == (BS,T,C)
        # B = batch_size, T = time_steps, C = num_channels, S = feature_size
        #  (using S for feature_size because F is taken by nn.functional)
        B, C, T, F = x.shape  # x.shape == (B, C, T, F)

        # rnn_state : [1,B, C*F]

        # unpack, permute, and repack
        x1 = x.permute(0, 2, 1, 3)  # x2.shape == (B,T,C,F)
        x_ = x1.reshape(B, T, C*F)  # x_.shape == (B,T,C*F)

        # run GRU
        y_, rnn_state = self.GRU(x_, rnn_state)  # y_.shape == (B,T,C*F)
        # unpack, permute, and repack
        y1 = y_.reshape(B, T, C, F)  # y1.shape == (B,F,T,C)
        y2 = y1.permute(0, 2, 1, 3)  # y2.shape == (B,C,T,F)

        output = self.conv(y2)
        output = self.bn(output)
        output = self.relu(output)
        return output, rnn_state

# TODO : Temporal Subband GRU
# Need to modify ONNX related codes
class TSGRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, skipGRU=False,n_band=7, **kwargs):
        super(TSGRUBlock, self).__init__()

        self.GRU = nn.GRU(in_channels, hidden_size, batch_first=True)
        self.conv = nn.Conv2d(hidden_size, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()

    def forward(self, x, rnn_state):
        # We want the GRU to consider the T axis as the one that unrolls the sequence,
        #  C the input_size, and BS the effective batch size --> x_.shape == (BS,T,C)
        # B = batch_size, T = time_steps, C = num_channels, S = feature_size
        #  (using S for feature_size because F is taken by nn.functional)
        B, C, T, F = x.shape  # x.shape == (B, C, T, F)


        # rnn_state : [1,B, C, F]

        # unpack, permute, and repack
        x1 = x.permute(0, 3, 2, 1)  # x2.shape == (B,F,T,C)
        x_ = x1.reshape(B * F, T, C)  # x_.shape == (BF,T,C)

        # run GRU
        y_, rnn_state = self.GRU(x_, rnn_state)  # y_.shape == (BF,T,C)
        # unpack, permute, and repack
        y1 = y_.reshape(B, F, T, self.hidden_size)  # y1.shape == (B,F,T,C)
        y2 = y1.permute(0, 3, 2, 1)  # y2.shape == (B,C,T,F)

        output = self.conv(y2)
        output = self.bn(output)
        output = self.relu(output)
        return output, rnn_state
 
    
class TLSTMBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, skipGRU=False,num_layers=1,**kwargs):
        super(TLSTMBlock, self).__init__()

        self.GRU= nn.LSTM(in_channels, hidden_size, batch_first=True,num_layers=num_layers)
        self.conv = nn.Conv2d(hidden_size, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()

    def forward(self, x, rnn_state):
        # We want the GRU to consider the T axis as the one that unrolls the sequence,
        #  C the input_size, and BS the effective batch size --> x_.shape == (BS,T,C)
        # B = batch_size, T = time_steps, C = num_channels, S = feature_size
        #  (using S for feature_size because F is taken by nn.functional)
        B, C, T, F = x.shape  # x.shape == (B, C, T, F)

        # unpack, permute, and repack
        x1 = x.permute(0, 3, 2, 1)  # x2.shape == (B,F,T,C)
        x_ = x1.reshape(B * F, T, C)  # x_.shape == (BF,T,C)

        # run GRU
        y_, rnn_state = self.GRU(x_, rnn_state)  # y_.shape == (BF,T,C)
        # unpack, permute, and repack
        y1 = y_.reshape(B, F, T, self.hidden_size)  # y1.shape == (B,F,T,C)
        y2 = y1.permute(0, 3, 2, 1)  # y2.shape == (B,C,T,F)

        output = self.conv(y2)
        output = self.bn(output)
        output = self.relu(output)
        return output, rnn_state
    
class T_FGRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels,**kwargs):
        super(T_FGRUBlock, self).__init__()
        self.hidden_size = hidden_size

        self.GRU = nn.GRU(in_channels, hidden_size, batch_first=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    # TODO : if it works, implement forward with hidden like TGRUBlock
    def forward(self, x, rnn_state=None):
        B, C, T, F = x.shape  # x.shape == (B, C, T, F)

        # unpack, permute, and repack
        x_ = x.reshape(B * C, T, F)  # x_.shape == (BC,T,F)

        # run GRU
        y_, rnn_state = self.GRU(x_, rnn_state)  # y_.shape == (BF,T,C)
        # unpack, permute, and repack
        y = y_.reshape(B, C, T, F)  # y1.shape == (B,F,T,C)

        output = self.bn(y)
        output = self.relu(output)
        # for now
        #return output, rnn_state
        return output

class FSABlock(nn.Module):
    def __init__(self,in_channels,hidden_size,out_channels,dropout=0.0) :
        super(FSABlock,self).__init__()
        
        self.SA = nn.MultiheadAttention(in_channels, hidden_size, batch_first = True,dropout=dropout)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        # x : [B,C,T,F] -> [B*T,F,C]
        B, C, T, F = x.shape
        x_ = x.permute(0, 2, 3, 1)  # x_.shape == (B,T,F,C)
        x_ = x_.reshape(B * T, F, C)
        
        y,h = self.SA(x_,x_,x_)
        
        y = y.reshape(B, T, F, C)
        output = y.permute(0, 3, 1, 2)  # output.shape == (B,C,T,F)
        output = self.bn(output)
        output = self.relu(output)
        return output
    
class FSA2Block(nn.Module):
    def __init__(self,in_channels,hidden_size,out_channels,dropout=0.0) :
        super(FSA2Block,self).__init__()
        
        self.SA = nn.MultiheadAttention(in_channels, hidden_size, batch_first = True,dropout=dropout)
        self.bn = nn.BatchNorm2d(in_channels)
        self.FF = nn.Linear(in_channels,in_channels)
        self.relu = nn.PReLU()
        
    def forward(self,x):
        # x : [B,C,T,F] -> [B*T,F,C]
        B, C, T, F = x.shape
        x_ = x.permute(0, 2, 3, 1)  # x_.shape == (B,T,F,C)
        x_ = x_.reshape(B * T, F, C)
        
        y,h = self.SA(x_,x_,x_)
        
        y = y.reshape(B, T, F, C)
        y = self.FF(y)
        output = y.permute(0, 3, 1, 2)  # output.shape == (B,C,T,F)
        output = self.bn(output)
        output = self.relu(output)
        return output
    
class FSA3Block(nn.Module):
    def __init__(self,in_channels,hidden_size,out_channels,dropout=0.0) :
        super(FSA3Block,self).__init__()


        self.pc = nn.Conv2d(in_channels, in_channels, kernel_size=(1,1))
        self.bnc = nn.BatchNorm2d(in_channels)
        
        self.SA = nn.MultiheadAttention(in_channels, hidden_size, batch_first = True,dropout=dropout)
        self.bnsa = nn.BatchNorm2d(in_channels)

        self.relu = nn.ReLU()
        
    def forward(self,x):
        # x : [B,C,T,F] -> [B*T,F,C]
        B, C, T, F = x.shape

        yc = self.pc(x)
        yc = self.bnc(yc)

        x_ = x.permute(0, 2, 3, 1)  # x_.shape == (B,T,F,C)
        x_ = x_.reshape(B * T, F, C)
        
        ysa,h = self.SA(x_,x_,x_)
        
        ysa = ysa.reshape(B, T, F, C)
        ysa = ysa.permute(0, 3, 1, 2)  # output.shape == (B,C,T,F)
        ysa = self.bnsa(ysa)
        ysa = self.relu(ysa)

        output = yc + ysa

        return output

#  Channel wise Attetnion
class CSABlock(nn.Module):
    def __init__(self,in_channels,hidden_size,out_channels,dropout=0.0) :
        super(CSABlock,self).__init__()
        
        self.SA = nn.MultiheadAttention(in_channels, hidden_size, batch_first = True,dropout=dropout)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.SiLU()
        
    def forward(self,x):
        # x : [B,C,T,F] -> [B*T,C,F]
        B, C, T, F = x.shape
        x_ = x.permute(0, 2, 1, 3)  # x_.shape == (B,T,C,F)
        x_ = x_.reshape(B * T, C, F)

        y,h = self.SA(x_,x_,x_)
        y = self.bn(y)
        
        y = y.reshape(B, T, C, F)
        output = y.permute(0, 2, 1, 3)  # output.shape == (B,C,T,F)
        #output = self.relu(output) 
        return output   

# Frequence Axis Transformer
class FATBlock(nn.Module):
    def __init__(self,in_channels,dropout=0.0) :
        super(FATBlock,self).__init__()
        
        self.T = nn.Transformer(in_channels, batch_first = True,dropout=dropout)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        # x : [B,C,T,F] -> [B*T,F,C]
        B, C, T, F = x.shape
        x_ = x.permute(0, 2, 3, 1)  # x_.shape == (B,T,F,C)
        x_ = x_.reshape(B * T, F, C)
        
        y = self.T(x_,x_)
        
        y = y.reshape(B, T, F, C)
        output = y.permute(0, 3, 1, 2)  # output.shape == (B,C,T,F)
        output = self.bn(output)
        output = self.relu(output)
        return output
    
