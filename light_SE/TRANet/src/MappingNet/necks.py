import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
    
