import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from pathlib import Path
from MapNet.necks_converter import *
from MapNet.edges import *

def is_tracing():
    # Taken for pytorch for compat in 1.6.0
    """
    Returns ``True`` in tracing (if a function is called during the tracing of
    code with ``torch.jit.trace``) and ``False`` otherwise.
    """
    return torch._C._is_tracing()
"""
Default configuration from nsh's achitecture.json(2023-01-16)
"""
architecture_orig = {
    "encoder": {
        "in_channels": 6, 
        "out_channels": 64, 
        "kernel_size": 5, 
        "stride": 1, 
        "padding": "same"
    },
    "layers": {
        "n_blocks": 6
    },
    "decoder": {
        "in_features": 64, 
        "out_features": 2
    },
    "freq_modeling": {
        "conv1": {"in_channels": 64, "out_channels": 64, "kernel_size": 5, "stride": 1, "padding": 2, "groups": 8},
        "full": {"in_features": 64, "out_features": 64, "num_groups": 8, "bias": True},
        "conv2": {"in_channels": 64, "out_channels": 64, "kernel_size": 5, "stride": 1, "padding": 2, "groups": 8}
    },
    "channel_modeling": {
        "conv1": {"in_channels": 64, "out_channels": 64, "kernel_size": 5, "stride": 1, "padding": 2, "groups": 8},
        "full": {"in_features": 64, "out_features": 64, "num_groups": 8, "bias": True},
        "conv2": {"in_channels": 64, "out_channels": 64, "kernel_size": 5, "stride": 1, "padding": 2, "groups": 8}
    },
    "TGRU": {"in_channels": 64, "hidden_size": 64, "out_channels": 64, "state_size": 17},
  }

class F_ConvBlock(nn.Module):
    ''' 
    1D Convolutional block with LayerNorm and PReLU activation.
    Expacted to model the correlation between neighboring frequency bins.
    Args :
        x : shape [B,F,T,C_in]
        out : shape [B,F,T,C_out]
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups = 1):
        super(F_ConvBlock, self).__init__()
        self.ln = nn.LayerNorm(out_channels)
        self.dconv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size,1), stride=stride, padding=(padding,0), groups=in_channels)
        self.pconv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=stride)
        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        x = x.transpose(-1,1) # B*T, F, C
        x = self.ln(x)
        x = x.transpose(-1,1) # B*T, C, F
        # x = x[...,None]
        x = x.unsqueeze(-1)
        x = self.dconv(x)
        x = self.pconv(x)
        x = self.act(x)
        x = x.squeeze(-1)
        return x
    
    
class F_FullbandBlock(nn.Module):
    ''' 
    Fullband block with LayerNorm and SiLU activation.
    To leverage spatial feature of speech in narrow band, this block is used.
    SiLU is shared by all TF bins.
    Args :
        x : shape [B,F,T,C_in]
        out : shape [B,F,T,C_out]
    '''
    def __init__(self, in_features: int, out_features: int, dim_hidden : int, num_groups: int, dropout : float, bias: bool = True) -> None:
        super(F_FullbandBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.weight = nn.Parameter(torch.empty((num_groups, out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.squeeze = nn.Sequential(nn.Conv2d(in_channels=dim_hidden, out_channels=num_groups, kernel_size=(1,1)), nn.SiLU())
        self.unsqueeze = nn.Sequential(nn.Conv2d(in_channels=num_groups, out_channels=dim_hidden, kernel_size=(1,1)), nn.SiLU())
        self.ln = nn.LayerNorm(dim_hidden)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def reset_parameters(self) -> None:
        # same as linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """shape [..., group, feature]"""
        x = x.transpose(-1,1)
        x = self.ln(x)
        x = x.transpose(-1,1)
        # x = x[...,None]
        x = x.unsqueeze(-1)
        x = self.squeeze(x)
        x = x.squeeze(-1)
        if self.dropout:
            BT, C, F = x.shape
            x = x.reshape(BT,1,-1,F)
            x = x.transpose(1,3)
            x = self.dropout(x)
            x = x.transpose(1,3)
            x = x.reshape(BT,C,F)
        x = torch.einsum("...gh,gkh->...gk", x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        # x = x[...,None]
        x = x.unsqueeze(-1)
        x = self.unsqueeze(x)
        x = x.squeeze(-1)
        return x

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, num_groups={self.num_groups}, bias={True if self.bias is not None else False}"

class _MappingNet_helper(nn.Module):
    def __init__(self,
                n_rfft, 
                architecture=architecture_orig,
                skipGRU=False,
                type_TBlock = "TGRU",
                type_FBlock = "FGRU",
                type_CBlock = "None",
                ):
        super(_MappingNet_helper, self).__init__()

        self.architecture = architecture
        self.tgru_state_size = self.architecture["TGRU"]["state_size"]
        self.n_rfft = n_rfft
        self.skipGRU = skipGRU
        self.dim_hidden = self.architecture["encoder"]["out_channels"]
        self.dim_in = self.architecture["encoder"]["in_channels"]
        self.dim_out = self.architecture["decoder"]["out_features"]
        
        # Freq block
        FBlock = []
        if type_FBlock == "CrossBand" :
            FBlock.append(F_ConvBlock(**self.architecture["freq_modeling"]["conv1"])) # channel fusion?
            FBlock.append(F_FullbandBlock(**self.architecture["freq_modeling"]["full"]))
            FBlock.append(F_ConvBlock(**self.architecture["freq_modeling"]["conv2"])) # channel fusion?
        else : 
            FBlock.append(nn.Identity())
        self.FBlocks = nn.ModuleList(FBlock)
        # Channel block
        CBlock = []
        if type_CBlock == "CrossChannel" :
            CBlock.append(F_ConvBlock(**self.architecture["channel_modeling"]["conv1"]))
            CBlock.append(F_FullbandBlock(**self.architecture["channel_modeling"]["full"]))
            CBlock.append(F_ConvBlock(**self.architecture["channel_modeling"]["conv2"]))
        else : 
            CBlock.append(nn.Identity())
        self.CBlocks = nn.ModuleList(CBlock)
            
        # Time block
        self.type_TBlock = type_TBlock
        if type_TBlock == "TGRU" :
            self.tgru = TGRUBlock(**self.architecture["TGRU"],skipGRU=skipGRU)
        elif type_TBlock == "TLSTM" :
            self.tgru = TLSTMBlock(**self.architecture["TGRU"],skipGRU=skipGRU)
        elif type_TBlock == "TFGRU" :
            self.tgru = TFGRUBlock(**self.architecture["TGRU"])
        else :
            raise Exception("Unknown type_TBlock : {}".format(type_TBlock))

    def create_dummy_states(self, batch_size, device, **kwargs):

        pe_state_shape = (1, self.n_rfft, 2, 2)
        if self.type_TBlock == "TGRU" :
            shape = (self.tgru.GRU.num_layers, batch_size * self.tgru_state_size, self.tgru.GRU.hidden_size)
            return torch.zeros(*shape).to(device), torch.zeros(*pe_state_shape).to(device)
        elif self.type_TBlock == "TLSTM" :
            shape = (self.tgru.GRU.num_layers, batch_size * self.tgru_state_size, self.tgru.GRU.hidden_size)
            return (torch.zeros(*shape).to(device),torch.zeros(*shape).to(device)), torch.zeros(*pe_state_shape).to(device)
        elif self.type_TBlock == "TFGRU" :
            shape = (1, batch_size, self.tgru.GRU.hidden_size)
            return torch.zeros(*shape).to(device), torch.zeros(*pe_state_shape).to(device)
        else : 
            raise Exception("Unknown type_TBlock : {}".format(self.type_TBlock))

    def forward(self, x, tgru_state=None, pe_state=None):
        if tgru_state is None:
            tgru_state, pe_state  = self.create_dummy_states(1,x.device)
        # x.size() = [B,F,T,2]
        # feature size is equal to the number of fft bins
        # B:batch_size, T:time_steps, C:channels, F:nfft
        # x_.shape == (B,2,F,T)

        # in this scope, and between the reshapes, the shape of the data is (B,C,T,F)
        B, F, T, C = x.shape
        # modeling in frequency domain       
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, C, F)
        for m in self.FBlocks:
            x = x + m(x)
        x = x.reshape(B, T, C, F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,C]
        
        # modeling in channel domain
        x = x.permute(0, 2, 1, 3) # x_.shape == (B,F,C,T)
        x = x.reshape(B * T, F, C)
        for m in self.CBlocks:
            x = x + m(x)
        x = x.reshape(B, T, F, C)
        x = x.permute(0, 2, 1, 3)  # [B,F,T,C]
        # modeling in time domain
        x = x.permute(0, 3, 2, 1)  # [B,C,T,F]
        x_, tgru_state = self.tgru(x, tgru_state)
        x = x_ + x
        x = x.permute(0, 3, 2, 1) # x_.shape == (B,F,T,C)
        
        return x , tgru_state, pe_state

    def to_onnx(self, output_fp, device=torch.device("cpu")):
        #output_folder = output_fp.parent
        #if not os.path.exists(output_folder):
        #    os.makedirs(output_folder)

        dummy_input = torch.randn(1,self.n_rfft, 1, 2).to(device)
        dummy_states, dummy_pe_state  = self.create_dummy_states(1,device)

        torch.onnx.export(
            self,
            (dummy_input, dummy_states, dummy_pe_state),
            output_fp,
            verbose=False,
            opset_version=16,
            input_names=["inputs", "gru_state_in", "pe_state_in"],
            output_names=["outputs", "gru_state_out", "pe_state_out"],
        )

class MappingNet(nn.Module):
    def __init__(self, 
        frame_size=512, 
        hop_size=128,
        architecture = architecture_orig,
        skipGRU=False,
        type_FBlock = "FSA",
        type_TBlock = "TGRU",
        type_CBlock = "None",
        type_skip = "cat",
        PLC = False,
        PLC_alpha = 0.3,
        ):
        super().__init__()
        self.helper = _MappingNet_helper(
            frame_size // 2 + 1,
            architecture = architecture,
            skipGRU=skipGRU,
            type_TBlock=type_TBlock,
            type_FBlock=type_FBlock,
            type_CBlock=type_CBlock,
            )
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.window = torch.hann_window(self.frame_size)
        self.encoder = nn.Conv2d(**architecture["encoder"])
        self.decoder = nn.Linear(**architecture["decoder"])
        self.n_blocks = architecture["layers"]["n_blocks"]
        self.layers = []
        for i in range(self.n_blocks):
            layer = _MappingNet_helper(
                frame_size // 2 + 1,
                architecture = architecture,
                skipGRU=skipGRU,
                type_FBlock=type_FBlock,
                type_TBlock=type_TBlock,
                type_CBlock=type_CBlock,
            )
            self.add_module("layer_{}".format(i+1),layer)
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)
        # preprocessing
        if PLC :
            self.m_in = PowerLawCompression(alpha=PLC_alpha)
            self.m_out = PowerLawDecompression(alpha=PLC_alpha)
        else :
            self.m_in = nn.Identity()
            self.m_out = nn.Identity()

    def forward(self, X, tgru_state_in1, tgru_state_in2, tgru_state_in3, tgru_state_in4, tgru_state_in5, tgru_state_in6 ):

        X = self.m_in(X)
        B, F, T, C = X.shape
        X = self.encoder(X.transpose(1,-1)).transpose(1,-1)
        X, tgru_state_out1, _ = self.layers[0](X, tgru_state_in1)
        X, tgru_state_out2, _ = self.layers[1](X, tgru_state_in2)
        X, tgru_state_out3, _ = self.layers[2](X, tgru_state_in3)
        X, tgru_state_out4, _ = self.layers[3](X, tgru_state_in4)
        X, tgru_state_out5, _ = self.layers[4](X, tgru_state_in5)
        X, tgru_state_out6, _ = self.layers[5](X, tgru_state_in6)
        
        Y = self.decoder(X)
        Y = self.m_out(Y)

        return Y, tgru_state_out1, tgru_state_out2, tgru_state_out3, tgru_state_out4, tgru_state_out5, tgru_state_out6

    def enhance_speech(self, x, _aux):
        return self.forward(x)[0].detach().cpu().numpy()

    def to_onnx(self, output_fp, device=torch.device("cpu")):
        self.helper.to_onnx(output_fp, device)


def test(
    architecture=architecture_orig,
    frame_size=512,
    hop_size=128,
    kernel_type="orig"
):
    batch_size = 2
    m = MappingNet(
        frame_size=frame_size,
        hop_size=hop_size,
        architecture=architecture,
        type_FBlock="CrossBand",  # Adjusted based on your code
        type_TBlock="TGRU",
        type_CBlock="None"  # Adjusted based on your code
    )
    inputs = torch.randn(batch_size, 2, 16000)  # Adjusted input shape to match stereo input
    y = m(inputs)
    print(y.shape)
    

if __name__ == "__main__":
    import argparse
    from torchinfo import summary as summary_
    from ptflops import get_model_complexity_info
    from thop import profile
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from utils.hparams import HParam

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--default', type=str, required=True,
                        help="default configuration")
    args = parser.parse_args()

    hp = HParam(args.config, args.default, merge_except=["architecture"])
    device = torch.device("cpu")

    model = MappingNet(
        frame_size=hp.audio.n_fft,
        hop_size=hp.audio.n_hop,
        architecture=hp.model.architecture,
        skipGRU=hp.model.skipGRU,
        type_TBlock=hp.model.type_TBlock,
        type_FBlock=hp.model.type_FBlock,
        type_CBlock=hp.model.type_CBlock,
        PLC=hp.model.PLC,
        PLC_alpha=hp.model.PLC_alpha,
    ).to(device)

    num_sample = 16000
    input = torch.randn(1, 2, num_sample).to(device)

    # thop
    # MACs_thop, params_thop = profile(model, inputs=(input,), verbose=False)
    # MACs_thop, params_thop = MACs_thop / 1e6, params_thop / 1e6

    # torchinfo
    model_profile = summary_(model, input_size=(1, 2, num_sample), device=device)
    MACs_torchinfo, params_torchinfo = model_profile.total_mult_adds / 1e6, model_profile.total_params / 1e6

    # print detail
    # print(f"thop: MMac: {MACs_thop}, Params: {params_thop}")
    print(f"torchinfo: MMac: {MACs_torchinfo}, Params: {params_torchinfo}")

    # Running test function
    test(architecture=hp.model.architecture, frame_size=hp.audio.n_fft, hop_size=hp.audio.n_hop)