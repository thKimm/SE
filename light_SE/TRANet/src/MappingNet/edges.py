import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
EPS = 1e-9

class ComplexConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=1,
        groups=1,
        causal=True,
        complex_axis=1,
        bias=True
    ):
        super(ComplexConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.causal = causal
        self.groups = groups
        self.dilation = dilation
        self.complex_axis = complex_axis
        self.real_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride, padding=[self.padding[0], 0], dilation=self.dilation, groups=self.groups,bias=bias)
        self.imag_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride, padding=[self.padding[0], 0], dilation=self.dilation, groups=self.groups,bias=bias)

        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)

        if bias : 
            nn.init.constant_(self.real_conv.bias, 0.)
            nn.init.constant_(self.imag_conv.bias, 0.)

    def forward(self, inputs):
        if self.padding[1] != 0 and self.causal:
            inputs = F.pad(inputs, [self.padding[1], 0, 0, 0])
        else:
            inputs = F.pad(inputs, [self.padding[1], self.padding[1], 0, 0])

        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)

        else:
            if isinstance(inputs, torch.Tensor):
                real, imag = torch.chunk(inputs, 2, self.complex_axis)
            
            real2real = self.real_conv(real)
            imag2imag = self.imag_conv(imag)

            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)
        return out


def complex_cat(inp, dim=1):
    reals, imags = torch.chunk(inp, 2, dim)
    return reals, imags

class ComplexLinearProjection(nn.Module):
    def __init__(self):
        super(ComplexLinearProjection, self).__init__()

    def forward(self, real, imag):
        """
        real, imag: B C F T
        """
        #inputs = torch.cat([real, imag], 1)
        #outputs = self.clp(inputs)
        #real, imag = outputs.chunk(2, dim=1)
        outputs = torch.sqrt(real**2+imag**2+1e-8)
        return outputs

class PhaseEncoderV0(nn.Module):
    def __init__(self, in_channels=1, out_channels = 4,type_norm="None", alpha=0.5,bias=True):
        super(PhaseEncoderV0, self).__init__()
        self.identity = nn.Identity()

    def forward(self, X):
        """
        X : [B,C(re,im),F,T]
        """
        return self.identity(X)

class PhaseEncoderV1(nn.Module):
    def __init__(self, in_channels=1, out_channels = 4,type_norm="None", alpha=0.5,bias=True):
        super(PhaseEncoderV1, self).__init__()
        self.complexnn = nn.Sequential(
                    nn.ConstantPad2d((0, 0, 0, 0), 0.0),
                    ComplexConv2d(in_channels, out_channels, (1, 3))
                )
        self.clp = ComplexLinearProjection()
        self.alpha = alpha

    def forward(self, X):
        """
        X : [B,C(re,im),F,T]
        """
        outs = self.complexnn(X)
        real, imag = complex_cat(outs, dim=1)
        amp = self.clp(real, imag)
        return amp**self.alpha

class PhaseEncoderV2(nn.Module):
    def __init__(self, in_channels = 1, out_channels=4, type_norm = "BatchNorm2d"):
        super(PhaseEncoderV2, self).__init__()
        self.complexnn = ComplexConv2d(in_channels, out_channels, (1, 3))

        if type_norm == "BatchNorm2d" : 
            self.norm = nn.BatchNorm2d(out_channels*2)
        elif type_norm == "InstanceNorm2d" : 
            self.norm = nn.InstanceNorm2d(out_channels*2,track_running_stats=True)
        self.cplx2real = nn.Conv2d(out_channels*2,out_channels,1)

    def forward(self, X):
        """
        X : [B,C(re,im),F,T]
        """
        outs = self.complexnn(X)
        outs = self.norm(outs)
        outs = self.cplx2real(outs)
        return outs
    
class PhaseEncoderV3(nn.Module):
    def __init__(self, out_channels, in_channels=1,type_norm = "BatchNorm2d",bias=True):
        super(PhaseEncoderV3, self).__init__()
        self.complexnn_depth = ComplexConv2d(in_channels, in_channels, (1, 3))
        if type_norm == "BatchNorm2d" : 
            self.norm = nn.BatchNorm2d(in_channels*2)
        elif type_norm == "InstanceNorm2d" : 
            self.norm = nn.InstanceNorm2d(in_channels*2,track_running_stats=True)
        self.complexnn_point = ComplexConv2d(in_channels, out_channels, (1, 1),bias=bias)
        self.cplx2real = nn.Conv2d(out_channels*2,out_channels,1,bias=bias)

    def forward(self, X):
        """
        X : [B,C(re,im),F,T]
        """
        outs = self.complexnn_depth(X)
        outs = self.norm(outs)
        outs = self.complexnn_point(outs)
        outs = self.cplx2real(outs)
        return outs
    
class PhaseEncoderV4(nn.Module):
    def __init__(self, out_channels, in_channels=2,type_norm = "BatchNorm2d"):
        super(PhaseEncoderV4, self).__init__()
        self.conv1= nn.Conv2d(in_channels, out_channels*2, (1, 3))

        if type_norm == "BatchNorm2d" : 
            self.norm = nn.BatchNorm2d(out_channels*2)
        elif type_norm == "InstanceNorm2d" : 
            self.norm = nn.InstanceNorm2d(out_channels*2,track_running_stats=True)
        self.conv2= nn.Conv2d(out_channels*2, out_channels, (1, 3),padding=(0,1))

    def forward(self, X):
        """
        X : [B,C(re,im),F,T]
        """
        outs = self.conv1(X)
        outs = self.norm(outs)
        outs = self.conv2(outs)
        return outs
    
"""
Real Phase Encoder
"""
class PhaseEncoderV5(nn.Module):
    def __init__(self, out_channels, in_channels=2,type_norm = "BatchNorm2d",bias=True):
        super(PhaseEncoderV5, self).__init__()
        self.pe = nn.Conv2d(in_channels, out_channels, (1, 3),bias=bias)
        if type_norm == "BatchNorm2d" : 
            self.norm = nn.BatchNorm2d(out_channels)
        elif type_norm == "InstanceNorm2d" : 
            self.norm = nn.InstanceNorm2d(out_channels,track_running_stats=True)

    def forward(self, X):
        """
        X : [B,C(re,im),F,T]
        """
        outs = self.pe(X)
        outs = self.norm(outs)
        return outs
    
"""
Real Phase Encoder with Mag 
"""
class PhaseEncoderV6(nn.Module):
    def __init__(self, out_channels, type_norm = "BatchNorm2d",bias=True, **kwargs):
        super(PhaseEncoderV6, self).__init__()
        self.pe = nn.Conv2d(3, out_channels, (1, 3),bias=bias)
        if type_norm == "BatchNorm2d" : 
            self.norm = nn.BatchNorm2d(out_channels)
        elif type_norm == "InstanceNorm2d" : 
            self.norm = nn.InstanceNorm2d(out_channels,track_running_stats=True)

    def forward(self, X):
        """
        X : [B,C(re,im),F,T]
        """
        mag = torch.norm(X,dim=1,keepdim=True)
        X = torch.cat([X,mag],dim=1)
        outs = self.pe(X)
        outs = self.norm(outs)
        return outs

"""
Real Phase Encoder with Mag, but norm separtely
"""
class PhaseEncoderV7(nn.Module):
    def __init__(self, in_channels = 2, out_channels=4, type_norm = "BatchNorm2d",bias=True, **kwargs):
        super(PhaseEncoderV7, self).__init__()
        self.pe = nn.Conv2d(in_channels, out_channels-2, (1, 3),bias=bias)
        self.me = nn.Conv2d(1,2,(1,3),bias=bias)

        if type_norm == "BatchNorm2d" : 
            self.norm_cplx = nn.BatchNorm2d(out_channels-2)
            self.norm_mag = nn.BatchNorm2d(2)
        elif type_norm == "InstanceNorm2d" : 
            self.norm_cplx = nn.InstanceNorm2d(out_channels-2,track_running_stats=True)
            self.norm_mag = nn.InstanceNorm2d(2)

    def forward(self, X):
        """
        X : [B,C(re,im),F,T]
        """
        cplx = self.pe(X)
        mag = self.me(torch.norm(X,dim=1,keepdim=True))

        cplx = self.norm_cplx(cplx)
        mag = self.norm_mag(mag)

        outs = torch.cat([cplx,mag],dim=1)
        return outs
    
"""
Real Phase Encoder with LPS 
"""
class PhaseEncoderV8(nn.Module):
    def __init__(self, out_channels, type_norm = "BatchNorm2d",bias=True, **kwargs):
        super(PhaseEncoderV8, self).__init__()
        self.pe = nn.Conv2d(3, out_channels, (1, 3),bias=bias)
        if type_norm == "BatchNorm2d" : 
            self.norm = nn.BatchNorm2d(out_channels)
        elif type_norm == "InstanceNorm2d" : 
            self.norm = nn.InstanceNorm2d(out_channels,track_running_stats=True)

    def forward(self, X):
        """
        X : [B,C(re,im),F,T]
        """
        mag = 10*torch.log10(torch.norm(X,dim=1,keepdim=True)+1)
        X = torch.cat([X,mag],dim=1)
        outs = self.pe(X)
        outs = self.norm(outs)
        return outs
    

"""
Real Phase Encoder with Mag, but Layer_norm separtely after conv
"""
class PhaseEncoderV9(nn.Module):
    def __init__(self, in_channels = 2, out_channels=4, type_norm = "BatchNorm2d",bias=True, **kwargs):
        super(PhaseEncoderV9, self).__init__()
        self.pe = nn.Conv2d(in_channels, out_channels-2, (1, 3),bias=bias)
        self.me = nn.Conv2d(1,2,(1,3),bias=bias)

        self.norm_cplx = nn.LayerNorm([2,257])
        self.norm_mag = nn.LayerNorm([2,257])
        

    def forward(self, X):
        """
        X : [B,C(re,im),F,T]
        """
        cplx = self.pe(X)
        mag = self.me(torch.norm(X,dim=1,keepdim=True))

        # B,C,F,T -> B,T,C,F
        cplx = cplx.permute(0,3,1,2)
        mag = mag.permute(0,3,1,2)

        cplx = self.norm_cplx(cplx)
        mag = self.norm_mag(mag)

        # B,T,C,F ->> B,C,F,T
        cplx = cplx.permute(0,2,3,1)
        mag = mag.permute(0,2,3,1)

        outs = torch.cat([cplx,mag],dim=1)
        return outs
    
"""
Real Phase Encoder with Mag, but Layer_norm separtely before conv
"""
class PhaseEncoderV10(nn.Module):
    def __init__(self, in_channels = 2, out_channels=4, type_norm = "BatchNorm2d",bias=True, **kwargs):
        super(PhaseEncoderV10, self).__init__()
        self.pe = nn.Conv2d(in_channels, out_channels-2, (1, 3),bias=bias)
        self.me = nn.Conv2d(1,2,(1,3),bias=bias)

        self.norm_cplx = nn.LayerNorm([2,257])
        

    def forward(self, X):
        """
        X : [B,C(re,im),F,T]
        """

        # B,C,F,T -> B,T,C,F
        X = X.permute(0,3,1,2)
        X = self.norm_cplx(X)
        # B,T,C,F -> B,C,F,T
        X = X.permute(0,2,3,1)
        
        cplx = self.pe(X)
        mag = self.me(torch.norm(X,dim=1,keepdim=True))

        outs = torch.cat([cplx,mag],dim=1)
        return outs
    
"""
PEv7 + GRU
"""
class PhaseEncoderV11(nn.Module):
    def __init__(self, in_channels = 2, out_channels=4, type_norm = "BatchNorm2d",bias=True, hidden_size = 257,**kwargs):
        super(PhaseEncoderV11, self).__init__()
        self.pe = nn.Conv2d(in_channels, out_channels-2, (1, 3),bias=bias)
        self.me = nn.Conv2d(1,2,(1,3),bias=bias)

        self.rnn = nn.GRU(input_size=out_channels,hidden_size=out_channels,num_layers=1,batch_first=True,bidirectional=True)
        self.act = nn.ReLU()

        if type_norm == "BatchNorm2d" : 
            self.norm_cplx = nn.BatchNorm2d(out_channels-2)
            self.norm_mag = nn.BatchNorm2d(2)
        elif type_norm == "InstanceNorm2d" : 
            self.norm_cplx = nn.InstanceNorm2d(out_channels-2,track_running_stats=True)
            self.norm_mag = nn.InstanceNorm2d(2)

    def forward(self, X):
        """
        X : [B,C(re,im),F,T]
        """
        cplx = self.pe(X)
        mag = self.me(torch.norm(X,dim=1,keepdim=True))

        cplx = self.norm_cplx(cplx)
        mag = self.norm_mag(mag)

        outs = torch.cat([cplx,mag],dim=1)
        return outs
    
# For Ablation Study
class PhaseEncoderV12(nn.Module):
    def __init__(self, **kwargs):
        super(PhaseEncoderV12, self).__init__()
    def forward(self, X):
        """
        X : [B,C(re,im),F,T]
        """
        return X[:,:,:,1:-1]
    
"""
Real Phase Encoder with Mag, but norm separtely
"""
class PhaseEncoderV13(nn.Module):
    def __init__(self, in_channels = 2, out_channels=3, type_norm = "BatchNorm2d",bias=True, **kwargs):
        super(PhaseEncoderV13, self).__init__()
        self.pe = nn.Conv2d(in_channels, out_channels-1, (1, 3),bias=bias)
        self.me = nn.Conv2d(1, 1, (1, 3),bias=bias)

        if type_norm == "BatchNorm2d" : 
            self.norm_cplx = nn.BatchNorm2d(out_channels-1)
            self.norm_mag = nn.BatchNorm2d(1)
        elif type_norm == "InstanceNorm2d" : 
            self.norm_cplx = nn.InstanceNorm2d(out_channels-1,track_running_stats=True)
            self.norm_mag = nn.InstanceNorm2d(1)

    def forward(self, X):
        """
        X : [B,C(re,im),F,T]
        """
        cplx = self.pe(X)
        mag = self.me(torch.norm(X,dim=1,keepdim=True))

        cplx = self.norm_cplx(cplx)
        mag = self.norm_mag(mag)

        outs = torch.cat([cplx,mag],dim=1)
        return outs
 
    


#############################################


#define custom_atan2 to support onnx conversion
def custom_atan2(y, x):
    pi = torch.from_numpy(np.array([np.pi])).to(y.device, y.dtype)
    ans = torch.atan(y / (x + 1e-6))
    ans += ((y > 0) & (x < 0)) * pi
    ans -= ((y < 0) & (x < 0)) * pi
    ans *= 1 - ((y > 0) & (x == 0)) * 1.0
    ans += ((y > 0) & (x == 0)) * (pi / 2)
    ans *= 1 - ((y < 0) & (x == 0)) * 1.0
    ans += ((y < 0) & (x == 0)) * (-pi / 2)
    return ans


class MEA(nn.Module):
    # class of mask estimation and applying
    def __init__(self,in_channels=4, mag_f_dim=3):
        super(MEA, self).__init__()
        self.mag_mask = nn.Conv2d(
            in_channels, mag_f_dim, kernel_size=(3, 1), padding=(1, 0))
        self.real_mask = nn.Conv2d(in_channels, 1, kernel_size=(3, 1), padding=(1, 0))
        self.imag_mask = nn.Conv2d(in_channels, 1, kernel_size=(3, 1), padding=(1, 0))
        kernel = torch.eye(mag_f_dim)
        kernel = kernel.reshape(mag_f_dim, 1, mag_f_dim, 1)
        self.register_buffer('kernel', kernel)
        self.mag_f_dim = mag_f_dim
        self.eps = 1e-12
    
    def forward(self, x, z):
        # x = input stft, x.shape = (B,F,T,2)
        mag = torch.norm(x, dim=-1)
        pha = custom_atan2(x[..., 1], x[..., 0])

        # stage 1
        mag_mask = self.mag_mask(z)
        mag_pad = F.pad(
            mag[:, None], [0, 0, (self.mag_f_dim-1)//2, (self.mag_f_dim-1)//2])
        mag = F.conv2d(mag_pad, self.kernel)
        mag = mag * mag_mask.relu()
        mag = mag.sum(dim=1)

        # stage 2
        real_mask = self.real_mask(z).squeeze(1)
        imag_mask = self.imag_mask(z).squeeze(1)

        mag_mask = torch.sqrt(torch.clamp(real_mask**2+imag_mask**2, self.eps))
        pha_mask = custom_atan2(imag_mask+self.eps, real_mask+self.eps)
        real = mag * mag_mask.relu() * torch.cos(pha+pha_mask)
        imag = mag * mag_mask.relu() * torch.sin(pha+pha_mask)
        return torch.stack([real, imag], dim=-1)


class MEA2(nn.Module):
    # class of mask estimation and applying
    def __init__(self,in_channels=4, mag_f_dim=3):
        super(MEA2, self).__init__()
    
    def _get_mask_phase(self, mag, mag_n, chirality):
        cos_phase = (1 + mag**2 - mag_n**2) / (2 * mag + EPS)
        # prevent numerical errors by clipping between -1 and 1 (almost)
        cos_phase = torch.clip(cos_phase, -0.999, 0.999)
        phase = torch.acos(cos_phase)
        return chirality * phase

    def _get_mask_mags(self, z, zn, beta):
        """
        Equation for calculating the mask magnitude
        mask_mag = beta * sigmoid(z-zn) = beta * (1 + e^-((z)-(zn)))^(-1)
        follow the 3.1 in link https://arxiv.org/pdf/2102.03207.pdf
        """
        sigma_z = torch.sigmoid(z - zn)
        sigma_zn = 1 - sigma_z
        beta = torch.minimum(beta, 1.0 / (torch.abs(sigma_z - sigma_zn) + EPS))
        return beta * sigma_z, beta * sigma_zn

    def _get_mask(self, z):
        #  z.shape == (B,T,C,S)
        zd, zd_not, psi, chi = torch.unbind(z, dim=2)  # zd.shape == (B,T,S)
        beta = 1 + F.softplus(psi)
        chirality = F.gumbel_softmax(chi, tau=1, hard=True)
        # mask for direct output
        mag_d, mag_d_not = self._get_mask_mags(zd, zd_not, beta)
        phase_d = self._get_mask_phase(mag_d, mag_d_not, chirality)
        # mask for reverberant output -> just remove the noise
        # mag_r, mag_r_not = self._get_mask_mags(zn_not, zn, beta)
        # phase_r = self._get_mask_phase(mag_r, mag_r_not, chirality)
        return mag_d, phase_d #, mag_r, phase_r

    def _get_mask_mag(self, z):
        #  z.shape == (B,T,C,S)
        zd, zd_not, psi, chi = torch.unbind(z, dim=2)  # zd.shape == (B,T,S)
        beta = 1 + F.softplus(psi)
        # mask for direct output
        mag_d, mag_d_not = self._get_mask_mags(zd, zd_not, beta)
        # mask for reverberant output -> just remove the noise
        # mag_r, mag_r_not = self._get_mask_mags(zn_not, zn, beta)
        # phase_r = self._get_mask_phase(mag_r, mag_r_not, chirality)
        return mag_d #, mag_r

    def _mask2signal(self, in_mag, in_phase, mask_mag, mask_phase):
        # in_mag.shape == (B,T,S)
        mask_mag = mask_mag.permute(0,2,1)
        mask_phase = mask_phase.permute(0,2,1)
        total_mag = (in_mag + EPS) * mask_mag  # EPS to avoid multiplying by 0
        total_phase = in_phase + mask_phase
        real = total_mag * torch.cos(total_phase)
        imag = total_mag * torch.sin(total_phase)
        stft = real + 1j * imag  # stft.shape (B,T,S)
        # we don't use pytorch istft because it behaves differently than our iris_al
        # frames = torch.fft.irfft(stft).transpose(1, 2)  # frames.shape == (B,S,T)
        out_signal = torch.istft(stft, self.frame_size, self.hop_size, self.frame_size, torch.hann_window(self.frame_size).to(stft.device))
        return out_signal  # out_signal.shape == (B,N), N = num_samples
    
    def _mask2stft(self, in_mag, in_phase, mask_mag, mask_phase):
        # in_mag.shape == (B,T,S)
        mask_mag = mask_mag.permute(0,2,1)
        mask_phase = mask_phase.permute(0,2,1)
        total_mag = (in_mag + EPS) * mask_mag  # EPS to avoid multiplying by 0
        total_phase = in_phase + mask_phase
        real = total_mag * torch.cos(total_phase)
        imag = total_mag * torch.sin(total_phase)
        # we don't use pytorch istft because it behaves differently than our iris_al
        # frames = torch.fft.irfft(stft).transpose(1, 2)  # frames.shape == (B,S,T)
        return (real,imag)  # out_signal.shape == (B,N), N = num_samples
    def _mask2signal_mag(self, in_mag, in_phase, mask_mag):
        # in_mag.shape == (B,T,S)
        mask_mag = mask_mag.permute(0,2,1)
        total_mag = (in_mag + EPS) * mask_mag  # EPS to avoid multiplying by 0
        real = total_mag * torch.cos(in_phase)
        imag = total_mag * torch.sin(in_phase)
        stft = real + 1j * imag  # stft.shape (B,T,S)
        # we don't use pytorch istft because it behaves differently than our iris_al
        # frames = torch.fft.irfft(stft).transpose(1, 2)  # frames.shape == (B,S,T)
        out_signal = torch.istft(stft, self.frame_size, self.hop_size, self.frame_size, torch.hann_window(self.frame_size).to(stft.device))
        return out_signal  # out_signal.shape == (B,N), N = num_samples
    
    def forward(self, x, z):
        # x = input stft, x.shape = (B,F,T,2)
        #z = z.permute(B,T,C,S)
        z = z.permute(0,3,1,2)
        x = x[:,:,:,:2]
        mask_mag_d, mask_phase_d = self._get_mask(z)
        in_real, in_imag = torch.unbind(x, dim=-1)
        stft = in_real+1j*in_imag
        in_mag = torch.abs(stft)
        in_phase = torch.angle(stft)
        # direct_out = self._mask2signal_mag(in_mag, in_phase, mask_mag_d)
        real,imag = self._mask2stft(in_mag, in_phase, mask_mag_d, mask_phase_d)
        return torch.stack([real, imag], dim=-1)
################ Data Processor ################

"""
Power Law Compression
Shetu, Shrishti Saha, et al. "Ultra Low Complexity Deep Learning Based Noise Suppression." arXiv preprint arXiv:2312.08132 (2023).
"""
# NOTE : torch.sign is not differentiable
# https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
class GradSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = torch.sign(x)
        ctx.save_for_backward(x)
        #return torch.tanh(x / epsilon)    
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output.clone()

class PowerLawCompression(nn.Module):
    def __init__(self, alpha=0.3, **kwargs):
        super(PowerLawCompression, self).__init__()
        self.alpha = alpha

    def forward(self, X):
        """
        X.shape == (B,F,T,2)
        """
        # Power Law Compression
        # NOTE : PLC on each real and image
        X[:,:,:,:] = GradSign.apply(X[:,:,:,:]) * torch.pow(torch.abs(X[:,:,:,:]), self.alpha)
        return X

class PowerLawDecompression(nn.Module):
    def __init__(self, alpha=0.3, **kwargs):
        super(PowerLawDecompression, self).__init__()
        self.alpha = alpha
        #self.eps = 1e-7
        self.eps = 0.0

    def forward(self, X):
        """
        X.shape == (B,F,T,2)
        """
        X[:,:,:,:] = GradSign.apply(X[:,:,:,:]) * torch.pow(torch.abs(X[:,:,:,:] + self.eps), 1/self.alpha)

        return X

class PerceptualContrastStretching(nn.Module):
    def __init__(self, encoding, **kwargs):
        super(PerceptualContrastStretching, self).__init__()
        self.encoding = encoding
        self.PCS = torch.ones(257)
        self.PCS[0:3] = 1
        self.PCS[3:6] = 1.070175439
        self.PCS[6:9] = 1.182456140
        self.PCS[9:12] = 1.287719298
        self.PCS[12:138] = 1.4
        self.PCS[138:166] = 1.322807018
        self.PCS[166:200] = 1.238596491
        self.PCS[200:241] = 1.161403509
        self.PCS[241:256] = 1.077192982

    def forward(self, X):
        """
        X.shape == (B,F,T,2)
        """
        X = X[...,0] + X[...,1]*1j
        mag = torch.abs(X)
        phase = torch.angle(X)
        if self.encoding:
            mag = torch.log1p(mag)
        else :
            mag = torch.expm1(mag)

        # Apply PCS elementwise
        PCS_expanded = self.PCS.unsqueeze(0).unsqueeze(-1).expand_as(mag)
        if self.encoding:
            mag = mag * PCS_expanded.to(mag.device)
        else :
            mag = mag / PCS_expanded.to(mag.device)
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        
        return torch.stack([real, imag], dim=-1)

"""
Channelwise Feature Orientation

Shetu, Shrishti Saha, et al. "Ultra Low Complexity Deep Learning Based Noise Suppression." arXiv preprint arXiv:2312.08132 (2023).
<-
Liu, Haohe, et al. "Channel-wise subband input for better voice and accompaniment separation on high resolution music." arXiv preprint arXiv:2008.05216 (2020).
"""
# ChannelwiseReorientation
class CR(nn.Module) : 
    def __init__(self, n_band, overlap=1/3, **kwargs):
        super(CR, self).__init__()
        self.n_band = n_band
        self.overlap = overlap
        """
        if type_window == "None" :
            self.window = torch.tensor(1.0)
        elif type_window == "Rectengular" : 
            self.window = torch.kaiser_window(window_length ,beta = 0.0)
        elif type_window == "Hanning":
            self.window = torch.hann_window(window_length)
        else :
            raise NotImplementedError
        """

    def forward(self,x):
        idx = 0

        B,C,F,T = x.shape
        n_freq = x.shape[2]
        sz_band = n_freq/(self.n_band*(1-self.overlap))
        sz_band = int(np.ceil(sz_band))
        y = torch.zeros(B,self.n_band*C,sz_band,T).to(x.device)
        
        for i in range(self.n_band):
            if idx+sz_band > F :
                sz_band = F - idx
            y[:,i*C:(i+1)*C,:sz_band,:] = x[:,:,idx:idx+sz_band,:]
            n_idx = idx + int(sz_band*(1-self.overlap))
            idx = n_idx
        return y

class iCR(nn.Module):
    def __init__(self,n_freq,out_channels=1,overlap=1/3):
        super(iCR, self).__init__()
        self.n_freq = n_freq
        self.out_channels = out_channels
        self.overlap = overlap

    def forward(self,x):
        # x : [B, n_band*C, sz_band, T]
        B,C,F,T = x.shape
        # in_channels must be dividable with out_channels
        
        n_band = int(C/self.out_channels)
        n_not_over = int((1-self.overlap)*F)
        
        y = torch.zeros(B,self.out_channels,self.n_freq,T).to(x.device)
        
        idx = 0
        for i in range(int(n_band)):
            n_idx = idx + int(F*(1-self.overlap)) 
            if idx + F > self.n_freq :
                n_idx = self.n_freq
            
            if n_idx != self.n_freq : 
                y[:,:,idx:n_idx,:] = x[:,i*self.out_channels:(i+1)*self.out_channels, :n_not_over,:]
                y[:,:,n_idx : idx + F] += x[:,i*self.out_channels:(i+1)*self.out_channels,n_not_over:,:]/2
            # last band
            else :
                toread = n_idx - idx
                y[:,:,idx:n_idx,:] = x[:,i*self.out_channels:(i+1)*self.out_channels, :toread,:]
            
            idx = n_idx
        return y
