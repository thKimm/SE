import torch
import torch.nn as nn
import librosa as rs
import numpy as np

EPS = 1e-9
class CosSDRLossSegment(nn.Module):
    """
    It's a cosine similarity between predicted and clean signal
        loss = - <y_true, y_pred> / (||y_true|| * ||y_pred||)
    This loss function is always bounded between -1 and 1
    Ref: https://openreview.net/pdf?id=SkeRTsAcYm
    Hyeong-Seok Choi et al., Phase-aware Speech Enhancement with Deep Complex U-Net,
    """
    def __init__(self, reduction=torch.mean):
        super(CosSDRLossSegment, self).__init__()
        self.reduction = reduction

    def forward(self, output, target, out_dict=True):
        num = torch.sum(target * output, dim=-1)
        den = torch.norm(target, dim=-1) * torch.norm(output, dim=-1)
        loss_per_element = -num / (den + EPS)
        loss = self.reduction(loss_per_element)
        return {"CosSDRLossSegment": loss} if out_dict else loss

class CosSDRLoss(nn.Module):
    def __init__(self, reduction=torch.mean):
        super(CosSDRLoss, self).__init__()
        self.segment_loss = CosSDRLossSegment(nn.Identity())
        self.reduction = reduction

    def forward(self, output, target, chunk_size=1024, out_dict=True):
        output = output[:,:output.shape[1] - output.shape[1]%chunk_size]
        target = target[:,:target.shape[1] - target.shape[1]%chunk_size]
        out_chunks = torch.reshape(output, [output.shape[0], -1, chunk_size])
        trg_chunks = torch.reshape(target, [target.shape[0], -1, chunk_size])
        loss_per_element = torch.mean(
            self.segment_loss(out_chunks, trg_chunks, False), dim=-1
        )
        loss = self.reduction(loss_per_element)
        return {"CosSDRLoss": loss} if out_dict else loss

# Weighted SDR
class CoswSDRLoss(nn.Module):
    def __init__(self, ratio=0.5,reduction=torch.mean):
        super(CoswSDRLoss, self).__init__()
        self.segment_loss = CosSDRLossSegment(nn.Identity())
        self.reduction = reduction
        self.ratio = ratio

    def forward(self, output, target, noisy, chunk_size=1024, out_dict=True):
        out_chunks = torch.reshape(output, [output.shape[0], -1, chunk_size])
        trg_chunks = torch.reshape(target, [target.shape[0], -1, chunk_size])

        ## TODO : WIP
        #noi_chunks = torch.reshape(noisy, [noise.shape[0], -1, chunk_size])
        #noi_est_chunk = torch.reshape(noisy, [noisy.shape[0], -1, chunk_size])
        #loss_per_element_noisy = torch.mean(
        #    self.segment_loss(noisy_chunks, trg_chunks, False), dim=-1
        #)

        loss_per_element = torch.mean(
            self.segment_loss(out_chunks, trg_chunks, False), dim=-1
        )

        loss = self.reduction(loss_per_element)
        return {"CosSDRLoss": loss} if out_dict else loss

class MultiscaleCosSDRLoss(nn.Module):
    def __init__(self, chunk_sizes, reduction=torch.mean):
        super(MultiscaleCosSDRLoss, self).__init__()
        self.chunk_sizes = chunk_sizes
        self.loss = CosSDRLoss(nn.Identity())
        self.reduction = reduction

    def forward(self, output, target, out_dict=True):
        loss_per_scale = [
            self.loss(output, target, cs, False) for cs in self.chunk_sizes
        ]
        loss_per_element = torch.mean(torch.stack(loss_per_scale), dim=0)
        loss = self.reduction(loss_per_element)
        return {"MultiscaleCosSDRLoss": loss} if out_dict else loss


class SpectrogramLoss(nn.Module):
    def __init__(self, reduction=torch.mean):
        super(SpectrogramLoss, self).__init__()
        self.gamma = 0.3
        self.reduction = reduction

    def forward(self, output, target, chunk_size=1024, out_dict=True):
        # stft.shape == (batch_size, fft_size, num_chunks)
        stft_output = torch.stft(output, chunk_size, hop_length=chunk_size//4, return_complex=True, center=False)
        stft_target = torch.stft(target, chunk_size, hop_length=chunk_size//4, return_complex=True, center=False)

        # clip is needed to avoid nan gradients in the backprop
        mag_output = torch.clip(torch.abs(stft_output), min=EPS)
        mag_target = torch.clip(torch.abs(stft_target), min=EPS)
        distance = mag_target**self.gamma - mag_output**self.gamma

        # average out
        loss_per_chunk = torch.mean(distance**2, dim=1)
        loss_per_element = torch.mean(loss_per_chunk, dim=-1)
        loss = self.reduction(loss_per_element)
        return {"SpectrogramLoss": loss} if out_dict else loss

class MultiscaleSpectrogramLoss(nn.Module):
    def __init__(self, chunk_sizes, reduction=torch.mean):
        super(MultiscaleSpectrogramLoss, self).__init__()
        self.chunk_sizes = chunk_sizes
        self.loss = SpectrogramLoss(nn.Identity())
        self.reduction = reduction

    def forward(self, output, target, out_dict=True):
        loss_per_scale = [
            self.loss(output, target, cs, False) for cs in self.chunk_sizes
        ]
        loss_per_element = torch.mean(torch.stack(loss_per_scale), dim=0)
        loss = self.reduction(loss_per_element)
        return {"MultiscaleSpectrogramLoss": loss} if out_dict else loss
    
class WMMSELoss(nn.Module):
    def __init__(self, n_fft = 512, alpha=0.9,sr=16000, n_mels = 40,eps = 1e-7):
        super(WMMSELoss,self).__init__()

        self.mel_basis = rs.filters.mel(sr=sr, n_fft=n_fft,n_mels=n_mels)
        self.mel_basis = torch.from_numpy(self.mel_basis)

        self.eps = eps
        self.n_fft = n_fft
        self.alpha = alpha

    def forward(self,output,target):
        if output.device != self.mel_basis.device:
            self.mel_basis = self.mel_basis.to(output.device)

        output = torch.stft(output, self.n_fft, return_complex=True, center=False)
        target = torch.stft(target, self.n_fft, return_complex=True, center=False)

        # add eps due to 'MmBackward nan' error in gradient
        s_hat_mag = torch.abs(output) + self.eps
        s_mag = torch.abs(target) + self.eps

        # scale
        s_mag= torch.log10(1+s_mag)
        s_hat_mag= torch.log10(1+s_hat_mag)

        # mel
        s = torch.matmul(self.mel_basis,s_mag)
        s_hat = torch.matmul(self.mel_basis,s_hat_mag)

        # Batch Norm
        s_mag = s_mag/torch.max(s_mag)
        s_hat_mag = s_hat_mag/torch.max(s_hat_mag)

        d = s - s_hat

        WMMSE = torch.mean(self.alpha *(d + d.abs())/2 + (1-self.alpha) * (d-d.abs()).abs()/2)

        return  WMMSE

    
class MultiscaleWMMSELoss(nn.Module) : 
    def __init__(self, chunk_sizes, alpha = 0.9, reduction = torch.mean):
        super(MultiscaleWMMSELoss,self).__init__()
        self.chunk_sizes = chunk_sizes
        self.losses = [WMMSELoss(n_fft=x,alpha = alpha) for x in chunk_sizes ]
        self.reduction = reduction

    def forward(self, output, target, out_dict=True):
        loss_per_scale = [l(output, target) for l in self.losses]
        loss_per_element = torch.mean(torch.stack(loss_per_scale), dim=0)
        loss = self.reduction(loss_per_element)
        return {"MultiscaleWMMMSELoss": loss} if out_dict else loss

class TrunetLoss(nn.Module):
    def __init__(self, frame_size_sdr, frame_size_spec):
        super(TrunetLoss, self).__init__()
        self.sdr_loss = MultiscaleCosSDRLoss(frame_size_sdr)
        self.spc_loss = MultiscaleSpectrogramLoss(frame_size_spec)

    def forward(self, outputs, targets, out_dict=False):
        # shape: (batch_size, direct_or_reverberant, num_samples)
        yd = outputs
        td = targets
        # d=direct, r=reverberant; reverb = reverberant - direct
        # fmt: off
        losses = {
            "MultiscaleSpectrogramLoss_Direct":      self.spc_loss(yd, td, out_dict=False),
            "MultiscaleCosSDRWavLoss_Direct":        self.sdr_loss(yd, td, out_dict=False),
        }
        # fmt: on
        return (
            losses if out_dict else torch.sum(torch.stack([v for v in losses.values()]))
        )
    
class HybridLoss(nn.Module):
    def __init__(self,frame_size_sdr, frame_size_spec,alpha=0.9):
        super(HybridLoss, self).__init__()
        self.sdr_loss = MultiscaleCosSDRLoss(frame_size_sdr)
        self.spc_loss = MultiscaleWMMSELoss(frame_size_spec,alpha=alpha)

    def forward(self, outputs, targets, out_dict=False):
        # shape: (batch_size, direct_or_reverberant, num_samples)
        yd = outputs
        td = targets
        # d=direct, r=reverberant; reverb = reverberant - direct
        # fmt: off
        losses = {
            "MultiscaleWMMSELoss_Direct": self.spc_loss(yd, td, out_dict=False),
            "MultiscaleCosSDRWavLoss_Direct": self.sdr_loss(yd, td, out_dict=False),
        }
        # fmt: on
        return (
            losses if out_dict else torch.sum(torch.stack([v for v in losses.values()]))
        )
def anti_wrapping_function(x):

    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

class PhaseLoss(nn.Module):
    def __init__(self, n_fft = 512, sr= 16000, alpha=0.9):
        super(PhaseLoss, self).__init__()
        self.n_fft = n_fft
        self.sr = sr
        self.alpha = alpha
        
    def forward(self, yd, td):
        Yd = torch.stft(yd, n_fft=self.n_fft, return_complex=True)
        Yd_phase = torch.angle(Yd)
        phase_r = torch.unsqueeze(Yd_phase, 1)
        Td = torch.stft(td, n_fft=self.n_fft, return_complex=True)
        Td_phase = torch.angle(Td)
        phase_g = torch.unsqueeze(Td_phase, 1)
        
        dim_freq = self.n_fft // 2 + 1
        dim_time = phase_r.size(-1)

        gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) - torch.triu(torch.ones(dim_freq, dim_freq), diagonal=2) - torch.eye(dim_freq)).to(phase_g.device)
        gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
        gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)

        iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) - torch.triu(torch.ones(dim_time, dim_time), diagonal=2) - torch.eye(dim_time)).to(phase_g.device)
        iaf_r = torch.matmul(phase_r, iaf_matrix)
        iaf_g = torch.matmul(phase_g, iaf_matrix)
        
        ip_loss = torch.mean(anti_wrapping_function(phase_r-phase_g))
        gd_loss = torch.mean(anti_wrapping_function(gd_r-gd_g))
        iaf_loss = torch.mean(anti_wrapping_function(iaf_r-iaf_g))

        return ip_loss, gd_loss, iaf_loss

class MultiscalePhaseLoss(nn.Module) : 
    def __init__(self, chunk_sizes, alpha = 0.9, reduction = torch.mean):
        super(MultiscalePhaseLoss,self).__init__()
        self.chunk_sizes = chunk_sizes
        self.losses = [PhaseLoss(n_fft=x,alpha = alpha) for x in chunk_sizes ]
        self.reduction = reduction

    def forward(self, output, target, out_dict=True):
        loss_per_scale = [l(output, target) for l in self.losses]
        loss_per_element = torch.mean(torch.stack(loss_per_scale), dim=0)
        loss = self.reduction(loss_per_element)
        return {"MultiscalePhaseLoss": loss} if out_dict else loss


class HybridPhaseLoss(nn.Module):
    def __init__(self,frame_size_sdr, frame_size_spec,alpha=0.9):
        super(HybridLoss, self).__init__()
        self.sdr_loss = MultiscaleCosSDRLoss(frame_size_sdr)
        self.spc_loss = MultiscaleWMMSELoss(frame_size_spec,alpha=alpha)
        self.phase_loss = phase_losses()
        
    def forward(self, outputs, targets, out_dict=False):
        # shape: (batch_size, direct_or_reverberant, num_samples)
        yd = outputs
        td = targets
        # d=direct, r=reverberant; reverb = reverberant - direct
        # fmt: off
        Yd = torch.stft(yd, n_fft=512, return_complex=True)
        Yd_phase = torch.angle(Yd)
        Yd_phase = torch.unsqueeze(Yd_phase, 1)
        Td = torch.stft(td, n_fft=512, return_complex=True)
        Td_phase = torch.angle(Td)
        Td_phase = torch.unsqueeze(Td_phase, 1)
        
        losses = {
            "MultiscaleWMMSELoss_Direct": self.spc_loss(yd, td, out_dict=False),
            "MultiscaleCosSDRWavLoss_Direct": self.sdr_loss(yd, td, out_dict=False),
            "MultiscalePhaseLoss_Direct": 0.3*self.phase_loss(Yd_phase, Td_phase, out_dict=False),
        }
        # fmt: on
        return (
            losses if out_dict else torch.sum(torch.stack([v for v in losses.values()]))
        )
        
class TrunetPhaseLoss(nn.Module):
    def __init__(self, frame_size_sdr, frame_size_spec):
        super(TrunetLoss, self).__init__()
        self.sdr_loss = MultiscaleCosSDRLoss(frame_size_sdr)
        self.spc_loss = MultiscaleSpectrogramLoss(frame_size_spec)
        self.phase_loss = MultiscalePhaseLoss(frame_size_spec)

    def forward(self, outputs, targets, out_dict=False):
        # shape: (batch_size, direct_or_reverberant, num_samples)
        yd = outputs
        td = targets
        # d=direct, r=reverberant; reverb = reverberant - direct
        # fmt: off
        losses = {
            "MultiscaleSpectrogramLoss_Direct":      self.spc_loss(yd, td, out_dict=False),
            "MultiscaleCosSDRWavLoss_Direct":        self.sdr_loss(yd, td, out_dict=False),
            "MultiscaleCosSDRWavLoss_Direct":        0.3*self.phase_loss(yd, td, out_dict=False),
        }
        # fmt: on
        return (
            losses if out_dict else torch.sum(torch.stack([v for v in losses.values()]))
        )