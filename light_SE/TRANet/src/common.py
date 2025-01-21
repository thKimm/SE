import os
import torch
import torch.nn as nn

# from mpSE.TRUNet_ori import TRUNet
# from mpSE.TRUNet_RIRI import TRUNet
# from mpSE.TRUNet_RIMRIM import TRUNet
from mpSE.TRUNet_MEA_BF import TRUNet
# from mpSE.TRUNet_MEA_BF_ANC import TRUNet
# from mpSE.TRUNet_MEA_BF_PCS import TRUNet
# from mpSE.TRUNet_MEA_BF_bottle import TRUNet
# from mpSE.TRUNet_MEA_BF_reverse import TRUNet
# from mpSE.TRUNet_MEA_BF_mapping import TRUNet
# from mpSE.TRUNet_MEA_BF_mapping_reverse import TRUNet
# from mpSE.TRUNet_MEA import TRUNet
from MappingNet.MappingNet_porting import MappingNet
# from MappingNet.MappingNet import MappingNet
# from MappingNet.MappingNet_ANC import MappingNet
from SpatialNet.SpatialNet import SpatialNet
import librosa as rs
from utils.metric import run_metric

def get_model(hp,device="cuda:0"):
    if hp.model.type == "UNet": 
        model = UNet().to(device)
    elif hp.model.type == "ResUNetOnFreq" :
        model = ResUNetOnFreq(
            c_in=c_in,
            c_out=c_out,
            n_fft=hp.audio.n_fft,
            n_block=5,
            norm = hp.model.norm,
            Softplus_thr=hp.model.Softplus_thr,
            activation = hp.model.activation,
            dropout = hp.model.dropout
            ).to(device)
    elif hp.model.type == "ResUNetOnFreq2" :
        model = ResUNetOnFreq2(
            c_in=c_in,
            c_out=c_out,
            n_fft=hp.audio.n_fft,
            n_block=5,
            norm = hp.model.norm,
            Softplus_thr=hp.model.Softplus_thr,
            activation = hp.model.activation,
            dropout = hp.model.dropout,
            multi_scale=hp.model.multi_scale
            ).to(device)
    elif hp.model.type == "FullSubNetPlus" : 
        model = FullSubNet_Plus(num_freqs = hp.model.n_freq).to(device)
    elif hp.model.type == "TRUMEA": 
        # from mpSE.TRUNet import TRUNet
        model = TRUNet(
            hp.audio.n_fft,
            hp.audio.n_hop,
            architecture=hp.model.architecture,
            kernel_type = hp.model.kernel_type,
            skipGRU= hp.model.skipGRU,
            phase_encoder=hp.model.phase_encoder,
            T_FGRU=hp.model.T_FGRU,
            type_TBlock=hp.model.type_TBlock,
            type_FBlock=hp.model.type_FBlock,
            type_CBlock=hp.model.type_CBlock,
            type_skip = hp.model.type_Skip,
            PLC = hp.model.PLC,
            PLC_alpha=hp.model.PLC_alpha,
            CR_use=hp.model.CR.use,
            CR_n_band=hp.model.CR.n_band,
            CR_overlap=hp.model.CR.overlap,

         ).to(device)
    elif hp.model.type == "TRUMEA_HnA" : 
        # from mpSE.TRUNet import TRUNet
        model = TRUNet(
            hp.audio.n_fft,
            hp.audio.n_hop,
            architecture=hp.model.architecture,
            kernel_type = hp.model.kernel_type,
            skipGRU= hp.model.skipGRU,
            phase_encoder=hp.model.phase_encoder,
            T_FGRU=hp.model.T_FGRU,
            type_TBlock=hp.model.type_TBlock,
            type_FBlock=hp.model.type_FBlock,
            type_CBlock=hp.model.type_CBlock,
            type_skip = hp.model.type_Skip,
            PLC = hp.model.PLC,
            PLC_alpha=hp.model.PLC_alpha,
            CR_use=hp.model.CR.use,
            CR_n_band=hp.model.CR.n_band,
            CR_overlap=hp.model.CR.overlap,

         ).to(device)
    elif hp.model.type == "MTFAA" :
        model = MTFAA_helper(
            n_fft = hp.model.n_fft,
            n_hop = hp.model.n_hop,
            n_erb = hp.model.n_erb,
            Co = hp.model.Co,
            type_encoder = hp.model.type_encoder,
            type_ASA = hp.model.type_ASA
        ).to(device)
    elif hp.model.type =="None":
        model = nn.Identity()
    elif hp.model.type == "MappingNet" :
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
    elif hp.model.type == "SpatialNet" :
        model = SpatialNet(
            **hp.model.init_args
        ).to(device)
    else : 
        raise Exception("ERROR::Unknown model type : {}".format(hp.model.type))

    return model

def run(
    hp,
    data,
    model,
    criterion=None,
    ret_output=False,
    device="cuda:0"
    ): 
    if hp.model.type == "FullSubNetPlus":
        data["input"][0]=data["input"][0].to(device)
        data["input"][1]=data["input"][1].to(device)
        data["input"][2]=data["input"][2].to(device)
        feature = data["input"]
        mask = model(feature[0],feature[1],feature[2])
        estim= model.output(mask,feature[1],feature[2])
    elif hp.model.type == "ResUNetOnFreq" : 
        feature = data["noisy"].to(device)
        mask = model(feature)
        estim= model.output(mask,feature)
    elif hp.model.type == "ResUNetOnFreq2" : 
        feature = data["noisy"].to(device)
        mask = model(feature)
        estim= model.output(mask,feature)
    elif hp.model.type == "ResUNetOnFreq3" : 
        feature = data["noisy"].to(device)
        mask = model(feature)
        estim= model.output(mask,feature)
    elif hp.model.type =="TRUMEA" : 
        feature = data["noisy"].to(device)
        estim = model(feature)
    elif hp.model.type =="TRUMEA_HnA" : 
        feature = data["noisy"].to(device)
        feature_AEC = data["AEC"].to(device)
        estim = model(feature, feature_AEC)
    elif hp.model.type =="MappingNet" : 
        feature = data["noisy"].to(device)
        feature_AEC = data["AEC"].to(device)
        estim = model(feature, feature_AEC)
        # estim = model(feature)
    elif hp.model.type =="MTFAA" : 
        feature = data["noisy"].to(device)
        estim = model(feature)
    elif hp.model.type =="SpatialNet" : 
        feature = data["noisy"].to(device)
        estim = model(feature, False)
    else : 
        raise Exception("ERROR::Unnkwon Model : {}".format(hp.model.type))

    if criterion is None : 
        return estim

    if hp.loss.type =="wSDRLoss" :
        clean= data["clean"].to(device)
        noisy= data["noisy"].to(device)
        """
        if not hp.model.mag_only : 
            estim =  estim
        else :
            spec= torch.stft(
                noisy,
                n_fft = hp.audio.n_fft,
                hop_length=hp.audio.n_hop,
                window=torch.hann_window(hp.audio.n_fft).to(device)
                )
            spec_real = spec[...,0]
            spec_imag = spec[...,1]
            phase = torch.atan2(spec_real,spec_imag)
            estim_spec = estim[:,0,:,:] * torch.exp(phase*1j)
            estim_wav =  torch.istft(
                estim_spec,
                n_fft = hp.audio.n_fft,
                hop_length=hp.audio.n_hop,
                window=torch.hann_window(hp.audio.n_fft).to(device)
                )
        """
        loss = criterion(estim,noisy.to(device),clean.to(device), alpha=hp.loss.wSDRLoss.alpha).to(device)
    elif hp.loss.type == "mwMSELoss" : 
        loss = criterion(estim,data["clean_spec"].to(device), alpha=hp.loss.mwMSELoss.alpha,sr=hp.data.sr,n_fft=hp.data.n_fft,device=device).to(device)
    elif hp.loss.type== "MSELoss":
        loss = criterion(estim,data["clean"].to(device))
    elif hp.loss.type == "mwMSELoss+wSDRLoss" : 
        estim_wav = torch.istft(estim[:,0,:,:],n_fft = hp.data.n_fft,hop_length=hp.data.n_hop,window=torch.hann_window(hp.data.n_fft).to(device))
        loss = criterion[0](estim,data["clean_spec"].to(device), alpha=hp.loss.mwMSELoss.alpha,sr=hp.data.sr,n_fft=hp.data.n_fft,device=device).to(device) + criterion[1](estim_wav,data["noisy_wav"].to(device),data["clean_wav"].to(device), alpha=hp.loss.wSDRLoss.alpha).to(device)
    else :
        if hp.data.use_RIR:
            loss = criterion(estim,data["clean"][:,0,:].to(device))
        else:
            loss = criterion(estim,data["clean"].to(device))

    if loss.isinf().any() : 
        print("Warning::There is inf in loss, nan_to_num(1e-7)")
        loss = torch.tensor(0.0).to(loss.device)
        loss.requires_grad_()

    if loss.isnan().any() : 
        print("Warning::There is nan in loss, nan_to_num(1e-7)")
        loss = torch.tensor(0.0).to(loss.device)
        loss.requires_grad_()

    if ret_output :
        return estim, loss
    else : 
        return loss
from tqdm import tqdm
def evaluate(hp, model,list_data,device="cuda:0"):
    #### EVAL ####
    model.eval()
    with torch.no_grad():
        ## Metric
        metric = {}
        for m in hp.log.eval : 
            metric["{}".format(m)] = 0.0
        pbar = tqdm(total=len(list_data), unit='samples', bar_format='{l_bar}{bar:25}{r_bar}{bar:-10b}', colour="BLUE", dynamic_ncols=True)
        for pair_data in list_data : 
            path_noisy = pair_data[0]
            path_clean = pair_data[1]
            path_AEC = pair_data[2]
            noisy = rs.load(path_noisy,sr=hp.data.sr,mono=False)[0]
            noisy = torch.unsqueeze(torch.from_numpy(noisy),0).to(device)
            aec = rs.load(path_AEC,sr=hp.data.sr,mono=False)[0]
            aec = torch.unsqueeze(torch.from_numpy(aec),0).to(device)
            estim = model(noisy,aec).cpu().detach().numpy()[0]
            clean = rs.load(path_clean,sr=hp.data.sr,mono=False)[0]
            clean = clean[0,:]
            if len(clean) > len(estim) :
                clean = clean[:len(estim)]
            else :
                estim = estim[:len(clean)]
            for m in hp.log.eval : 
                val= run_metric(estim,clean,m) 
                metric["{}".format(m)] += val
            pbar.update(1)
        for m in hp.log.eval : 
            key = "{}".format(m)
            metric[key] /= len(list_data)
    return metric


###### from audio_zen.acoustics.feature
def mag_phase(complex_tensor):
    return torch.abs(complex_tensor), torch.angle(complex_tensor)

def MRI(X):
    mag, _ = mag_phase(X)
    mag = mag.unsqueeze(0)
    real = (X.real).unsqueeze(0)
    imag = (X.imag).unsqueeze(0)

    return mag.float(),real.float(),imag.float()

