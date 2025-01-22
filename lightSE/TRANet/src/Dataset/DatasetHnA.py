import os
from os.path import join
from glob import glob
import torch
import torch.nn.functional as F
import librosa as rs
import numpy as np
import random
# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
from scipy import signal
warnings.filterwarnings('ignore')
import utils.speex.speex_module as speex


class DatasetHnA(torch.utils.data.Dataset):
    def __init__(self,hp,is_train=True):
        self.hp = hp
        self.is_train = is_train

        if is_train : 
            self.list_clean = glob(join(hp.data.clean,"**","*.wav"),recursive=True)
            self.list_ego = glob(join(hp.data.ego,"**","*.wav"),recursive=True)
            self.list_noise = glob(join(hp.data.noise,"**","*.wav"),recursive=True)
            self.list_RIR   = glob(join(hp.data.RIR,"**","*.wav"),recursive=True)
        else :
            # self.list_noisy = glob(join(hp.data.dev.root,"*","noisy","*.wav"),recursive=True)
            # self.list_RIR   = None
            # self.list_clean = None

            # self.eval={}
            # self.eval["with_reverb"]={}
            # self.eval["no_reverb"]={}

            # self.eval["with_reverb"] = glob(join(hp.data.dev.root,"with_reverb","noisy","*.wav"),recursive=True)

            # self.eval["no_reverb"] = glob(join(hp.data.dev.root,"no_reverb","noisy","*.wav"),recursive=True)
            self.list_clean = glob(join(hp.data.dev.clean,"**","*.wav"),recursive=True)
            self.list_noisy = glob(join(hp.data.dev.noisy,"**","*.wav"),recursive=True)
        self.range_noiseSNR = hp.data.noiseSNR
        self.range_egoSNR = hp.data.egoSNR
        self.target_dB_FS = -25
        self.target_dB_FS_floating_value = 10

        self.len_data = hp.data.len_data
        self.n_item = hp.data.n_item

        self.sr = hp.data.sr
        self.AEC_frame_size = 256
        self.AEC_filter_length = 256 * 16
        self.speex = speex
        self.echo_state = self.speex.init_echo_state(self.AEC_frame_size, self.AEC_filter_length, self.sr)
        self.preprocess_state = self.speex.init_preprocess_state(self.AEC_frame_size, self.sr, self.echo_state)
        if is_train : 
            print("DatasetHnA[train:{}] | len : {} | clean {} | ego noise : {} | noise : {} | RIR : {}".format(is_train,len(self.list_clean),len(self.list_clean),len(self.list_ego),len(self.list_noise),len(self.list_RIR)))
        else :
            print("DatasetHnA[train:{}] | len : {} | clean {} | noisy : {}".format(is_train,len(self.list_clean),len(self.list_clean),len(self.list_noisy)))

    '''
    AEC processing
    '''
    def float_to_short(self, x):
        x = x * 32768.0
        x[x < -32767.5] = -32768
        x[x > 32766.5] = 32767
        x = np.floor(0.5 + x)
        return x

    def main_loop(self, u, d):
        self.echo_state = speex.init_echo_state(self.AEC_frame_size, self.AEC_filter_length, self.sr)
        self.preprocess_state = speex.init_preprocess_state(self.AEC_frame_size, self.sr, self.echo_state)
        assert u.shape == d.shape
        # u = float_to_short(u)
        # d = float_to_short(d)

        y = []
        Ly = np.zeros_like(u, dtype=np.int16)
        end_point = len(u)
        for n in range(0, end_point, self.AEC_frame_size):
            # the break operation not understand.
            # only for signal channel AEC
            if n+self.AEC_frame_size > end_point:
                break
            u_frame = u[n:n+self.AEC_frame_size]
            d_frame = d[n:n+self.AEC_frame_size]
            y_frame = np.zeros_like(u_frame)
            self.speex.echo_cancellation(self.echo_state, d_frame, u_frame, y_frame)
            Ly[n:n+self.AEC_frame_size] = y_frame
            self.speex.preprocess_run(self.preprocess_state, y_frame)
            y.append(y_frame)
        u_frame = np.concatenate([u[n:], np.zeros(self.AEC_frame_size-len(u[n:]))])
        d_frame = np.concatenate([d[n:], np.zeros(self.AEC_frame_size-len(d[n:]))])
        self.speex.echo_cancellation(self.echo_state, d_frame, u_frame, y_frame)
        self.speex.preprocess_run(self.preprocess_state, y_frame)
        y.append(y_frame)
        y = np.concatenate(y[1:])
        self.speex.destroy_echo_state(self.echo_state)
        self.speex.destroy_preprocess_state(self.preprocess_state)
        return y/32768.0, Ly/32768.0
    
    def match_length(self,wav,idx_start=None) : 
        if max(wav.shape) > self.len_data : 
            left = max(wav.shape) - self.len_data
            if idx_start is None :
                idx_start = np.random.randint(left)
            if len(wav.shape) > 1:
                wav = wav[:,idx_start:idx_start+self.len_data]
            else:
                wav = wav[idx_start:idx_start+self.len_data]
        elif max(wav.shape) < self.len_data : 
            shortage = self.len_data - max(wav.shape) 
            wav = np.pad(wav,(0,shortage))
        return wav, idx_start
    
    def match_length2(self,wav,idx_start=None,wavlen=None) : 
        if max(wav.shape) > wavlen : 
            left = max(wav.shape) - wavlen
            if idx_start is None :
                idx_start = np.random.randint(left)
            if len(wav.shape) > 1:
                wav = wav[:,idx_start:idx_start+wavlen]
            else:
                wav = wav[idx_start:idx_start+wavlen]
        elif max(wav.shape) < wavlen : 
            shortage = wavlen - max(wav.shape) 
            if len(wav.shape) == 1:
                padding_config = (0, shortage)  # Padding for 1D
            elif len(wav.shape) == 2:
                padding_config = ((0, 0), (0, shortage))  # Padding for 2D
            wav = np.pad(wav, padding_config, mode='constant', constant_values=0)
        return wav, idx_start

    @staticmethod
    def norm_amplitude(y, scalar=None, eps=1e-6):
        if not scalar:
            scalar = np.max(np.abs(y)) + eps

        return y / scalar, scalar

    @staticmethod
    def tailor_dB_FS(y, target_dB_FS=-25, eps=1e-6):
        rms = np.sqrt(np.mean(y ** 2))
        scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
        y *= scalar
        return y, rms, scalar

    @staticmethod
    def is_clipped(y, clipping_threshold=0.999):
        return np.any(np.abs(y) > clipping_threshold)

    def mix(self,clean,ego,noise,rir,noise_rir=None,eps=1e-7):
        if self.hp.task == "LG_HnA":
            if rir is not None:
                clean_ = np.repeat(signal.fftconvolve(clean, rir[0,:])[:,np.newaxis],3,axis=1).T
                for i in range(3):
                    clean_[i,:] = signal.fftconvolve(clean, rir[i,:])
                clean = clean_
            if noise_rir is not None:
                noise_ = np.repeat(signal.fftconvolve(noise, noise_rir[0,:][:max(clean_.shape)])[:,np.newaxis],3,axis=1).T
                for i in range(3):
                    noise_[i,:] = signal.fftconvolve(noise, noise_rir[i,:][:max(clean_.shape)])
                noise = noise_
        else :
            if rir is not None:
                
                clean = signal.fftconvolve(clean, rir)[:len(clean)]
        clean,_ = self.match_length(clean)
        clean, _ = self.norm_amplitude(clean)
        clean, _, _ = self.tailor_dB_FS(clean, self.target_dB_FS)
        clean_rms = (clean[:-1,:] ** 2).mean() ** 0.5
        noise,_ = self.match_length2(noise,wavlen=max(clean.shape))
        noise, _ = self.norm_amplitude(noise)
        noise, _, _ = self.tailor_dB_FS(noise, self.target_dB_FS)
        noise_rms = (noise[:-1,:] ** 2).mean() ** 0.5
        ego,_ = self.match_length2(ego,wavlen=max(clean.shape))
        ego, _ = self.norm_amplitude(ego)
        ego, _, _ = self.tailor_dB_FS(ego, self.target_dB_FS)
        ego_rms = (ego[:-1,:] ** 2).mean() ** 0.5

        egoSNR = noisy_target_dB_FS = np.random.randint(
            self.range_egoSNR[0],self.range_egoSNR[1]
        )
        noiseSNR = noisy_target_dB_FS = np.random.randint(
            self.range_noiseSNR[0],self.range_noiseSNR[1]
        )
        snr_ego_scalar = clean_rms / (10 ** (egoSNR / 20)) / (ego_rms + eps)
        snr_noise_scalar = clean_rms / (10 ** (noiseSNR / 20)) / (noise_rms + eps)
        ego *= snr_ego_scalar
        noise *= snr_noise_scalar
        if not self.hp.data.use_BGnoise:
            fin_noise = ego + noise
        else :
            fin_noise = ego
            clean = clean + noise
        
        noisy = clean + fin_noise
        
            # rescale noisy RMS
        if self.is_clipped(noisy):
            noisy_scalar = np.max(np.abs(noisy)) / (0.99 - eps)  # same as divide by 1
            noisy = noisy / noisy_scalar
            clean = clean / noisy_scalar
        
        ref  = self.float_to_short(np.concatenate([noisy[2, :],noisy[2, :]])).astype(np.int16) # noise 
        mic1 = self.float_to_short(np.concatenate([noisy[0, :],noisy[0, :]])).astype(np.int16) # noisy
        mic2 = self.float_to_short(np.concatenate([noisy[1, :],noisy[1, :]])).astype(np.int16) # noisy
        aec_result1, _ = self.main_loop(ref, mic1)
        aec_result2, _ = self.main_loop(ref, mic2)
        aec_result = np.stack([aec_result1[len(noisy[2, :]):],aec_result2[len(noisy[2, :]):]],axis = 0,dtype=np.float32)
    
        
        
        # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
        noisy_target_dB_FS = np.random.randint(
            self.target_dB_FS - self.target_dB_FS_floating_value,
            self.target_dB_FS + self.target_dB_FS_floating_value
        )

        # rescale noisy RMS
        noisy[:-1,:], _, noisy_scalar = self.tailor_dB_FS(noisy[:-1,:], noisy_target_dB_FS)
        clean *= noisy_scalar
        if self.is_clipped(noisy[:-1,:]):
            noisy_scalar = np.max(np.abs(noisy[:-1,:])) / (0.99 - eps)  # same as divide by 1
            noisy = noisy / noisy_scalar
            clean = clean / noisy_scalar
            aec_result = aec_result / noisy_scalar
        noisy[-1,:] /= np.max(np.abs(noisy))
        fin_noise[-1,:] /= np.max(np.abs(noisy))
        
        if self.hp.data.SNR_target is not None:
            clean_rms = (clean[:-1,:] ** 2).mean() ** 0.5
            finnoise_rms = (fin_noise[:-1,:] ** 2).mean() ** 0.5
            SNR_ = 20 * np.log10(clean_rms / (finnoise_rms+eps) + eps)
            SNR_target = self.hp.data.SNR_target + SNR_
            
            snr_scalar = clean_rms / (10 ** (SNR_target / 20)) / (finnoise_rms + eps)
            clean += fin_noise * snr_scalar
        return noisy, clean, aec_result, fin_noise
    
    def get_clean_dev(self,path_noisy):
        path_after_root = path_noisy.split(self.hp.data.dev.root)[-1]
        dev_type = path_after_root.split("/")[1]

        fid = path_noisy.split("_")[-1]
        fid = fid.split(".")[0]
        
        path_clean = os.path.join(self.hp.data.dev.root,dev_type,"clean","clean_fileid_{}.wav".format(fid))
        return path_clean
    def get_clean_dev2(self,path_noisy):
        path_root,fid = path_noisy.split("noisy")
        path_clean = path_root+"clean"+fid
        return path_clean
    
    def get_aec_dev(self,path_noisy):
        path_root,fid = path_noisy.split("noisy")
        path_aec = path_root+"AEC"+fid
        return path_aec
    
    def __getitem__(self, idx):
        
        if self.is_train : 
            # sample clean

            path_clean = random.choice(self.list_clean)
            while "carpet" in path_clean:
                path_clean = random.choice(self.list_clean)
            clean = rs.load(path_clean,sr=self.sr)[0]

            # sample noise
            path_noise = random.choice(self.list_noise)
            noise = rs.load(path_noise,sr=self.sr,mono=False)[0]
            
            # sample noise
            path_ego = random.choice(self.list_ego)
            ego = rs.load(path_ego,sr=self.sr,mono=False)[0]

            if self.hp.data.use_RIR : 
                # sample RIR
                path_RIR = random.choice(self.list_RIR)
                RIR = rs.load(path_RIR,sr=self.sr,mono=False)[0]
                path_noise_RIR = random.choice(self.list_RIR)
                while path_noise_RIR == path_RIR:
                    path_noise_RIR = random.choice(self.list_RIR)
                RIR_Noise = rs.load(path_noise_RIR,sr=self.sr,mono=False)[0]
            else :
                RIR = None
                if len(noise.shape) > 1:
                    noise = noise[0,:].squeeze()
                if len(ego.shape) > 1:
                    ego = ego[0,:].squeeze()
                RIR_Noise = None
            
            ## Length Match
            clean,_ = self.match_length(clean)
            # noise,_ = self.match_length(noise)
            # mix 
            noisy,clean,aec_result,noise = self.mix(clean,ego,noise,RIR,RIR_Noise)
        else :
            path_noisy = self.list_noisy[idx]
            path_clean = self.get_clean_dev2(path_noisy)
            path_aec = self.get_aec_dev(path_noisy)
            # if self.hp.data.use_RIR:
            clean = rs.load(path_clean,sr=self.sr,mono=False)[0]
            noisy = rs.load(path_noisy,sr=self.sr,mono=False)[0]
            aec_result = rs.load(path_aec,sr=self.sr,mono=False)[0]
            # else:
            #     clean = rs.load(path_clean,sr=self.sr,mono=True)[0]
            #     noisy = rs.load(path_noisy,sr=self.sr,mono=True)[0]
            #     aec_result = rs.load(path_aec,sr=self.sr,mono=True)[0]
            noisy,_ = self.match_length2(noisy,wavlen=self.len_data)
            clean,_ = self.match_length2(clean,wavlen=self.len_data)
            aec_result,_ = self.match_length2(aec_result,wavlen=self.len_data)
            noise = np.zeros_like(noisy)
        
        data = {"noisy":noisy,"clean":clean,"AEC":aec_result, "noise":noise}
        return  data

    def __len__(self):
        if self.is_train : 
            # return len(self.list_clean)
            return self.n_item
        else :
            return len(self.list_clean)
        
    def get_eval(self,idx) : 

        path_reverb = self.eval["with_reverb"][idx]
        path_no_reverb = self.eval["no_reverb"][idx]

        path_clean_reverb = self.get_clean_dev(path_reverb)
        path_clean_no_reverb = self.get_clean_dev(path_no_reverb)

        return [path_reverb,path_clean_reverb],[path_no_reverb,path_clean_no_reverb]

## DEV
if __name__ == "__main__" : 
    import sys
    sys.path.append("./")
    from utils.hparams import HParam
    hp = HParam("../config/SPEAR/v20.yaml","../config/SPEAR/default.yaml")
  




