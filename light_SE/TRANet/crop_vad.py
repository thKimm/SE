import librosa
import numpy as np
from src.utils.IIP_STFT import IIPSTFT
from src.utils.LRT_HangOverScheme import VAD
import soundfile as sf
def crop(file_path,dataset):
    clean_path = f"/home/nas3/user/thk/TRANet/DB/{dataset}/eval/clean/" + os.path.basename(file_path)
    y = librosa.load(file_path, sr=16000,mono=False)[0][0,:]
    clean = librosa.load(clean_path, sr=16000,mono=False)[0][0,:]
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    clean = clean.astype(np.float32)
    clean /= np.max(np.abs(clean))
    start_idx = 0
    end_idx = len(y)-1
    while clean[start_idx] < 0.05:
        start_idx += 1
    while clean[end_idx] < 0.05 and start_idx < end_idx:
        end_idx -= 1
    vad = np.zeros_like(y)
    vad[start_idx:end_idx] = 1
    
    return y*vad

import glob
import os
from tqdm.auto import tqdm

datasets = ["t0","t5","t10","t15"]
for dataset in datasets:
    path = f"/home/nas3/user/thk/TRANet/DB/{dataset}/eval/"
    # folder_name = ["noisy", "AEC"]
    folder_name = ["LAEC"]
    for fname in folder_name:
        audio_path = os.path.join(path, fname)
        for file in tqdm(glob.glob(os.path.join(audio_path, "*.wav")), desc=f"{dataset}_{fname}",colour="red",dynamic_ncols=True):
            result = crop(file,dataset)
            save_path = os.path.join(path, "vad", fname)
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, os.path.basename(file))
            sf.write(save_file, result, 16000)
            