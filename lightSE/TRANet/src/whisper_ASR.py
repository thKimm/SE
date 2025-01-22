from transformers import pipeline
import librosa
import numpy as np
from utils.IIP_STFT import IIPSTFT
from utils.LRT_HangOverScheme import VAD
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device="cuda:2")

import csv

def log_evaluation_scores(model, version, dataset, snr_in, snr_out, snr_improvement, cer, cer_bf, cer_wf, recognizer):
    # CSV 파일의 경로 설정
    csv_file = 'evaluation_scores.csv'

    # 헤더 정의
    fieldnames = ['Model', 'Version', 'Dataset', 'SNR-in', 'SNR-out' ,'SNR-Improvement', 'CER', 'CER_BF', 'CER_WF', 'recognizer']

    # 파일이 존재하는지 확인
    file_exists = os.path.isfile(csv_file)

    # 파일 열기 (존재하지 않으면 새 파일 생성)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # 파일이 존재하지 않으면 헤더 작성
        if not file_exists:
            writer.writeheader()

        # 성능 평가 스코어 작성
        writer.writerow({
            'Model': model,
            'Version': version,
            'Dataset': dataset,
            'SNR-in' : snr_in,
            'SNR-out' : snr_out,
            'SNR-Improvement': snr_improvement,
            'CER':    f'{cer*100:.2f}',
            'CER_BF': cer_bf,
            'CER_WF': cer_wf,
            'recognizer': recognizer
        })


def transcribe(file_path):
    file_name = os.path.basename(file_path)
    # clean_path = "/home/TRANet/DB/d2/eval/clean/" + file_name
    y, sr = librosa.load(file_path, sr=None,mono=True)
    # clean, _ = librosa.load(file_path, sr=sr,mono=True)
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    # clean = clean.astype(np.float32)
    # clean /= np.max(np.abs(clean))
    # vad = np.abs(clean) > 0.2
    
    # return transcriber({"sampling_rate": sr, "raw": y*vad}, generate_kwargs={"language" : "korean"})["text"] 
    return transcriber({"sampling_rate": sr, "raw": y}, generate_kwargs={"language" : "korean"})["text"] 

def transcribe_vad(file_path,dataset):
    clean_path = f"/home/nas3/user/thk/TRANet/DB/{dataset}/eval/clean/" + os.path.basename(file_path)
    y = librosa.load(file_path, sr=16000,mono=False)[0]
    if len(y.shape) > 1:
        y = y[0,:]
    clean = librosa.load(clean_path, sr=16000,mono=False)[0][0,:]
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    clean = clean.astype(np.float32)
    clean /= np.max(np.abs(clean))
    start_idx = 0
    end_idx = len(y)-1
    while clean[start_idx] < 0.05:
        start_idx += 1
    while clean[end_idx] < 0.05:
        end_idx -= 1
    vad = np.zeros_like(y)
    vad[start_idx:end_idx] = 1
    
    return transcriber({"sampling_rate": 16000, "raw": y*vad}, generate_kwargs={"language" : "korean"})["text"] 

import glob
import os
from tqdm.auto import tqdm

# datasets = ["t0","t5","t10","t15"]
datasets = ["t0","t5"]
AEC = dict()
transcribe_path = "/home/nas3/user/thk/TRANet/transcribe/"
# folder_name = ["clean", "noisy", "AEC"]
# folder_name = ["noisy", "AEC"]
model = "MappingNet"
# model = "SpatialNet"
# version_name = ["default","t3","t9"]
version_name = ["v3","v9"]
# version_name = ["default"]
folder_name = ["output","WF","BF"]
# folder_name = ["output"]
for v in version_name:
    for dataset in datasets:
        for fname in folder_name:
            # path = f"/home/nas3/user/thk/TRANet/DB/{dataset}/eval/"
            path = f"/home/nas3/user/thk/TRANet/inference/"
            transcribe_name = f"{model}_{v}_{dataset}"
            audio_path = os.path.join(path,fname,transcribe_name)
            AEC[f"{transcribe_name}_{fname}"] = dict()
            with open(os.path.join(transcribe_path,fname, f"{dataset}_{transcribe_name}_whisper.txt"), "w", encoding="utf-8") as f:
                for file in tqdm(glob.glob(os.path.join(audio_path, "*.wav")), desc=f"{transcribe_name}",colour="red",dynamic_ncols=True):
                    text = transcribe(file)
                    # print(text)
                    f.write(f"{os.path.basename(file).split('_Noise')[0]}: {text}\n")
                    AEC[f"{transcribe_name}_{fname}"][os.path.basename(file).split('_Noise')[0]] = text
                f.close()
            AEC[f"{transcribe_name}_{fname}_vad"] = dict()
            with open(os.path.join(transcribe_path, 'vad',f"{dataset}_{transcribe_name}_whisper.txt"), "w", encoding="utf-8") as f:
                for file in tqdm(glob.glob(os.path.join(audio_path, "*.wav")), desc=f"{dataset}_{transcribe_name}_vad",colour="blue",dynamic_ncols=True):
                    try:
                        text = transcribe_vad(file,dataset)
                    except:
                        text = transcribe_vad(file,dataset)
                        import ipdb; ipdb.set_trace()
                    # print(text)
                    f.write(f"{os.path.basename(file).split('_Noise')[0]}: {text}\n")
                    AEC[f"{transcribe_name}_{fname}_vad"][os.path.basename(file).split('_Noise')[0]] = text
                f.close()

            ''' when transcription is exist'''
            # AEC[f"{transcribe_name}_{fname}"] = dict()
            # with open(os.path.join(transcribe_path,f"{dataset}_{transcribe_name}.txt"), "r", encoding="utf-8") as f:
            #     for line in f:
            #         line = line.strip().split(":")
            #         AEC[f"{transcribe_name}_{fname}"][line[0]] = line[1].lstrip()
            # AEC[f"{transcribe_name}_{fname}_vad"] = dict()
            # with open(os.path.join(transcribe_path, f"{dataset}_{transcribe_name}_vad.txt"), "r", encoding="utf-8") as f:
            #     for line in f:
            #         line = line.strip().split(":")
            #         AEC[f"{transcribe_name}_{fname}_vad"][line[0]] = line[1].lstrip()
                
label_transcribe = {}
label_txt = "/home/nas3/user/thk/TRANet/DB/eval/label_transcribe.txt"
import utils.whisper_metric as whisper_metric
CER = whisper_metric.CharacterErrorRate()
CER_result = dict()
with open(label_txt, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip().split(":")
        label_transcribe[line[0].split("_Noise")[0]] = line[1].lstrip()
    f.close()
for key in AEC.keys():
    dist = 0
    length = 0
    CER_result[key] = []
    for line in AEC[key].keys():
        dist, length = CER.metric(label_transcribe[line],AEC[key][line])
        cer = dist / length
        CER_result[key].append(cer)
    mean_CER = np.mean(CER_result[key])
    
    log_evaluation_scores(
            model="-",
            version="-",
            dataset=key,
            snr_in="-",
            snr_out="-",
            snr_improvement='-',
            cer=mean_CER,
            cer_bf="-",
            cer_wf="-",
            recognizer="wihsper"
        )