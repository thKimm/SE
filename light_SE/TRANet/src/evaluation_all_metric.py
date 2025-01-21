
import argparse
import os,glob
import numpy as np
import librosa as rs
import soundfile as sf
from Dataset.DatasetGender import DatasetGender
from Dataset.DatasetSPEAR import DatasetSPEAR
from Dataset.DatasetDNS import DatasetDNS

from utils.hparams import HParam
from utils.metric import run_metric

from tqdm.auto import tqdm

# from common_sep import run,get_model
from common import run,get_model

from utils.IIP_STFT import IIPSTFT
from utils.mcwf import MCWF
from utils.MVDR_RLS import mvdr_rls
import utils.whisper_metric as whisper_metric
nfft = 512
nShift = nfft//4
fs =16000
nch = 2
leftval = 10
rightval = 0
diagval = 1e-3
CER = whisper_metric.CharacterErrorRate()
iipSTFT = IIPSTFT(nfft,nfft,nShift,fs)

import torch
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
            'SNR-in' : f'{snr_in:.2f}',
            'SNR-out' : f'{snr_out:.2f}',
            'SNR-Improvement': f'{snr_improvement:.2f}',
            'CER': f'{cer*100:.2f}%',
            'CER_BF': f'{cer_bf*100:.2f}%',
            'CER_WF': f'{cer_wf*100:.2f}%',
            'recognizer': recognizer
        })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--default', type=str, required=True,
                        help="default configuration")
    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--dataset_name', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--task','-t',type=str,required=True)
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--device','-d',type=str,required=False,default="cpu")
    parser.add_argument('--dir_input','-i',type=str,required=True)
    parser.add_argument('--dir_output','-o',type=str,required=False,default=None)
    parser.add_argument('--dir_BF','-b',type=str,required=False,default=None)
    parser.add_argument('--dir_WF','-w',type=str,required=False,default=None)
    parser.add_argument('--recognizer','-r',type=str,required=False,default=None)
    args = parser.parse_args()
    if args.recognizer == 'whisper':
        from transformers import pipeline
    elif args.recognizer == 'google':
        from google_ASR import transcribe_file
    # settings
    # hp = HParam(args.config,args.default)
    hp = HParam(args.config)
    
    dataset_name = args.dataset_name
    
    # if args.dir_output is None and os.path.exists(f'./inference/output/{args.task}_{args.version_name}_{dataset_name}'):
    #     args.dir_output = f'./inference/output/{args.task}_{args.version_name}_{dataset_name}'
    # if args.dir_BF is None and os.path.exists(f'./inference/BF/{args.task}_{args.version_name}_{dataset_name}'):
    #     args.dir_BF = f'./inference/BF/{args.task}_{args.version_name}_{dataset_name}'
    # if args.dir_WF is None and os.path.exists(f'./inference/WF/{args.task}_{args.version_name}_{dataset_name}'):
    #     args.dir_WF = f'./inference/WF/{args.task}_{args.version_name}_{dataset_name}'
    
    
    DL_trasncribe_path = f'./transcribe/output/{args.task}_{args.version_name}_{dataset_name}_transcribe_{args.recognizer}.txt'
    BF_trasncribe_path = f'./transcribe/BF/{args.task}_{args.version_name}_{dataset_name}_transcribe_{args.recognizer}.txt'                         
    WF_transcribe_path = f'./transcribe/WF/{args.task}_{args.version_name}_{dataset_name}_transcribe_{args.recognizer}.txt'
    version = args.version_name

    if args.dir_BF is None :
        MVDR = mvdr_rls(nfft, nch, nShift)
        args.dir_BF = f'./inference/BF/{args.task}_{args.version_name}_{args.dataset_name}/'
        if not os.path.exists(args.dir_BF):
            os.makedirs(args.dir_BF,exist_ok=True)
    else :
        MVDR = None
    if args.dir_WF is None :
        MCWF = MCWF
        args.dir_WF = f'./inference/WF/{args.task}_{args.version_name}_{args.dataset_name}/'
        if not os.path.exists(args.dir_WF):
            os.makedirs(args.dir_WF,exist_ok=True)
    else :
        MCWF = None
    if not os.path.exists(DL_trasncribe_path):
        if args.recognizer == 'whisper':
            def transcribe_whisper(audio):
                transcribe = transcriber({"sampling_rate": hp.data.sr, "raw": audio}, generate_kwargs={"language" : "korean"})["text"]
                return transcribe
            transcriber = transcribe_whisper
        elif args.recognizer == 'google':
            def transcribe_google(audio):
                sf.write(f"{args.device}tmp.wav",audio,hp.data.sr)
                transcribe = transcribe_file(f"{args.device}tmp.wav")
                os.remove(f"{args.device}tmp.wav")
                return transcribe
            transcriber = transcribe_google
        ASR = dict()
    else :
        transcriber = None
        ASR = dict()
        with open(DL_trasncribe_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split(":")
                file_name = line[0].split("_Noise")[0]
                transcribe = line[1].lstrip()
                ASR[file_name] = transcribe
            f.close()
    if not os.path.exists(BF_trasncribe_path):
        if not transcriber:
            if args.recognizer == 'whisper':
                def transcribe_whisper(audio):
                    transcribe = transcriber({"sampling_rate": hp.data.sr, "raw": audio}, generate_kwargs={"language" : "korean"})["text"]
                    return transcribe
                transcriber = transcribe_whisper
            elif args.recognizer == 'google':
                def transcribe_google(audio):
                    sf.write(f"{args.device}tmp.wav",audio,hp.data.sr)
                    transcribe = transcribe_file(f"{args.device}tmp.wav")
                    os.remove(f"{args.device}tmp.wav")
                    return transcribe
                transcriber = transcribe_google
        ASR_BF = dict()
    else :
        ASR_BF = dict()
        with open(BF_trasncribe_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split(":")
                file_name = line[0].split("_Noise")[0]
                transcribe = line[1].lstrip()
                ASR_BF[file_name] = transcribe
            f.close()
    if not os.path.exists(WF_transcribe_path):
        if not transcriber:
            if args.recognizer == 'whisper':
                def transcribe_whisper(audio):
                    transcribe = transcriber({"sampling_rate": hp.data.sr, "raw": audio}, generate_kwargs={"language" : "korean"})["text"]
                    return transcribe
                transcriber = transcribe_whisper
            elif args.recognizer == 'google':
                def transcribe_google(audio):
                    sf.write(f"{args.device}tmp.wav",audio,hp.data.sr)
                    transcribe = transcribe_file(f"{args.device}tmp.wav")
                    os.remove(f"{args.device}tmp.wav")
                    return transcribe
                transcriber = transcribe_google
        ASR_WF = dict()
    else :
        ASR_WF = dict()
        with open(WF_transcribe_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split(":")
                file_name = line[0].split("_Noise")[0]
                transcribe = line[1].lstrip()
                ASR_WF[file_name] = transcribe
            f.close()

    global device
    device = args.device
    if args.dir_output is None:
        model = get_model(hp,args.device)
        if args.chkpt is None:
            modelsave_path = hp.log.root +'/'+'chkpt' + '/' + version
            chkpt = modelsave_path + '/bestmodel.pt'
        else :
            chkpt = args.chkpt
        try : 
            model.load_state_dict(torch.load(chkpt, map_location=device)["model"])
        except KeyError :
            model.load_state_dict(torch.load(chkpt, map_location=device))
        output_path = f'./inference/output/{args.task}_{args.version_name}_{args.dataset_name}/'
        if not os.path.exists(output_path):
            os.makedirs(output_path,exist_ok=True)
    else :
        output_path = args.dir_output
    #### EVAL ####
    ## Metric
    list_clean = [x for x in glob.glob(os.path.join(args.dir_input,"clean","**","*.wav"),recursive=True)]
    # SNR-i
    metric = dict()
    metric["SNR_in"] = 0.0
    metric["SNR_out"] = 0.0
    # CER
    label_transcribe = {}
    label_txt = "/home/nas3/user/thk/TRANet/DB/eval/label_transcribe.txt"
    with open(label_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split(":")
            label_transcribe[line[0].split("_Noise")[0]] = line[1].lstrip()
        f.close()
    CER_DL = []
    CER_BF = []
    CER_WF = []
    cer_length = 0
    # eval
    for i in tqdm(list_clean, desc=f"evaluate",colour="red",dynamic_ncols=True):

        clean = rs.load(i,sr=hp.data.sr,mono=False)[0]
        noisy_path = args.dir_input + 'noisy/' + i.split("/")[-1]
        noisy = rs.load(noisy_path,sr=hp.data.sr,mono=False)[0]
        AEC_path = args.dir_input + 'AEC/' + i.split("/")[-1]
        AEC = rs.load(AEC_path,sr=hp.data.sr,mono=False)[0]
        if args.dir_output is not None :
            estim_path = args.dir_output+'/' + i.split("/")[-1]
            estim = rs.load(estim_path,sr=hp.data.sr,mono=True)[0]
        else :
            estim = run(hp,{"noisy":torch.tensor(noisy).unsqueeze(0),"AEC":torch.tensor(AEC).unsqueeze(0)},model,ret_output=True,device=args.device)
            # estim = run(hp,{"noisy":torch.tensor(noisy).unsqueeze(0)},model,ret_output=True,device=args.device)
            estim = estim[0].to("cpu").detach().numpy().squeeze()
        # SNR improve
        val_in = run_metric(noisy[0,:],clean[0,:],"SNR") 
        metric["SNR_in"] += val_in
        val_out = run_metric(estim,clean[0,:len(estim)],"SNR") 
        metric["SNR_out"] += val_out
        noisy = noisy[:2,:]/np.max(np.abs(estim))
        estim = estim/np.max(np.abs(estim))
        sf.write(os.path.join(output_path ,i.split("/")[-1]),estim,samplerate=hp.data.sr)
        # post processing
        noisy = noisy[:2,:]
        if MVDR is not None or MCWF is not None:
            X = iipSTFT.STFT(noisy.T)
            S = iipSTFT.STFT(estim.reshape(-1,1))
        if MVDR is not None:
            estim_BF = MVDR.process(X,S)
            estim_bf = iipSTFT.iSTFT(estim_BF,noisy.shape[1])
            sf.write(os.path.join(args.dir_BF,i.split("/")[-1]),estim_bf,samplerate=hp.data.sr)
        else :
            estim_bf = rs.load(os.path.join(args.dir_BF,i.split("/")[-1]),sr=hp.data.sr,mono=True)[0]
        if MCWF is not None :
            estim_WF = MCWF(X, S, leftval, rightval, diagval)
            estim_wf = iipSTFT.iSTFT(estim_WF[0],noisy.shape[1])
            sf.write(os.path.join(args.dir_WF ,i.split("/")[-1]),estim_wf,samplerate=hp.data.sr)
        else :
            estim_wf = rs.load(os.path.join(args.dir_WF,i.split("/")[-1]),sr=hp.data.sr,mono=True)[0]
        # CER
        try :  
            dist, length = CER.metric(label_transcribe[i.split("/")[-1].split("_Noise")[0]],ASR[i.split("/")[-1].split("_Noise")[0]])
            # CER_DL.append(dist)
            # cer_length += length
            CER_DL.append(dist/length)
        except :
            estim_ = estim.astype(np.float32)
            estim_ /= np.max(np.abs(estim_))
            # text = transcriber({"sampling_rate": hp.data.sr, "raw": estim_}, generate_kwargs={"language" : "korean"})["text"]
            text = transcriber(estim_)
            ASR[i.split("/")[-1].split("_Noise")[0]] = text
            dist, length = CER.metric(label_transcribe[i.split("/")[-1].split("_Noise")[0]],text)
            # CER_DL.append(dist)
            # cer_length += length
            CER_DL.append(dist/length)
        try :
            dist, length = CER.metric(label_transcribe[i.split("/")[-1].split("_Noise")[0]],ASR_BF[i.split("/")[-1].split("_Noise")[0]])
            CER_BF.append(dist/length)
        except :
            estim_bf_ = estim_bf.astype(np.float32)
            estim_bf_ /= np.max(np.abs(estim_bf_))
            # text = transcriber({"sampling_rate": hp.data.sr, "raw": np.squeeze(estim_bf_)}, generate_kwargs={"language" : "korean"})["text"] 
            text = transcriber(estim_bf_)
            ASR_BF[i.split("/")[-1].split("_Noise")[0]] = text
            dist, length = CER.metric(label_transcribe[i.split("/")[-1].split("_Noise")[0]],text)
            # CER_BF.append(dist)
            CER_BF.append(dist/length)
        try :
            dist, length = CER.metric(label_transcribe[i.split("/")[-1].split("_Noise")[0]],ASR_WF[i.split("/")[-1].split("_Noise")[0]])
            CER_WF.append(dist/length)
        except :
            estim_wf_ = estim_wf.astype(np.float32)
            estim_wf_ /= np.max(np.abs(estim_wf_))
            # text = transcriber({"sampling_rate": hp.data.sr, "raw": np.squeeze(estim_wf_)}, generate_kwargs={"language" : "korean"})["text"]
            text = transcriber(estim_wf_)
            ASR_WF[i.split("/")[-1].split("_Noise")[0]] = text
            dist, length = CER.metric(label_transcribe[i.split("/")[-1].split("_Noise")[0]],text)
            CER_WF.append(dist/length)
    
    cer    = np.mean(CER_DL)
    cer_bf = np.mean(CER_BF)
    cer_wf = np.mean(CER_WF)
    metric["SNR_in"] /= len(list_clean)
    metric["SNR_out"] /= len(list_clean)
    if not os.path.exists(DL_trasncribe_path):
        with open(DL_trasncribe_path, "w", encoding="utf-8") as f:
            for file in ASR.keys():
                f.write(f"{file}: {ASR[file]}\n")
            f.close()
    if not os.path.exists(BF_trasncribe_path):
        with open(BF_trasncribe_path, "w", encoding="utf-8") as f:
            for file in ASR_BF.keys():
                f.write(f"{file}: {ASR_BF[file]}\n")
            f.close()
    if not os.path.exists(WF_transcribe_path):
        with open(WF_transcribe_path, "w", encoding="utf-8") as f:
            for file in ASR_WF.keys():
                f.write(f"{file}: {ASR_WF[file]}\n")
            f.close()
        
    log_evaluation_scores(
        model=hp.model.type,
        version=args.version_name,
        dataset=dataset_name,
        snr_in=metric["SNR_in"],
        snr_out=metric["SNR_out"],
        snr_improvement=metric["SNR_out"]-metric["SNR_in"],
        cer=cer,
        cer_bf=cer_bf,
        cer_wf=cer_wf,
        recognizer=args.recognizer
    )
    # # output directory check
    # os.makedirs(f"evaluation/{hp.task}",exist_ok=True)
    # os.makedirs(f"evaluation/{hp.task}/{hp.model.type}",exist_ok=True)
    # os.makedirs(f"evaluation/{hp.task}/{hp.model.type}/{version}",exist_ok=True)
    # with open(f"evaluation/{hp.task}/{hp.model.type}/{version}/log_noisy_pre.txt",'w') as f :
    #     for k in metric.keys() :
    #         f.write("'{}':'{}'\n".format(k, metric[k]))
    #     f.write(f'\nSNR improvement : {metric["SNR_out"]-metric["SNR_in"]}\n')


