# pip install --upgrade google-api-python-client
# pip install google-cloud-speech

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

import os,glob
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "asr-431108-4e3c1fb3bd62.json"

def transcribe_streaming(stream_file):
    print("streaming {}".format(stream_file))
    """Streams transcription of the given audio file."""
    import io
    from google.cloud import speech

    client = speech.SpeechClient()
    
    stream = []
    with io.open(stream_file, "rb") as audio_file :
        # dump header
        audio_file.read(44)
        while True : 
            content = audio_file.read(128)
            if not content:
                # eof
                break
            stream.append(content)

    requests = (
        speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in stream
    )
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )

    # streaming_recognize returns a generator.
    responses = client.streaming_recognize(
        config=streaming_config,
        requests=requests,
    )
    
    name_file = stream_file.split("\\")[-1]
    id_file = name_file.split(".")[0]
    with open("output/{}_stream.txt".format(id_file),"w") as f : 
        for response in responses:
            # Once the transcription has settled, the first result will contain the
            # is_final result. The other results will be for subsequent portions of
            # the audio.
            for result in response.results:
                #print("{} ".format(result.is_final),end = "")
                #print("{:.2f}:".format(result.stability),end = "")
                alternatives = result.alternatives
                # best only
                f.write(u"{}| ".format(alternatives[0].transcript))
                print(u"{}| ".format(alternatives[0].transcript),end = "")
                #print("||",end = "")
            f.write("\n")
            print("")

def transcribe_file(speech_file):
    # print("file {}".format(speech_file))
    """Transcribe the given audio file."""
    from google.cloud import speech
    import io

    client = speech.SpeechClient()

    with io.open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ko-KR",
    )

    response = client.recognize(config=config, audio=audio)
    # name_file = speech_file.split("\\")[-1]
    # id_file = name_file.split(".")[0]
    # with open("output/{}_file.txt".format(id_file),"w") as f : 
    #     # Each result is for a consecutive portion of the audio. Iterate through
    #     # them to get the transcripts for the entire audio file.
    #     for result in response.results:
    #         # The first alternative is the most likely one for this portion.
    #         f.write(u"{}".format(result.alternatives[0].transcript))
    #         print(u"{}".format(result.alternatives[0].transcript))
    #     f.write("\n")
    try:
        
        return response.results[0].alternatives[0].transcript
    except:
        return ""
    
def transcribe_file(speech_file):
    # print("file {}".format(speech_file))
    """Transcribe the given audio file."""
    from google.cloud import speech
    import io

    client = speech.SpeechClient()

    with io.open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ko-KR",
    )

    response = client.recognize(config=config, audio=audio)
    try:
        
        return response.results[0].alternatives[0].transcript
    except:
        return ""
    

if __name__ == "__main__": 
    datasets = ["t0","t5"]
    model = "SpatialNet"
    version = "default"
    from tqdm.auto import tqdm
    AEC = dict()
    for dataset in datasets:
        # path = f"/home/nas3/user/thk/TRANet/DB/{dataset}/eval/vad/"
        path = f"/home/nas3/user/thk/TRANet/inference/oracleWF/"
        transcribe_path = "/home/nas3/user/thk/TRANet/transcribe/"
        os.makedirs(transcribe_path, exist_ok=True)
        # folder_name = ["noisy", "AEC"]
        folder_name = ["oracleWF_vad"]
        # folder_name = ["output"]
        for fname in folder_name:
            AEC[f"{dataset}_{fname}"] = dict()
            # audio_path = os.path.join(path, fname, f"{model}_{version}_{dataset}")
            audio_path = os.path.join(path, fname, dataset)
            print(audio_path, glob.glob(os.path.join(audio_path, "*.wav")))
            with open(os.path.join(transcribe_path, f"{dataset}_{fname}_google.txt"), "w", encoding="utf-8") as f:
            # with open(os.path.join(transcribe_path,fname, f"{model}_{version}_{dataset}_transcribe_google.txt"), "w", encoding="utf-8") as f:
                for file in tqdm(glob.glob(os.path.join(audio_path, "*.wav")), desc=f"{dataset}_{fname}",colour="red",dynamic_ncols=True):
                    text = transcribe_file(file)
                    f.write(f"{os.path.basename(file).split('_Noise')[0]}: {text}\n")
                    AEC[f"{dataset}_{fname}"][os.path.basename(file).split('_Noise')[0]] = text
                f.close()
    # list_target = glob.glob(os.path.join("data","Disney_test_streaming_ch","*.wav"))
    # print(len(list_target))
    # for path in list_target : 
    #     transcribe_streaming(path)
        # transcribe_file(path)
    
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
    import numpy as np
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
            model=model,
            version="version",
            dataset=key,
            snr_in="-",
            snr_out="-",
            snr_improvement="-",
            cer=mean_CER,
            cer_bf="-",
            cer_wf="-",
            recognizer="google"
        )
    import ipdb; ipdb.set_trace()