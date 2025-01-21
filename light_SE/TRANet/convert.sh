VERSION=v2
DEVICE=cuda:0
TASK=MappingNet

set -e

echo Convert $1 $TASK $VERSION
python ./src/torch2tflite_nobuco.py \
    -c ./config/${TASK}/${VERSION}.yaml \
    --default ./config/${TASK}/default.yaml \
    -v ${VERSION} \
    --name ${TASK}_${VERSION} \

# python ./src/tfliterun.py \
#     --model_path ./Convert/${TASK}_${VERSION}.tflite \
#     --input_path /home/thk/TRANet/DB/eval_/noisy/001_female_30s_seoul-01_Noise_TV_16k_166_rec_4ch_train_5_215_-5.334552556738372.wav \
#     --output_path ./Convert/${TASK}_${VERSION}.wav
