VERSION=default
DEVICE=cuda:1
TASK=SpatialNet
VERSION_DATA=t0
set -e
echo SNR, CER, CER - BF, CER - WF evaluation at $1 $TASK $VERSION for $VERSION_DATA

python ./src/evaluation_all_metric.py \
        -c ./config/${TASK}/${VERSION}.yaml \
        --default ./config/${TASK}/default.yaml \
        -d ${DEVICE} \
        -t ${TASK} \
        -v ${VERSION} \
        --dataset_name ${VERSION_DATA} \
        -i ./DB/${VERSION_DATA}/eval/  \
        -r google # whisper, google
        -o ./inference/output/${TASK}_${VERSION}_${VERSION_DATA} \
        # -b ./inference/BF/${TASK}_${VERSION}_${VERSION_DATA} \
        # -w ./inference/WF/${TASK}_${VERSION}_${VERSION_DATA} \
        # --chkpt /home/thk/TRANet/log/LG_HnA_v3/chkpt/default/model_SNR_6.692031383514404_epoch_16.pt \




