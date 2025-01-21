VERSION=default
DEVICE=cuda:2
TASK=SpatialNet
VERSION_DATA=t0
set -e
VERSION=default
echo SNR evaluation $1 $VERSION

python ./src/evaluate_noisy.py \
        -c ./config/${TASK}/${VERSION}.yaml \
        --default ./config/${TASK}/default.yaml \
        -d ${DEVICE} \
        -v ${VERSION} \
        -i ./DB/${VERSION_DATA}/eval/  \
        -o ./inference/output/${TASK}_${VERSION}_${VERSION_DATA} \
        --dataset_name ${VERSION_DATA} \
        # --chkpt /home/thk/TRANet/log/LG_HnA_v4/chkpt/default/model_SNR_6.692031383514404_epoch_16.pt \




