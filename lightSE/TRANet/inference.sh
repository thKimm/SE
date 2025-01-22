VERSION=default
echo Train $1
DEVICE=cuda:1
TASK=SpatialNet
VERSION_DATA=t0

echo Train $1 $VERSION
python ./src/infer.py -c ./config/${TASK}/${VERSION}.yaml --default ./config/${TASK}/default.yaml -v ${VERSION} -d ${DEVICE} -i ./inference/input -o ./inference/output
#python ./src/train.py -c ./config/mpSE/${VERSION}.yaml --default ./config/mpSE/default.yaml -v ${VERSION} -d ${DEVICE} --chkpt /home/nas/user/kbh/mpSE/chkpt/${VERSION}/bestmodel.pt --step 1308000 -e 300

