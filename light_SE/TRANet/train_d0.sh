VERSION=v7
echo Train $1
DEVICE=cuda:0
TASK=LGHnA
VERSION_DATA=v0

echo Train $1 $VERSION
python ./src/train.py -c ./config/${TASK}/${VERSION}.yaml --default ./config/${TASK}/default.yaml -v ${VERSION} -d ${DEVICE} 
#python ./src/train.py -c ./config/mpSE/${VERSION}.yaml --default ./config/mpSE/default.yaml -v ${VERSION} -d ${DEVICE} --chkpt /home/nas/user/kbh/mpSE/chkpt/${VERSION}/bestmodel.pt --step 1308000 -e 300

