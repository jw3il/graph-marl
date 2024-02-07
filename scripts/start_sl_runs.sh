#!/bin/bash

# Allows using multiple GPUs (e.g. "0 1 2 3") and also assigning multiple
# jobs to the same GPU (e.g. "0 0").
GPUS="0"

OFF="echo"

DATE=$(date +%Y%m%d_%H%M%S)
DIR_NAME="${DATE}_logs_netmon_sl"

mkdir -p $DIR_NAME

# create backup of the code
git log -n 1 >> $DIR_NAME/git.txt
git status >> $DIR_NAME/git.txt
git diff >> $DIR_NAME/git.txt
cp -r src $DIR_NAME/src
# copy this script
cp $0 $DIR_NAME/$(basename "$0")

run() {
    RUN_NAME="$1"
    shift
    RUN_ARGS="$@"
    echo "(set -x; time python -u src/sl.py $RUN_ARGS --filename=${DIR_NAME}/${RUN_NAME}.h5) > ${DIR_NAME}/${RUN_NAME}.log 2>&1"
}

REST_ARGS="--iterations=50_000 --num-samples-train=99_000 --validate-after=500 --disable-progressbar"
time (
for i in 0 1 2
do
run "netmon-1it-8seq-$i" --seed=$i --netmon-iterations=1 --sequence-length=8 $REST_ARGS
run "netmon-1it-16seq-$i" --seed=$i --netmon-iterations=1 --sequence-length=16 $REST_ARGS
run "netmon-2it-8seq-$i" --seed=$i --netmon-iterations=2 --sequence-length=8 $REST_ARGS
run "netmon-4it-8seq-$i" --seed=$i --netmon-iterations=4 --sequence-length=8 $REST_ARGS

run "gconvlstm-1it-8seq-$i" --seed=$i --netmon-iterations=1 --sequence-length=8 --netmon-agg-type=gconvlstm $REST_ARGS
run "gconvlstm-1it-16seq-$i" --seed=$i --netmon-iterations=1 --sequence-length=16 --netmon-agg-type=gconvlstm $REST_ARGS
run "gconvlstm-2it-8seq-$i" --seed=$i --netmon-iterations=2 --sequence-length=8 --netmon-agg-type=gconvlstm $REST_ARGS
run "gconvlstm-4it-8seq-$i" --seed=$i --netmon-iterations=4 --sequence-length=8 --netmon-agg-type=gconvlstm $REST_ARGS

run "graphsage-8it-1seq-$i" --seed=$i --netmon-iterations=8 --sequence-length=1 --netmon-agg-type=graphsage --netmon-rnn-type=none $REST_ARGS
run "graphsage-16it-1seq-$i" --seed=$i --netmon-iterations=16 --sequence-length=1 --netmon-agg-type=graphsage --netmon-rnn-type=none $REST_ARGS

run "antisymgcn-8it-1seq-$i" --seed=$i --netmon-iterations=8 --sequence-length=1 --netmon-agg-type=antisymgcn --netmon-rnn-type=none $REST_ARGS
run "antisymgcn-16it-1seq-$i" --seed=$i --netmon-iterations=16 --sequence-length=1 --netmon-agg-type=antisymgcn --netmon-rnn-type=none $REST_ARGS
done
) | simple_gpu_scheduler --gpus $GPUS
