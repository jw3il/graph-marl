#!/bin/bash

# Allows using multiple GPUs (e.g. "0 1 2 3") and also assigning multiple
# jobs to the same GPU (e.g. "0 0").
GPUS="0"

BASE_PARAMS="--gamma=0.9 --epsilon=1.0 --hidden-dim=512,256 --mini-batch-size=32 --device=cuda --lr=0.001 --tau=0.01 --step-before-train=10_000 --capacity=200_000 --eval-episodes=1000 --eval-episode-steps=300 --disable-progressbar"
LIMITS="--step-between-train=10 --total-steps=250_000"
RECURRENT="--sequence-length=8"

OFF="echo"

DATE=$(date +%Y%m%d_%H%M%S)
DIR_NAME="${DATE}_logs_fixed_baseline"

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
    echo "(set -x; time python -u src/main.py $RUN_ARGS --comment=${RUN_NAME}) > ${DIR_NAME}/${RUN_NAME}.log 2>&1"
}

time (
# Graphs G_A, G_B, G_C, selected
for seed in 971182936 923430603 1704443687 324821133
do
for i in 0 1 2
do
run "fixed-nocong-shortest-paths-eval-t$seed-$i" --policy=heuristic --eval --seed=$i --no-congestion --topology-init-seed=$seed --train-topology-allow-eval-seed --episode-steps=300 --random-topology=0 --disable-progressbar --eval-output-dir=${DIR_NAME}/fixed-nocong-shortest-paths-eval-t$seed-$i/eval
run "fixed-nocong-dqn-t$seed-$i" --seed=$i --no-congestion --topology-init-seed=$seed --train-topology-allow-eval-seed --episode-steps=300 --model=dqn --random-topology=0 $BASE_PARAMS $LIMITS
run "fixed-nocong-dqnr-t$seed-$i" --seed=$i --no-congestion --topology-init-seed=$seed --train-topology-allow-eval-seed --episode-steps=300 --model=dqnr --random-topology=0 $BASE_PARAMS $RECURRENT $LIMITS
run "fixed-nocong-commnet-t$seed-$i" --seed=$i --no-congestion --topology-init-seed=$seed --train-topology-allow-eval-seed --episode-steps=300 --model=commnet --random-topology=0 $BASE_PARAMS $RECURRENT $LIMITS
run "fixed-nocong-dgn-t$seed-$i" --seed=$i --no-congestion --topology-init-seed=$seed --train-topology-allow-eval-seed --episode-steps=300 --model=dgn --random-topology=0 $BASE_PARAMS $LIMITS

run "fixed-shortest-paths-eval-t$seed-$i" --policy=heuristic --eval --seed=$i --topology-init-seed=$seed --train-topology-allow-eval-seed --episode-steps=300 --random-topology=0 --disable-progressbar --eval-output-dir=${DIR_NAME}/fixed-shortest-paths-eval-t$seed-$i/eval
run "fixed-dqn-t$seed-$i" --seed=$i --topology-init-seed=$seed --train-topology-allow-eval-seed --episode-steps=300 --model=dqn --random-topology=0 $BASE_PARAMS $LIMITS
run "fixed-dqnr-t$seed-$i" --seed=$i --topology-init-seed=$seed --train-topology-allow-eval-seed --episode-steps=300 --model=dqnr --random-topology=0 $BASE_PARAMS $RECURRENT $LIMITS
run "fixed-commnet-t$seed-$i" --seed=$i --topology-init-seed=$seed --train-topology-allow-eval-seed --episode-steps=300 --model=commnet --random-topology=0 $BASE_PARAMS $RECURRENT $LIMITS
run "fixed-dgn-t$seed-$i" --seed=$i --topology-init-seed=$seed --train-topology-allow-eval-seed --episode-steps=300 --model=dgn --random-topology=0 $BASE_PARAMS $LIMITS
done
done
) | simple_gpu_scheduler --gpus $GPUS
