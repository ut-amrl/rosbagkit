#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

dataset_dir="/home/dongmyeong/Projects/datasets/CODa"
sequences=(1)

trap "echo 'Script interrupted'; exit;" SIGINT

for seq in "${sequences[@]}" ; do
    echo "Generating depth for sequence ${seq}..."
    python $PROJECT_DIR/src/depth_generation/generate_depth_coda.py \
        --dataset_dir=${dataset_dir} \
        --seq=${seq}
done