#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

DATASET_DIR=$PROJECT_DIR/data/CODa
sequences=(10)

trap "echo 'Script interrupted'; exit;" SIGINT

for seq in "${sequences[@]}" ; do
    echo "Generating depth for sequence ${seq}..."
    python $PROJECT_DIR/src/depth_generation/generate_depth_coda.py \
        --dataset_dir $DATASET_DIR --seq=$seq --window_size 1 --visualize
done