#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

ROBOT="trevor"
DATASET_DIR=$PROJECT_DIR/data/SARA/$ROBOT
scenes=(
  2024-07-18-18-40-08
)

trap "echo 'Script interrupted'; exit;" SIGINT

for scene in "${scenes[@]}" ; do
  python $PROJECT_DIR/src/depth_generation/generate_depth_warthog.py \
    --dataset_dir $DATASET_DIR --scene $scene --window_size 31
done