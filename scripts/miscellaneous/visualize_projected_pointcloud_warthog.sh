#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

ROBOT="trevor"
dataset_dir=$DATASET_DIR/SARA/$ROBOT

scenes=(
  2024-07-18-18-40-08
)

trap "echo 'Script interrupted'; exit;" SIGINT

for scene in "${scenes[@]}" ; do
  python $PROJECT_DIR/src/miscellaneous/visualize_projected_pointcloud.py \
    --dataset warthog --dataset_dir $dataset_dir --scene $scene --img_type rectified
done