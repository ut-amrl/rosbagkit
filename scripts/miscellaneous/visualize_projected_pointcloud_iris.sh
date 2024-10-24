#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

ROBOT="trevor"
DATASET_DIR=$PROJECT_DIR/data/IRIS/$ROBOT

scenes=(
  2024-08-13-15-28-36
)

trap "echo 'Script interrupted'; exit;" SIGINT

for scene in "${scenes[@]}" ; do
  python $PROJECT_DIR/src/miscellaneous/visualize_projected_pointcloud.py \
    --dataset IRIS --dataset_dir $DATASET_DIR --scene $scene --cam cam_right
done