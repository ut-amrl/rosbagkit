#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

ROBOT="trevor"
DATASET_DIR=$PROJECT_DIR/data/IRIS/$ROBOT

scenes=(
  2024-08-13-15-28-36
  # 2024-08-13-16-56-52
)

trap "echo 'Script interrupted'; exit;" SIGINT

for scene in "${scenes[@]}" ; do
  python $PROJECT_DIR/src/miscellaneous/visualize_rgbd_image.py \
    --image_dir $DATASET_DIR/2d_rect/cam_right/$scene \
    --depth_dir $DATASET_DIR/2d_depth/cam_right/$scene
done