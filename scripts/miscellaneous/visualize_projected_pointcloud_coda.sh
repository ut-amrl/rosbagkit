#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

dataset_dir="/home/dongmyeong/Projects/datasets/CODa"
sequences=(9)

trap "echo 'Script interrupted'; exit;" SIGINT

for seq in "${sequences[@]}" ; do
  python $PROJECT_DIR/src/miscellaneous/visualize_projected_pointcloud.py \
    --dataset CODa --dataset_dir $dataset_dir --scene $seq --img_type rectified
done