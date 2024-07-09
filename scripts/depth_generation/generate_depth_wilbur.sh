#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

dataset_dir="/home/dongmyeong/Projects/datasets/SARA/wilbur"
scenes=(
  mout-forest-loop-1_2024-04-10-10-29-03
)

trap "echo 'Script interrupted'; exit;" SIGINT

for scene in "${scenes[@]}" ; do
  python $PROJECT_DIR/src/depth_generation/generate_depth_wilbur.py \
    --dataset_dir $dataset_dir --scene $scene --window_size 31
done