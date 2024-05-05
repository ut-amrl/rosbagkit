#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

sequences=(1)
dataset_dir="/home/dongmyeong/Projects/datasets/CODa"

trap "echo 'Script interrupted'; exit;" SIGINT

for seq in "${sequences[@]}"; do
  python $PROJECT_DIR/py_scripts/pointcloud_compensation/compensate_pointcloud_coda.py \
    --dataset_dir $dataset_dir \
    --seq $seq
done