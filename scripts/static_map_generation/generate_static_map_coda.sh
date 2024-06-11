#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

dataset_dir="/home/dongmyeong/Projects/datasets/CODa"
sequences=(2 5 10)

trap "echo 'Script interrupted'; exit;" SIGINT

python $PROJECT_DIR/src/static_map_generation/generate_static_map.py \
--dataset CODa --dataset_dir=$dataset_dir --scenes ${sequences[@]} \
--blind=15.0 --voxel_size=0.01 --nb_neighbors=10 --std_ratio=1.0 --name=2510