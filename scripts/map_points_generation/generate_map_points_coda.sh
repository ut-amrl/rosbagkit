#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

DATASET_DIR="/home/dongmyeong/Projects/datasets/CODa"
sequences=(10)
name="10"

trap "echo 'Script interrupted'; exit;" SIGINT

python $PROJECT_DIR/src/static_map_generation/generate_static_map.py \
--dataset CODa --dataset_dir $DATASET_DIR --scenes ${sequences[@]} \
--blind=15.0 --voxel_size=0.01 --nb_neighbors=10 --std_ratio=1.0 --name=$name \