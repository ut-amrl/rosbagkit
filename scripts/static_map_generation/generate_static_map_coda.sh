#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

dataset_dir="/home/dongmyeong/Projects/datasets/CODa"
# sequences=(0 1 2 3 4 5 6 7 9 10 11 12 13 16 17 18 19 20 21 22)
sequences=(0) 

trap "echo 'Script interrupted'; exit;" SIGINT

python $PROJECT_DIR/src/static_map_generation/generate_static_map.py \
--dataset CODa --dataset_dir $dataset_dir --scenes ${sequences[@]} \
--blind 5.0 --voxel_size 0.1 --nb_neighbors 100 --std_ratio 1.0 --visualize