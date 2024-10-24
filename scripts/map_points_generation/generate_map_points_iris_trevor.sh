#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

ROBOT="trevor"
DATASET_DIR=$PROJECT_DIR/data/IRIS/$ROBOT
scenes=(
  # 2024-08-13-15-28-36
  # 2024-08-13-16-56-52
  2024-08-13-17-07-31
)

trap "echo 'Script interrupted'; exit;" SIGINT

for scene in "${scenes[@]}" ; do
  python $PROJECT_DIR/src/map_points_generation/generate_map_points.py \
    --dataset IRIS --dataset_dir $DATASET_DIR --scenes $scene --name raw_$scene \
    --skip 3 --blind 1.0 --voxel_size 0.03 --nb_neighbors 10 --std_ratio 1.0 \
    --visualize
done