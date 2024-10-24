#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

ROBOT="wilbur"
DATASET_DIR=$PROJECT_DIR/data/IRIS/$ROBOT
scenes=(
  mout-forest-loop-1_2024-04-10-10-29-03
)

trap "echo 'Script interrupted'; exit;" SIGINT

for scene in "${scenes[@]}" ; do
  python $PROJECT_DIR/src/map_points_generation/generate_map_points.py \
    --dataset IRIS --dataset_dir $DATASET_DIR --scenes $scene --name raw_$scene \
    --skip 5 --blind 3.0 --voxel_size 0.05 --nb_neighbors 10 --std_ratio 1.0 \
    --visualize
done