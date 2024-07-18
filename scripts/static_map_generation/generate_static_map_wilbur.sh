#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

DATASET_DIR=$PROJECT_DIR/data/SARA/wilbur
scenes=(
  mout-forest-loop-1_2024-04-10-10-29-03
)

trap "echo 'Script interrupted'; exit;" SIGINT

for scene in "${scenes[@]}" ; do
  python $PROJECT_DIR/src/static_map_generation/generate_static_map.py \
    --dataset wilbur --dataset_dir $DATASET_DIR --scenes $scene \
    --blind=20.0 --voxel_size=0.5 --nb_neighbors=10 --std_ratio=1.0 --name=$scene \
    --visualize
done