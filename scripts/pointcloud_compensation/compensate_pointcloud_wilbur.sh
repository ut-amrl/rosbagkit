#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

DATASET_DIR=$PROJECT_DIR/data/SARA/wilbur
scenes=(
  mout-forest-loop-1_2024-04-10-10-29-03
)

pc_topic="/wilbur/lidar_points_center"

trap "echo 'Script interrupted'; exit;" SIGINT

for scene in "${scenes[@]}"; do
  # Compensate pointcloud
  python $PROJECT_DIR/src/pointcloud_compensation/compensate_pointcloud_bagfile.py \
    --bagfile $DATASET_DIR/bagfiles/$scene.bag \
    --pc_topic $pc_topic \
    --ref_posefile $DATASET_DIR/poses/$scene/fast_lio.txt \
    --out_pc_dir $DATASET_DIR/3d_comp/$scene \
    --out_timestamps $DATASET_DIR/timestamps/$scene/os1.txt \
    --out_posefile $DATASET_DIR/poses/$scene/os1.txt
done