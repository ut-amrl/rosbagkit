#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

ROBOT="trevor"
DATASET_DIR=$PROJECT_DIR/data/SARA/$ROBOT
scenes=(
  2024-07-18-18-40-08
)

pc_topic=/$ROBOT/lidar_points_center

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