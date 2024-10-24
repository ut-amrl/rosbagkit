#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

ROBOT=trevor
DATASET_DIR=$PROJECT_DIR/data/IRIS/$ROBOT
scenes=(
  2024-08-12-11-32-46
  # 2024-08-12-11-45-44
  # 2024-08-12-12-00-13
  # 2024-08-12-12-16-01
  # 2024-08-12-12-29-20
  # 2024-08-12-12-39-26
  # 2024-08-12-17-18-39
  # 2024-08-12-17-28-12
  # 2024-08-12-17-41-07
  # 2024-08-13-11-25-22
  # 2024-08-13-11-28-57
  # 2024-08-13-11-34-23
  # 2024-08-13-11-37-53
  # 2024-08-13-11-48-00
  # 2024-08-13-11-59-52
  # 2024-08-13-12-05-41
  # 2024-08-13-12-13-51
  # 2024-08-13-15-28-36
  # 2024-08-13-15-39-15
  # 2024-08-13-16-20-38
  # 2024-08-13-16-27-54
  # 2024-08-13-16-36-24
  # 2024-08-13-16-41-44
  # 2024-08-13-16-45-06
  # 2024-08-13-16-48-47
  # 2024-08-13-16-54-12
  # 2024-08-13-16-56-52
  # 2024-08-13-17-07-31
  # 2024-08-13-17-14-03
)

pc_topic=/$ROBOT/lidar_points

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