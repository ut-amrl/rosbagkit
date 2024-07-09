#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

dataset_dir="/home/dongmyeong/Projects/datasets/SARA/wilbur"
scenes=(
  mout-forest-loop-1_2024-04-10-10-29-03
)

pc_topic="/wilbur/lidar_points_center"

trap "echo 'Script interrupted'; exit;" SIGINT

for scene in "${scenes[@]}"; do
  # Compensate pointcloud
  python $PROJECT_DIR/src/pointcloud_compensation/compensate_pointcloud_bagfile.py \
    --bagfile $dataset_dir/bagfiles/$scene.bag \
    --pc_topic $pc_topic \
    --ref_posefile $dataset_dir/poses/$scene/fast_lio.txt \
    --out_pc_dir $dataset_dir/3d_comp/$scene \
    --out_timestamps $dataset_dir/timestamps/$scene/3d_comp.txt \
    --out_posefile $dataset_dir/poses/$scene/os1.txt
done