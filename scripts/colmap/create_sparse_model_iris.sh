#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

DATASET_PATH=/media/dongmyeong/T9/IRIS
scenes=(
#   2024-08-12-11-32-46
#   2024-08-12-11-45-44
#   2024-08-12-12-00-13
#   2024-08-12-12-16-01
#   2024-08-12-12-29-20
#   2024-08-12-12-39-26
#   2024-08-12-17-18-39
#   2024-08-12-17-28-12
#   2024-08-12-17-41-07
#   2024-08-13-11-25-22
#   2024-08-13-11-28-57
#   2024-08-13-11-34-23
#   2024-08-13-11-37-53
#   2024-08-13-11-48-00
#   2024-08-13-11-59-52
#   2024-08-13-12-05-41
#   2024-08-13-12-13-51
#   2024-08-13-15-28-36
#   2024-08-13-15-39-15
#   2024-08-13-16-20-38
#   2024-08-13-16-27-54
#   2024-08-13-16-36-24
#   2024-08-13-16-41-44
#   2024-08-13-16-45-06
#   2024-08-13-16-48-47
#   2024-08-13-16-54-12
  2024-08-13-16-56-52
#   2024-08-13-17-07-31
#   2024-08-13-17-14-03
)

trap "echo 'Script interrupted'; exit;" SIGINT


for scene in "${scenes[@]}" ; do
  project_dir=$DATASET_PATH/colmap/$scene

  # 1. Create Project Directory
  mkdir -p $project_dir/sparse
  # ln -s $DATASET_PATH/2d_rect/cam_right/$scene $project_dir/images

  # 2. create model with known poses
  python $PROJECT_DIR/tasks/colmap/create_colmap_model.py \
    --project_dir $project_dir \
    --image_dirs $project_dir/images \
    --cam_pose_files $DATASET_PATH/poses/$scene/cam_right.txt
done 