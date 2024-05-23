#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

sequences=(1)
dataset_dir="/home/dongmyeong/Projects/datasets/CODa"

trap "echo 'Script interrupted'; exit;" SIGINT

for seq in "${sequences[@]}"; do
  python $PROJECT_DIR/src/pointcloud_registration/gicp.py \
    --pc_dir $dataset_dir/3d_comp/os1/$seq \
    --pose_file $dataset_dir/poses/point_lio/sync_$seq.txt \
    --out_pose_file $dataset_dir/poses/gicp/$seq.txt \
    --plot
done