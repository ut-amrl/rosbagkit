#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

dataset_dir="/home/dongmyeong/Projects/datasets/CODa"
sequences=(12 13)

trap "echo 'Script interrupted'; exit;" SIGINT

for seq in "${sequences[@]}" ; do
  python $PROJECT_DIR/src/miscellaneous/sync_cam_pose.py \
    --ref_pose_file $dataset_dir/correct/$seq.txt \
    --target_timestamps $dataset_dir/timestamps/$seq.txt \
    --extrinsic $dataset_dir/calibrations/$seq/calib_os1_to_cam0.yaml \
    --out_pose_file $dataset_dir/poses/cam0/$seq.txt

  python $PROJECT_DIR/src/miscellaneous/sync_cam_pose.py \
    --ref_pose_file $dataset_dir/correct/$seq.txt \
    --target_timestamps $dataset_dir/timestamps/$seq.txt \
    --extrinsic $dataset_dir/calibrations/$seq/calib_os1_to_cam1.yaml \
    --out_pose_file $dataset_dir/poses/cam1/$seq.txt
done