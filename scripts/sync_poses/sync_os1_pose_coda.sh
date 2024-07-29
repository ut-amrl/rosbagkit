#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

DATASET_DIR=$PROJECT_DIR/data/CODa
sequences=(10 13)

trap "echo 'Script interrupted'; exit;" SIGINT

for seq in "${sequences[@]}" ; do
  python $PROJECT_DIR/src/miscellaneous/sync_poses.py \
    --ref_pose_file $DATASET_DIR/poses/interactive_slam/$seq.txt \
    --target_timestamps $DATASET_DIR/timestamps/$seq.txt \
    --out_pose_file $DATASET_DIR/poses/interactive_slam_sync/$seq.txt
done