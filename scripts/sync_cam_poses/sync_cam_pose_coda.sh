#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

DATASET_DIR=$PROJECT_DIR/data/CODa
sequences=(16 17)

trap "echo 'Script interrupted'; exit;" SIGINT

for seq in "${sequences[@]}" ; do
  python $PROJECT_DIR/src/miscellaneous/sync_cam_pose.py \
    --ref_pose_file $DATASET_DIR/correct/$seq.txt \
    --target_timestamps $DATASET_DIR/timestamps/$seq.txt \
    --extrinsic $DATASET_DIR/calibrations/$seq/calib_os1_to_cam0.yaml \
    --out_pose_file $DATASET_DIR/poses/cam0/$seq.txt

  python $PROJECT_DIR/src/miscellaneous/sync_cam_pose.py \
    --ref_pose_file $DATASET_DIR/correct/$seq.txt \
    --target_timestamps $DATASET_DIR/timestamps/$seq.txt \
    --extrinsic $DATASET_DIR/calibrations/$seq/calib_os1_to_cam1.yaml \
    --out_pose_file $DATASET_DIR/poses/cam1/$seq.txt
done