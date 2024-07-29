#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

ROBOT="trevor"
DATASET_DIR=$PROJECT_DIR/data/SARA/$ROBOT
scenes=(
  2024-07-18-18-40-08
)

trap "echo 'Script interrupted'; exit;" SIGINT

for scene in "${scenes[@]}" ; do
  python $PROJECT_DIR/src/miscellaneous/sync_poses.py \
    --ref_pose_file $DATASET_DIR/poses/$scene/os1.txt \
    --target_timestamps $DATASET_DIR/timestamps/$scene/img_aux_front.txt \
    --extrinsic $DATASET_DIR/calibrations/$scene/os_to_cam_aux_front.yaml \
    --out_pose_file $DATASET_DIR/poses/$scene/cam_aux_front.txt

  python $PROJECT_DIR/src/miscellaneous/sync_poses.py \
    --ref_pose_file $DATASET_DIR/poses/$scene/os1.txt \
    --target_timestamps $DATASET_DIR/timestamps/$scene/img_aux_rear.txt \
    --extrinsic $DATASET_DIR/calibrations/$scene/os_to_cam_aux_rear.yaml \
    --out_pose_file $DATASET_DIR/poses/$scene/cam_aux_rear.txt
done