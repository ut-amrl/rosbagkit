#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

DATASET_DIR=$PROJECT_DIR/data/SARA/wilbur
scenes=(
  mout-forest-loop-1_2024-04-10-10-29-03
)

trap "echo 'Script interrupted'; exit;" SIGINT

for scene in "${scenes[@]}" ; do
  python $PROJECT_DIR/src/miscellaneous/sync_cam_pose.py \
    --ref_pose_file $DATASET_DIR/poses/$scene/os1.txt \
    --target_timestamps $DATASET_DIR/timestamps/$scene/img_aux.txt \
    --extrinsic $DATASET_DIR/calibrations/$scene/os_to_cam_aux.yaml \
    --out_pose_file $DATASET_DIR/poses/$scene/cam_aux.txt
done