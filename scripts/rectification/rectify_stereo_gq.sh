#!/bin/bash

PROJECT_DIR=$(realpath $(dirname "$0")/../..)
DATASET_PATH=/robodata/dlee/Datasets/IRIS

scenes=(
  2024-08-12-11-32-46
  2024-08-12-11-45-44
  2024-08-12-12-00-13
  2024-08-12-12-16-01
  2024-08-12-12-29-20
  2024-08-12-12-39-26
  2024-08-12-17-18-39
  2024-08-12-17-28-12
  2024-08-12-17-41-07
  2024-08-13-11-25-22
  2024-08-13-11-28-57
  2024-08-13-11-34-23
  2024-08-13-11-37-53
  2024-08-13-11-48-00
  2024-08-13-11-59-52
  2024-08-13-12-05-41
  2024-08-13-12-13-51
  2024-08-13-15-28-36
  2024-08-13-15-39-15
  2024-08-13-16-20-38
  2024-08-13-16-27-54
  2024-08-13-16-36-24
  2024-08-13-16-41-44
  2024-08-13-16-45-06
  2024-08-13-16-48-47
  2024-08-13-16-54-12
  2024-08-13-16-56-52
  2024-08-13-17-07-31
  2024-08-13-17-14-03
  2024-08-15-12-40-46
  2024-08-15-12-44-16
  2024-08-15-12-53-53
  2024-08-15-12-56-35
  2024-08-15-13-08-57
  2024-08-15-13-23-03
  2024-08-15-13-27-41
  2024-08-15-13-33-25
  2024-08-15-13-37-11
  2024-08-15-13-43-35
  2024-08-15-13-48-29
  2024-08-15-13-52-10
  2024-08-15-13-59-13
)


trap "echo 'Script interrupted'; exit;" SIGINT


for scene in "${scenes[@]}" ; do
  python "$PROJECT_DIR/scripts/rectification/rectify_stereo.py" \
    --image_path $DATASET_PATH/2d_raw/$scene \
    --output_path $DATASET_PATH/2d_rect/$scene \
    --left_calib $DATASET_PATH/calibrations/cam_left_intrinsics.yaml \
    --right_calib $DATASET_PATH/calibrations/cam_right_intrinsics.yaml \
    --left_timestamps $DATASET_PATH/2d_raw/$scene/timestamp_left.txt \
    --right_timestamps $DATASET_PATH/2d_raw/$scene/timestamp_right.txt \
    --extrinsics $DATASET_PATH/calibrations/cam_left_to_right.yaml
done
