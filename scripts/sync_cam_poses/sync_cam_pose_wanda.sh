#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

DATASET_DIR=$PROJECT_DIR/data/SARA/wanda
scenes=(
  gq_appld_south_tour_01_2024-03-14-10-08-34
  gq_appld_wandagq_32_field_foresttrail_06_2024-03-15-11-17-44
  gq_appld_wandagq_32_forest_02_2024-03-15-12-02-37
  gq_appld_wandagq_32_forest_03_2024-03-15-12-16-36
  gq_appld_forest_mission_autonomous_deployment_01_2024-03-12-14-15-49
  # gq_TN_Menu_A_datacollect_02_2024-02-21-17-23-09
)

trap "echo 'Script interrupted'; exit;" SIGINT

for scene in "${scenes[@]}" ; do
  python $PROJECT_DIR/src/miscellaneous/sync_cam_pose.py \
    --ref_pose_file $DATASET_DIR/poses/$scene/os1.txt \
    --target_timestamps $DATASET_DIR/timestamps/$scene/img_left.txt \
    --extrinsic $DATASET_DIR/calibrations/$scene/os_to_cam_left.yaml \
    --out_pose_file $DATASET_DIR/poses/$scene/cam_left.txt

  python $PROJECT_DIR/src/miscellaneous/sync_cam_pose.py \
    --ref_pose_file $DATASET_DIR/poses/$scene/os1.txt \
    --target_timestamps $DATASET_DIR/timestamps/$scene/img_right.txt \
    --extrinsic $DATASET_DIR/calibrations/$scene/os_to_cam_right.yaml \
    --out_pose_file $DATASET_DIR/poses/$scene/cam_right.txt
done