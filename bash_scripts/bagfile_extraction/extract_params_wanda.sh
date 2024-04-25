#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

os_frame="wanda/center_ouster_link"
cam_left_frame="wanda/stereo_left_link"
cam_right_frame="wanda/stereo_right_link"
cam_left_info="/wanda/stereo_left/camera_info"
cam_right_info="/wanda/stereo_right/camera_info"

dataset_dir="/home/dongmyeong/Projects/datasets/SARA"
scenes=(
  gq_appld_south_tour_01_2024-03-14-10-08-34
  gq_appld_wandagq_32_field_foresttrail_06_2024-03-15-11-17-44
  gq_appld_wandagq_32_forest_02_2024-03-15-12-02-37
  gq_appld_wandagq_32_forest_03_2024-03-15-12-16-36
  gq_appld_forest_mission_autonomous_deployment_01_2024-03-12-14-15-49
)

trap "echo 'Script interrupted'; exit;" SIGINT

for scene in "${scenes[@]}" ; do
  python $PROJECT_DIR/py_scripts/bagfile_extraction/extract_tf.py \
    --bagfile=$dataset_dir/bagfiles/$scene.bag \
    --source_frame=$os_frame \
    --target_frame=$cam_left_frame \
    --outfile=$dataset_dir/calibrations/$scene/os_to_cam_left.yaml

  python $PROJECT_DIR/py_scripts/bagfile_extraction/extract_tf.py \
    --bagfile=$dataset_dir/bagfiles/$scene.bag \
    --source_frame=$os_frame \
    --target_frame=$cam_right_frame \
    --outfile=$dataset_dir/calibrations/$scene/os_to_cam_right.yaml

  python $PROJECT_DIR/py_scripts/bagfile_extraction/extract_camera_info.py \
    --bagfile=$dataset_dir/bagfiles/$scene.bag \
    --info_topics $cam_left_info $cam_right_info \
    --outfile $dataset_dir/calibrations/$scene/cam_left_intrinsics.yaml \
              $dataset_dir/calibrations/$scene/cam_right_intrinsics.yaml
done