#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

IMG_LEFT_TOPIC="/wanda/stereo_left/image_rect_color/compressed"
IMG_RIGHT_TOPIC="/wanda/stereo_right/image_rect_color/compressed"

dataset_dir="/home/dongmyeong/Projects/datasets/SARA"
scenes=(
  gq_appld_south_tour_01_2024-03-14-10-08-34
  gq_appld_wandagq_32_field_foresttrail_06_2024-03-15-11-17-44
  gq_appld_wandagq_32_forest_02_2024-03-15-12-02-37
  gq_appld_wandagq_32_forest_03_2024-03-15-12-16-36
  gq_TN_Menu_A_datacollect_02_2024-02-21-17-23-09
  gq_appld_forest_mission_autonomous_deployment_01_2024-03-12-14-15-49
)

trap "echo 'Script interrupted'; exit;" SIGINT

for scene in "${scenes[@]}" ; do
    python $PROJECT_DIR/py_scripts/bagfile_extraction/extract_images.py \
        --bagfile=${dataset_dir}/bagfiles/${scene}.bag \
        --img_left_topic=${IMG_LEFT_TOPIC} \
        --img_right_topic=${IMG_RIGHT_TOPIC} \
        --img_outdir=${dataset_dir}/2d_rect/${scene} \
        --ts_outdir=${dataset_dir}/timestamps/${scene} \
        --sync
done