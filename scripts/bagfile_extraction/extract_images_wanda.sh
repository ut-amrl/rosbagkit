#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

IMG_LEFT_TOPIC="/wanda/stereo_left/image_rect_color/compressed"
IMG_RIGHT_TOPIC="/wanda/stereo_right/image_rect_color/compressed"

DATASET_DIR=$PROJECT_DIR/data/SARA/wanda
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
  python $PROJECT_DIR/src/bagfile_extraction/extract_images.py \
    --bagfile $DATASET_DIR/bagfiles/${scene}.bag \
    --img_left_topic $IMG_LEFT_TOPIC \
    --img_left_outdir $DATASET_DIR/2d_rect/$scene/left \
    --ts_left_file $DATASET_DIR/timestamps/$scene/img_left.txt \
    --prefix_left "2d_rect_left_" \
    --img_right_topic $IMG_RIGHT_TOPIC \
    --img_right_outdir $DATASET_DIR/2d_rect/$scene/right \
    --ts_right_file $DATASET_DIR/timestamps/$scene/img_right.txt \
    --prefix_right "2d_rect_right_" \
    --sync
done