#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

IMG_AUX_TOPIC="/crl_rzr/multisense_back/aux/image_color/compressed"
DISPARITY_LEFT_TOPIC="/crl_rzr/multisense_back/left/disparity"

dataset_dir="/home/dongmyeong/Projects/datasets/SARA/crl_rzr"
scenes=(
  gq_TN_e3-baseline_rfv_250_remission_01_2024-02-09-14-39-36
)

trap "echo 'Script interrupted'; exit;" SIGINT

for scene in "${scenes[@]}" ; do
  python $PROJECT_DIR/py_scripts/bagfile_extraction/extract_images.py \
    --bagfile=${dataset_dir}/bagfiles/${scene}.bag \
    --img_left_topic=${IMG_AUX_TOPIC} \
    --img_left_outdir=${dataset_dir}/2d_raw/${scene}/aux \
    --ts_left_file=${dataset_dir}/timestamps/${scene}/img_aux.txt \
    --prefix_left="2d_raw_aux_"

  python $PROJECT_DIR/py_scripts/bagfile_extraction/extract_images.py \
    --bagfile=${dataset_dir}/bagfiles/${scene}.bag \
    --img_left_topic=${DISPARITY_LEFT_TOPIC} \
    --img_left_outdir=${dataset_dir}/disparity/${scene}/left \
    --ts_left_file=${dataset_dir}/timestamps/${scene}/disparity_left.txt \
    --prefix_left="disparity_left_"

done