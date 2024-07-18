#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

IMG_TOPIC="/wilbur/multisense_forward/aux/image_rect_color/compressed"

dataset_dir=$PROJECT_DIR/data/SARA/wilbur
scenes=(
  mout-forest-loop-1_2024-04-10-10-29-03
)

trap "echo 'Script interrupted'; exit;" SIGINT

for scene in "${scenes[@]}" ; do
  python $PROJECT_DIR/src/bagfile_extraction/extract_images.py \
    --bagfile $dataset_dir/bagfiles/$scene.bag \
    --img_left_topic $IMG_TOPIC \
    --img_left_outdir $dataset_dir/2d_rect/$scene \
    --ts_left_file $dataset_dir/timestamps/$scene/img_aux.txt \
    --prefix_left "2d_rect_aux_"
done