#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

ROBOT=wilbur
DATASET_DIR=$PROJECT_DIR/data/IRIS/$ROBOT

scenes=(
  mout-forest-loop-1_2024-04-10-10-29-03
)

img_front_topic=/$ROBOT/multisense_forward/aux/image_rect_color/compressed

trap "echo 'Script interrupted'; exit;" SIGINT

for scene in "${scenes[@]}" ; do
  python $PROJECT_DIR/src/bagfile_extraction/extract_images.py \
    --bagfile $DATASET_DIR/bagfiles/$scene.bag \
    --img_topics $img_front_topic \
    --img_outdirs $DATASET_DIR/2d_rect/cam_aux/$scene \
    --ts_outfiles $DATASET_DIR/timestamps/$scene/cam_aux.txt \
    --prefixs "2d_rect_aux_"
done