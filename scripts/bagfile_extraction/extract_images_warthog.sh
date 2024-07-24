#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

ROBOT="trevor"
DATASET_DIR=$PROJECT_DIR/data/SARA/$ROBOT

scenes=(
  2024-07-18-18-40-08
)

img_front_topic=/$ROBOT/multisense_forward/aux/image_rect_color
img_rear_topic=/$ROBOT/multisense_rear/aux/image_rect_color
depth_front_topic=/$ROBOT/multisense_forward/depth
depth_rear_topic=/$ROBOT/multisense_rear/depth

trap "echo 'Script interrupted'; exit;" SIGINT

for scene in "${scenes[@]}" ; do
  img_outdir=$DATASET_DIR/2d_rect/$scene
  depth_outdir=$DATASET_DIR/depth/$scene
  ts_outdir=$DATASET_DIR/timestamps/$scene

  python $PROJECT_DIR/src/bagfile_extraction/extract_images.py \
    --bagfile $DATASET_DIR/bagfiles/$scene.bag \
    --img_topics $img_front_topic $img_rear_topic \
    --img_outdirs $img_outdir/front $img_outdir/rear \
    --ts_outfiles $ts_outdir/img_aux_front.txt $ts_outdir/img_aux_rear.txt \
    --prefixs 2d_rect_aux_front_ 2d_rect_aux_rear_

  # python $PROJECT_DIR/src/bagfile_extraction/extract_images.py \
  #   --bagfile $DATASET_DIR/bagfiles/$scene.bag \
  #   --img_topics $depth_front_topic $depth_rear_topic \
  #   --img_outdirs $depth_outdir/front $depth_outdir/rear \
  #   --ts_outfiles $ts_outdir/depth_front.txt $ts_outdir/depth_rear.txt \
  #   --prefixs depth_front_ depth_rear_
done