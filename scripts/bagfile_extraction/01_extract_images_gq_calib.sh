#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

# input/output paths
BAGFILE_PATH=/robodata/ARL_SARA/GQ-dataset/bagfiles
OUTPUT_PATH=/robodata/dlee/Datasets/IRIS

# topics
IMG_LEFT_TOPIC=/trevor/stereo_left/image_rect_color
IMG_RIGHT_TOPIC=/trevor/stereo_right/image_rect_color
IMG_FORWARD_TOPIC=/trevor/multisense_forward/aux/image_rect_color
IMG_REAR_TOPIC=/trevor/multisense_rear/aux/image_rect_color

scenes=(
  data_collect_gq-trevor-2
)


trap "echo 'Script interrupted'; exit;" SIGINT


# Main loop
for scene in "${scenes[@]}" ; do

  img_topics=(
    "$IMG_LEFT_TOPIC"
    "$IMG_RIGHT_TOPIC"
    "$IMG_FORWARD_TOPIC"
    "$IMG_REAR_TOPIC"
  )

  img_outdirs=(
    "$OUTPUT_PATH/2d_rect/$scene/cam_left"
    "$OUTPUT_PATH/2d_rect/$scene/cam_right"
    "$OUTPUT_PATH/2d_rect/$scene/cam_forward"
    "$OUTPUT_PATH/2d_rect/$scene/cam_rear"
  )

  ts_outfiles=(
    "$OUTPUT_PATH/2d_rect/$scene/times_left.txt"
    "$OUTPUT_PATH/2d_rect/$scene/times_right.txt"
    "$OUTPUT_PATH/2d_rect/$scene/times_forward.txt"
    "$OUTPUT_PATH/2d_rect/$scene/times_rear.txt"
  )

  prefixs=(
    "2d_rect_left_"
    "2d_rect_right_"
    "2d_rect_forward_"
    "2d_rect_rear_"
  )

  # Extract images
  python "$PROJECT_DIR/scripts/bagfile_extraction/01_extract_images.py" \
    --bagfile "$BAGFILE_PATH/$scene.bag" \
    --img_topics "${img_topics[@]}" \
    --img_outdirs "${img_outdirs[@]}" \
    --ts_outfiles "${ts_outfiles[@]}" \
    --prefixs "${prefixs[@]}"
done
