#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

BAGFILE_PATH=/robodata/ARL_SARA/GQ-dataset/bagfiles
OUTPUT_PATH=/robodata/dlee/Datasets/IRIS
TIME_RANGE_FILE=$OUTPUT_PATH/extract_time_range.txt

scenes=(
  # 2024-08-12-11-32-46
  # 2024-08-12-11-45-44
  # 2024-08-12-12-00-13
  # 2024-08-12-12-16-01
  # 2024-08-12-12-29-20
  # 2024-08-12-12-39-26
  # 2024-08-12-17-18-39
  # 2024-08-12-17-28-12
  # 2024-08-12-17-41-07
  # 2024-08-13-11-25-22
  # 2024-08-13-11-28-57
  # 2024-08-13-11-34-23
  # 2024-08-13-11-37-53
  # 2024-08-13-11-48-00
  # 2024-08-13-11-59-52
  # 2024-08-13-12-05-41
  # 2024-08-13-12-13-51
  # 2024-08-13-15-28-36
  # 2024-08-13-15-39-15
  # 2024-08-13-16-20-38
  # 2024-08-13-16-27-54
  2024-08-13-16-36-24
  # 2024-08-13-16-41-44
  # 2024-08-13-16-45-06
  # 2024-08-13-16-48-47
  # 2024-08-13-16-54-12
  2024-08-13-16-56-52
  # 2024-08-13-17-07-31
  # 2024-08-13-17-14-03
  # 2024-08-15-12-40-46
  # 2024-08-15-12-44-16
  # 2024-08-15-12-53-53
  # 2024-08-15-12-56-35
  # 2024-08-15-13-08-57
  # 2024-08-15-13-23-03
  # 2024-08-15-13-27-41
  # 2024-08-15-13-33-25
  # 2024-08-15-13-37-11
  # 2024-08-15-13-43-35
  # 2024-08-15-13-48-29
  # 2024-08-15-13-52-10
  # 2024-08-15-13-59-13
)

# topics
img_left_topic=/trevor/stereo_left/image_rect_color/compressed
img_right_topic=/trevor/stereo_right/image_rect_color/compressed
img_front_topic=/trevor/multisense_forward/aux/image_rect_color
img_rear_topic=/trevor/multisense_rear/aux/image_rect_color

get_time_args() {
  local scene=$1
  # Check if the time range file is readable
  if [[ ! -r "$TIME_RANGE_FILE" ]]; then
    echo "Error: Cannot read the time range file '$TIME_RANGE_FILE'." >&2
    exit 1
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    # Split the line into an array
    read -ra parts <<< "$line"
    # Check if the first part (scene ID) matches
    if [[ "${parts[0]}" == "$scene" ]]; then
      # Check if there are time range parts
      if [[ "${#parts[@]}" -eq 3 ]]; then
        echo "--start_time ${parts[1]} --end_time ${parts[2]}"
      else
        echo ""  # Return empty string for scenes without a specific time range
      fi
      return 0
    fi
  done < "$TIME_RANGE_FILE"

  # If not found, return empty string
  echo ""
}


trap "echo 'Script interrupted'; exit;" SIGINT

# Main loop
for scene in "${scenes[@]}" ; do
  # Get the time range arguments
  time_args=$(get_time_args "$scene")
  if [[ -z "$time_args" ]]; then
    echo "Info: No specific time range found for $scene. Processing entire scene."
  else
    echo "Processing $scene with time range: $time_args"
  fi

  # Extract images
  python $PROJECT_DIR/scripts/bagfile_extraction/extract_images.py \
    --bagfile $BAGFILE_PATH/$scene.bag \
    --img_topics $img_right_topic \
    --img_outdirs $OUTPUT_PATH/2d_rect/cam_right/$scene \
    --ts_outfiles $OUTPUT_PATH/timestamps/$scene/cam_right.txt \
    --prefixs 2d_rect_right_ \
    $time_args

  python $PROJECT_DIR/scripts/bagfile_extraction/extract_images.py \
    --bagfile $BAGFILE_PATH/$scene.bag \
    --img_topics $img_front_topic \
    --img_outdirs $OUTPUT_PATH/2d_rect/cam_front/$scene \
    --ts_outfiles $OUTPUT_PATH/timestamps/$scene/cam_front.txt \
    --prefixs 2d_rect_front_ \
    $time_args
done
