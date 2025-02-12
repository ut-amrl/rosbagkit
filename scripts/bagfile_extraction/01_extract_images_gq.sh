#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

# input/output paths
BAGFILE_PATH=/robodata/ARL_SARA/GQ-dataset/bagfiles
OUTPUT_PATH=/robodata/dlee/Datasets/IRIS
TIME_RANGE_FILE=$OUTPUT_PATH/extract_time_range.txt

# topics
IMG_LEFT_TOPIC=/trevor/stereo_left/image_rect_color/compressed
IMG_RIGHT_TOPIC=/trevor/stereo_right/image_rect_color/compressed
IMG_FORWARD_TOPIC=/trevor/multisense_forward/aux/image_rect_color
IMG_FORWARD_LEFT_TOPIC=/trevor/multisense_forward/left/image_rect
IMG_FORWARD_RIGHT_TOPIC=/trevor/multisense_forward/right/image_rect
IMG_REAR_TOPIC=/trevor/multisense_rear/aux/image_rect_color
IMG_REAR_LEFT_TOPIC=/trevor/multisense_rear/left/image_rect
IMG_REAR_RIGHT_TOPIC=/trevor/multisense_rear/right/image_rect
DEPTH_FORWARD_TOPIC=/trevor/multisense_forward/depth
DEPTH_REAR_TOPIC=/trevor/multisense_rear/depth

scenes=(
  2024-08-12-11-32-46
  2024-08-12-11-45-44
  2024-08-12-12-00-13
  2024-08-12-12-16-01
  2024-08-12-12-29-20
  2024-08-12-12-39-26
  2024-08-12-17-18-39
  2024-08-12-17-28-12
  2024-08-12-17-41-07
  2024-08-13-11-25-22
  2024-08-13-11-28-57
  2024-08-13-11-34-23
  2024-08-13-11-37-53
  2024-08-13-11-48-00
  2024-08-13-11-59-52
  2024-08-13-12-05-41
  2024-08-13-12-13-51
  2024-08-13-15-28-36
  2024-08-13-15-39-15
  2024-08-13-16-20-38
  2024-08-13-16-27-54
  2024-08-13-16-36-24
  2024-08-13-16-41-44
  2024-08-13-16-45-06
  2024-08-13-16-48-47
  2024-08-13-16-54-12
  2024-08-13-16-56-52
  2024-08-13-17-07-31
  2024-08-13-17-14-03
  2024-08-15-12-40-46
  2024-08-15-12-44-16
  2024-08-15-12-53-53
  2024-08-15-12-56-35
  2024-08-15-13-08-57
  2024-08-15-13-23-03
  2024-08-15-13-27-41
  2024-08-15-13-33-25
  2024-08-15-13-37-11
  2024-08-15-13-43-35
  2024-08-15-13-48-29
  2024-08-15-13-52-10
  2024-08-15-13-59-13
)


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

  img_topics=(
    "$IMG_LEFT_TOPIC"
    "$IMG_RIGHT_TOPIC"
    # "$IMG_FORWARD_TOPIC"
    # "$IMG_FORWARD_LEFT_TOPIC"
    # "$IMG_FORWARD_RIGHT_TOPIC"
    # "$IMG_REAR_TOPIC"
    # "$DEPTH_FORWARD_TOPIC"
  )

  img_outdirs=(
    "$OUTPUT_PATH/2d_raw/$scene/cam_left"
    "$OUTPUT_PATH/2d_raw/$scene/cam_right"
    # "$OUTPUT_PATH/2d_raw/$scene/cam_forward"
    # "$OUTPUT_PATH/2d_raw/$scene/cam_forward_left"
    # "$OUTPUT_PATH/2d_raw/$scene/cam_forward_right"
    # "$OUTPUT_PATH/2d_raw/$scene/cam_rear"
    # "$OUTPUT_PATH/2d_depth/$scene/multisense_forward"
  )

  ts_outfiles=(
    "$OUTPUT_PATH/2d_raw/$scene/timestamp_left.txt"
    "$OUTPUT_PATH/2d_raw/$scene/timestamp_right.txt"
    # "$OUTPUT_PATH/2d_raw/$scene/timestamp_forward.txt"
    # "$OUTPUT_PATH/2d_raw/$scene/timestamp_forward_left.txt"
    # "$OUTPUT_PATH/2d_raw/$scene/timestamp_forward_right.txt"
    # "$OUTPUT_PATH/2d_raw/$scene/timestamp_rear.txt"
    # "$OUTPUT_PATH/2d_depth/$scene/timestamp_multisense_forward.txt"
  )

  prefixs=(
    "2d_raw_left_"
    "2d_raw_right_"
    # "2d_raw_forward_"
    # "2d_raw_forward_left_"
    # "2d_raw_forward_right_"
    # "2d_raw_rear_"
    # "2d_depth_forward_"
  )

  # Extract images
  python "$PROJECT_DIR/scripts/bagfile_extraction/01_extract_images.py" \
    --bagfile "$BAGFILE_PATH/$scene.bag" \
    --img_topics "${img_topics[@]}" \
    --img_outdirs "${img_outdirs[@]}" \
    --ts_outfiles "${ts_outfiles[@]}" \
    --prefixs "${prefixs[@]}" \
    $time_args
done
