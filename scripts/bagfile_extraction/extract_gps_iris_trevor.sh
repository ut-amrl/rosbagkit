#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

ROBOT=trevor
DATASET_DIR=$PROJECT_DIR/data/IRIS/$ROBOT
TIME_RANGE_FILE=$DATASET_DIR/extract_time_range.txt

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
  2024-08-13-11-34-23
  # 2024-08-13-11-37-53
  # 2024-08-13-11-48-00
  # 2024-08-13-11-59-52
  # 2024-08-13-12-05-41
  # 2024-08-13-12-13-51
  # 2024-08-13-15-28-36
  # 2024-08-13-15-39-15
  # 2024-08-13-16-20-38
  # 2024-08-13-16-27-54
  # 2024-08-13-16-36-24
  # 2024-08-13-16-41-44
  # 2024-08-13-16-45-06
  # 2024-08-13-16-48-47
  # 2024-08-13-16-54-12
  # 2024-08-13-16-56-52
  # 2024-08-13-17-07-31
  # 2024-08-13-17-14-03
)

# topics
gps_topic=/$ROBOT/gps

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
  python $PROJECT_DIR/src/bagfile_extraction/extract_gps.py \
    --bagfile $DATASET_DIR/bagfiles/$scene.bag \
    --gps_topic $gps_topic --gps_outfile $DATASET_DIR/gps/$scene.txt \
    $time_args
done