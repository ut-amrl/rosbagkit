#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

ROBOT="trevor"
DATASET_DIR=$PROJECT_DIR/data/IRIS/$ROBOT
scenes=(
  2024-08-12-11-32-46
  2024-08-12-11-45-44
  2024-08-12-12-00-13
  2024-08-12-12-16-01
  2024-08-12-12-29-20
  2024-08-12-12-39-26
  2024-08-12-17-16-42
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
)

origin_frame="map"
pc_frame="os1"
pc_topic="/compensated_points"
odom_topic="/odom"
blind=3.0

cleanup() {
  echo "Terminating background processes..."
  kill $PID1 $PID
  wait $PID1 $PID
  exit 1  # Exit script with a status indicating failure
}

trap cleanup SIGINT

# # Check if roscore is already running
# if pgrep -x "roscore" > /dev/null; then
#   # Kill existing roscore
#   pkill -f "roscore"
#   sleep 3
# fi

# # Start roscore
# roscore & PID=$!
# sleep 3

for scene in "${scenes[@]}"; do
  # Start odometry_saver
  ( exec roslaunch odometry_saver online.launch \
    save_pose_only:=false \
    dst_directory:=$DATASET_DIR/interactive_slam/$scene \
    points_topic:=$pc_topic odom_topic:=$odom_topic \
    endpoint_frame:=$pc_frame origin_frame:=$origin_frame ) &
  PID1=$!

  sleep 3

  # Publish compensated pointcloud and odometry
  python $PROJECT_DIR/src/publish_data/publish_compensated_data.py \
    --pc_dir $DATASET_DIR/3d_comp/$scene \
    --pose_file $DATASET_DIR/poses/$scene/os1.txt \
    --pc_size 3 --blind $blind \
    --origin_frame $origin_frame --pc_frame $pc_frame \
    --pc_topic $pc_topic --odom_topic $odom_topic -r 10 &
  wait $!

  echo "Terminating background processes..."
  kill $PID1
  wait $PID1 2>/dev/null
done
echo "Finish publishing compensated pointcloud and odometry."

# Terminate roscore
kill $PID
wait $PID 2>/dev/null