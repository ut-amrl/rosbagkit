#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

ROBOT=trevor
DATASET_DIR=$PROJECT_DIR/data/IRIS/$ROBOT

scenes=(
  2024-08-12-11-32-46
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
  # 2024-08-13-16-36-24
  # 2024-08-13-16-41-44
  # 2024-08-13-16-45-06
  # 2024-08-13-16-48-47
  # 2024-08-13-16-54-12
  # 2024-08-13-16-56-52
  # 2024-08-13-17-07-31
  # 2024-08-13-17-14-03
)

pc_topic=/$ROBOT/lidar_points_center
imu_topic=/$ROBOT/imu/data

# Function to handle script termination
cleanup() {
  echo "Terminating background processes..."
  kill $PID1 $PID2 $PID
  wait $PID1 $PID2 $PID
  exit 1  # Exit script with a status indicating failure
}

# Trap SIGINT (Ctrl+C) and call the cleanup function
trap cleanup SIGINT

# Check if roscore is already running
if pgrep -x "roscore" > /dev/null; then
  # Kill existing roscore
  pkill -f "roscore"
  sleep 3
fi

# Start roscore
roscore & PID=$!
sleep 3

for scene in "${scenes[@]}"; do
  # Start FAST-LIO
  ( exec roslaunch fast_lio mapping_${ROBOT}.launch --wait ) &
  PID1=$!

  # Start odometry_saver
  ( exec roslaunch odometry_saver fast_lio.launch \
      dataset:=$ROBOT \
      save_pose_only:=false \
      pose_file:=$DATASET_DIR/poses/$scene/fast_lio.txt \
      dst_directory:=$DATASET_DIR/interactive/odom/$scene ) &
  PID2=$!

  # Wait for both background processes to start
  sleep 3

  # Start rosbag play
  rosbag play $DATASET_DIR/bagfiles/$scene.bag --clock --topic $pc_topic $imu_topic &
  wait $!

  echo "Rosbag play finished. Press 'Y' to terminate background processes..."
  while true; do
    read -n 1 -s key
    if [[ $key == "Y" || $key == "y" ]]; then
      break
    fi
  done

  kill $PID1 $PID2
  wait $PID1 $PID2 2>/dev/null
done
echo "All done!"

# Terminate roscore
kill $PID
wait $PID 2>/dev/null
