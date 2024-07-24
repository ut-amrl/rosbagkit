#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

ROBOT="trevor"
DATASET_DIR=$PROJECT_DIR/data/SARA/$ROBOT

scenes=(
  2024-07-18-18-40-08
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
      save_pose_only:=true \
      pose_file:=$DATASET_DIR/poses/$scene/fast_lio.txt \
      dst_directory:=$DATASET_DIR/fast-lio_results/$scene ) &
  PID2=$!

  # Wait for both background processes to start
  sleep 3

  # Start rosbag play
  rosbag play $DATASET_DIR/bagfiles/$scene.bag --clock --topic $pc_topic $imu_topic &
  wait $!

  echo "Rosbag play finished. Terminating background processes..."
  kill $PID1 $PID2
  wait $PID1 $PID2 2>/dev/null
done
echo "All done!"

# Terminate roscore
kill $PID
wait $PID 2>/dev/null
