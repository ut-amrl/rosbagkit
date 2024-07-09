#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

dataset_dir="/home/dongmyeong/Projects/datasets/SARA/wilbur"
scenes=(
  mout-forest-loop-1_2024-04-10-10-29-03
)

pc_topic="/wilbur/lidar_points_center"
imu_topic="/wilbur/imu/data"

# Define the paths to your catkin workspace setup files
setup_ws1="/home/dongmyeong/Projects/others/fast-lio/devel/setup.bash"
setup_ws2="/home/dongmyeong/Projects/others/interactive_slam/devel/setup.bash"

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
  ( source $setup_ws1 && exec roslaunch fast_lio mapping_wilbur.launch --wait ) &
  PID1=$!

  # Start odometry_saver
  ( source $setup_ws2 && exec roslaunch odometry_saver fast_lio.launch \
      dataset:=wilbur \
      save_pose_only:=true \
      pose_file:=$dataset_dir/poses/$scene/fast_lio.txt \
      dst_directory:=$dataset_dir/fast-lio_results/$scene ) &
  PID2=$!

  # Wait for both background processes to start
  sleep 3

  # Start rosbag play
  rosbag play $dataset_dir/bagfiles/$scene.bag --clock --topic $pc_topic $imu_topic &
  wait $!

  echo "Rosbag play finished. Terminating background processes..."
  kill $PID1 $PID2
  wait $PID1 $PID2 2>/dev/null
done
echo "All done!"

# Terminate roscore
kill $PID
wait $PID 2>/dev/null
