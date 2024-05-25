#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

dataset_dir="/home/dongmyeong/Projects/datasets/SARA/crl_rzr"
scenes=(
  gq_TN_e3-baseline_rfv_250_remission_01_2024-02-09-14-39-36
)

pc_topic="/crl_rzr/velodyne_front_horiz_points"
imu_topic="/crl_rzr/imu/data"

# Define the paths to your catkin workspace setup files
setup_ws1="/home/dongmyeong/Projects/others/Point-LIO/devel/setup.bash"
setup_ws2="/home/dongmyeong/Projects/others/interactive_slam/devel/setup.bash"
rviz=true

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

# LiDAR-Inertial Odometry
for scene in "${scenes[@]}"; do
  # Start Point-LIO
  ( source $setup_ws1 && exec roslaunch point_lio mapping_crl_rzr.launch rviz:=$rviz --wait ) &
  PID1=$!

  # Start odometry_saver
  ( source $setup_ws2 && exec roslaunch odometry_saver point_lio.launch \
      dataset:=crl_rzr \
      save_pose_only:=true \
      pose_file:=$dataset_dir/poses/$scene/point_lio.txt ) &
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

kill $PID
wait $PID 2>/dev/null