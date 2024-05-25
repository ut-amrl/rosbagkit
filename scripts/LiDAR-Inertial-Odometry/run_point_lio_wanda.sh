#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

dataset_dir="/home/dongmyeong/Projects/datasets/SARA/wanda"
scenes=(
  # gq_appld_south_tour_01_2024-03-14-10-08-34
  gq_appld_wandagq_32_field_foresttrail_06_2024-03-15-11-17-44
  # gq_appld_wandagq_32_forest_02_2024-03-15-12-02-37
  # gq_appld_wandagq_32_forest_03_2024-03-15-12-16-36
  # gq_TN_Menu_A_datacollect_02_2024-02-21-17-23-09
  # gq_appld_forest_mission_autonomous_deployment_01_2024-03-12-14-15-49
)

pc_topic="/wanda/lidar_points"
imu_topic="/wanda/imu/data"

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
  ( source $setup_ws1 && exec roslaunch point_lio mapping_wanda.launch rviz:=$rviz --wait ) &
  PID1=$!

  # Start odometry_saver
  ( source $setup_ws2 && exec roslaunch odometry_saver point_lio.launch \
      dataset:=wanda save_pose_only:=true \
      pose_file:=$dataset_dir/poses/$scene/point_lio.txt --wait) &
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