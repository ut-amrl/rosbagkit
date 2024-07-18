#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

DATASET_DIR=$PROJECT_DIR/data/SARA/wanda
scenes=(
  gq_appld_south_tour_01_2024-03-14-10-08-34
  gq_appld_wandagq_32_field_foresttrail_06_2024-03-15-11-17-44
  gq_appld_wandagq_32_forest_02_2024-03-15-12-02-37
  gq_appld_wandagq_32_forest_03_2024-03-15-12-16-36
  gq_appld_forest_mission_autonomous_deployment_01_2024-03-12-14-15-49
  gq_TN_Menu_A_datacollect_02_2024-02-21-17-23-09
)

pc_topic="/wanda/lidar_points"
imu_topic="/wanda/imu/data"

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
  ( source $setup_ws1 && exec roslaunch fast_lio mapping_wanda.launch --wait ) &
  PID1=$!

  # Start odometry_saver
  ( source $setup_ws2 && exec roslaunch odometry_saver fast_lio.launch \
      dataset:=wanda \
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

  # echo "Converting odometry format..."
  # python $PROJECT_DIR/src/interactive_slam/convert_odom_format.py \
  #   --odom_dir $DATASET_DIR/fast_lio_results/$scene \
  #   --pc_outdir $DATASET_DIR/3d_comp/$scene \
  #   --pose_outfile $DATASET_DIR/poses/$scene/os1.txt \
  #   --prefix 3d_comp_os1_
done
echo "All done!"

# Terminate roscore
kill $PID
wait $PID 2>/dev/null
