#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

dataset_dir="/home/dongmyeong/Projects/datasets/CODa"
sequences=(0 1 2 3 4 5 6 7 9 10 11 12 13 16 17 18 19 20 21 22)

pc_frame="os1"
pc_topic="/ouster_points"
imu_topic="/imu/data"

# Define the paths to your catkin workspace setup files
setup_ws1="/home/dongmyeong/Projects/others/Point-LIO/devel/setup.bash"
setup_ws2="/home/dongmyeong/Projects/interactive_slam/devel/setup.bash"
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
roscore & PID3=$!
sleep 3

# LiDAR-Inertial Odometry
for seq in "${sequences[@]}"; do
  # Start Point-LIO
  ( source $setup_ws1 && exec roslaunch point_lio mapping_coda.launch rviz:=$rviz --wait ) &
  PID1=$!

  # Start odometry_saver
  ( source $setup_ws2 && exec roslaunch odometry_saver point_lio.launch \
      dataset:=coda \
      save_pose_only:=true \
      pose_file:=$dataset_dir/poses/point-lio/$seq.txt ) &
  PID2=$!

  # Wait for both background processes to start
  sleep 3

  # Execute the third command in the foreground
  python $PROJECT_DIR/py_scripts/publish_data/publish_raw_data.py \
    --dataset CODa --dataset_dir $dataset_dir --scene $seq \
    --pc_frame $pc_frame --pc_topic $pc_topic --imu_topic $imu_topic &
  wait $!

  echo "Rosbag play finished. Terminating background processes..."
  kill $PID1 $PID2
  wait $PID1 $PID2 2>/dev/null
done
echo "All done!"

kill $PID
wait $PID 2>/dev/null