#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

sequences=(1)
dataset_dir="/home/dongmyeong/Projects/datasets/CODa"

setup_ws2="/home/dongmyeong/Projects/interactive_slam/devel/setup.bash"

cleanup() {
  echo "Terminating background processes..."
  kill $PID2 $PID3
  wait $PID2 $PID3
  exit 1  # Exit script with a status indicating failure
}

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

for seq in "${sequences[@]}"; do
  # Start odometry_saver
  ( source $setup_ws2 && exec roslaunch odometry_saver online.launch \
    dataset:=coda save_pose_only:=false \
    dst_directory:=${dataset_dir}/gicp_results/${seq}  \
    points_topic:=/ouster_points odom_topic:=/odom \
    endpoint_frame:=os1 origin_frame:=map) &
  PID2=$!

  sleep 3

  # Publish compensated pointcloud and odometry
  python $PROJECT_DIR/py_scripts/publish_data/publish_compensated_data.py \
    --dataset CODa --dataset_dir $dataset_dir --scene $seq &
  wait $!

  echo "Terminating background processes..."
  kill $PID2
done
echo "Finish publishing compensated pointcloud and odometry."