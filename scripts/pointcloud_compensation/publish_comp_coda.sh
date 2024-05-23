#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

dataset_dir="/home/dongmyeong/Projects/datasets/CODa"
sequences=(0 1 2 3 4 5 6 7 9 10 11)

origin_frame="map"
pc_frame="os1"
pc_topic="/ouster_points"
odom_topic="/odom"
blind=3.0

setup_ws1="/home/dongmyeong/Projects/interactive_slam/devel/setup.bash"

cleanup() {
  echo "Terminating background processes..."
  kill $PID1 $PID
  wait $PID1 $PID
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
roscore & PID=$!
sleep 3

for seq in "${sequences[@]}"; do
  # Start odometry_saver
  ( source $setup_ws1 && exec roslaunch odometry_saver online.launch \
    dataset:=coda save_pose_only:=false \
    dst_directory:=$dataset_dir/point-lio_results/$seq  \
    points_topic:=$pc_topic odom_topic:=$odom_topic \
    endpoint_frame:=$pc_frame origin_frame:=$origin_frame) &
  PID1=$!

  sleep 3

  # Publish compensated pointcloud and odometry
  python $PROJECT_DIR/src/publish_data/publish_compensated_data.py \
    --dataset CODa --dataset_dir=$dataset_dir --scene=$seq --blind=$blind \
    --origin_frame=$origin_frame --pc_frame=$pc_frame \
    --pc_topic=$pc_topic --odom_topic=$odom_topic &
  wait $!

  echo "Terminating background processes..."
  kill $PID1
  wait $PID1 2>/dev/null
done
echo "Finish publishing compensated pointcloud and odometry."

# Terminate roscore
kill $PID
wait $PID 2>/dev/null