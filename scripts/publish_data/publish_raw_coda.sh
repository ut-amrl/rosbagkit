#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

dataset_dir="/home/dongmyeong/Projects/datasets/CODa"
# sequences=(0 1 2 3 4 5 6 7 9 10 11 12 13 16 17 18 19 20 21 22)
sequences=(0)

pc_frame="os1"
pc_topic="/ouster_points"
imu_topic="/imu/data"

# Function to handle script termination
cleanup() {
  echo "Terminating background processes..."
  kill $PID
  wait $PID
  exit 1  # Exit script with a status indicating failure
}

# Trap SIGINT (Ctrl+C) and call the cleanup function
trap cleanup SIGINT

# Check if roscore is already running
if pgrep -x "roscore" > /dev/null; then
  pkill -f "roscore"
  sleep 3
fi

# Start roscore
roscore & PID=$!
sleep 3

# LiDAR-Inertial Odometry
for seq in "${sequences[@]}"; do
  python $PROJECT_DIR/src/publish_data/publish_raw_data.py \
    --dataset CODa --dataset_dir $dataset_dir --scene $seq \
    --pc_frame $pc_frame --pc_topic $pc_topic --imu_topic $imu_topic &
  wait $!
done
echo "All done!"

kill $PID
wait $PID 2>/dev/null