#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

dataset_dir="/home/dongmyeong/Projects/datasets/SARA/wanda"
scenes=(
  gq_appld_wandagq_32_field_foresttrail_06_2024-03-15-11-17-44
  gq_appld_south_tour_01_2024-03-14-10-08-34
  gq_appld_wandagq_32_forest_02_2024-03-15-12-02-37
  gq_appld_wandagq_32_forest_03_2024-03-15-12-16-36
  gq_TN_Menu_A_datacollect_02_2024-02-21-17-23-09
  gq_appld_forest_mission_autonomous_deployment_01_2024-03-12-14-15-49
)

origin_frame="map"
pc_frame="os1"
pc_topic="/ouster_points"
odom_topic="/odom"
blind=0.0

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

for scene in "${scenes[@]}"; do
  # Publish compensated pointcloud and odometry
  python $PROJECT_DIR/src/publish_data/publish_compensated_data.py \
      --dataset Wanda --dataset_dir=$dataset_dir --scene=$scene --blind=$blind \
      --origin_frame=$origin_frame --pc_frame=$pc_frame \
      --pc_topic=$pc_topic --odom_topic=$odom_topic --pub_image &
  wait $!
done
echo "Finish publishing compensated pointcloud and odometry."

# Terminate roscore
kill $PID
wait $PID 2>/dev/null