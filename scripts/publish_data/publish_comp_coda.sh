#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

DATASET_DIR=$PROJECT_DIR/data/CODa
sequences=(7 9 10 11 12 13 16 17)

origin_frame="map"
pc_frame="os1"
pc_topic="/compensated_points"
odom_topic="/odom"
blind=3.0
pc_size=3

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
  ( exec roslaunch odometry_saver online.launch \
    dataset:=coda \
    save_pose_only:=false \
    dst_directory:=$DATASET_DIR/interactive_slam/$seq  \
    points_topic:=$pc_topic odom_topic:=$odom_topic \
    endpoint_frame:=$pc_frame origin_frame:=$origin_frame) &
  PID1=$!

  sleep 3

  # Publish compensated pointcloud and odometry
  python $PROJECT_DIR/src/publish_data/publish_compensated_data.py \
    --pc_dir $DATASET_DIR/3d_comp/os1/$seq \
    --pose_file $DATASET_DIR/poses/fast_lio_sync/$seq.txt \
    --pc_size $pc_size --blind $blind \
    --origin_frame $origin_frame --pc_frame $pc_frame \
    --pc_topic $pc_topic --odom_topic $odom_topic -r 5 &
  wait $!

  echo "Terminating background processes..."
  kill $PID1
  wait $PID1 2>/dev/null
done
echo "Finish publishing compensated pointcloud and odometry."

# Terminate roscore
kill $PID
wait $PID 2>/dev/null