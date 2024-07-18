#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

DATASET_DIR=$PROJECT_DIR/data/CODa
sequences=(10)
#sequences=(0 1 2 3 4 5 6 7 9 10 11 12 13 16 17 18 19 20 21 22)

pc_frame="os1"
pc_topic="/ouster_points"
imu_topic="/imu/data"

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

for seq in "${sequences[@]}"; do
  # Start FAST-LIO
  ( exec roslaunch fast_lio mapping_coda.launch --wait ) &
  PID1=$!

  # Start odometry_saver
  ( exec roslaunch odometry_saver fast_lio.launch \
      dataset:=coda \
      save_pose_only:=false \
      pose_file:=$DATASET_DIR/poses/fast_lio/$seq.txt \
      dst_directory:=$DATASET_DIR/fast-lio_results/$seq --wait ) &
  PID2=$!

  # Wait for both background processes to start
  sleep 3

  # Execute the third command in the foreground
  python $PROJECT_DIR/src/publish_data/publish_raw_data.py \
    --dataset CODa --dataset_dir $DATASET_DIR --scene $seq \
    --pc_frame $pc_frame --pc_topic $pc_topic --imu_topic $imu_topic &
  wait $!

  echo "Rosbag play finished. Terminating background processes..."
  kill $PID1 $PID2
  wait $PID1 $PID2 2>/dev/null

  # echo "Converting odometry format..."
  # python $PROJECT_DIR/src/interactive_slam/convert_odom_format.py \
  #   --odom_dir $DATASET_DIR/fast_lio_results/$seq \
  #   --pc_outdir $DATASET_DIR/3d_comp/$seq \
  #   --pose_outfile $DATASET_DIR/poses/os1/$seq.txt \
  #   --prefix 3d_comp_os1_${seq}_
done
echo "All done!"

kill $PID
wait $PID 2>/dev/null