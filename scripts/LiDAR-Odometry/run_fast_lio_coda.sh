#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

dataset_dir="/home/dongmyeong/Projects/datasets/CODa"
sequences=(0)
#sequences=(0 1 2 3 4 5 6 7 9 10 11 12 13 16 17 18 19 20 21 22)

# Define the paths to your catkin workspace setup files
setup_ws1="/home/dongmyeong/Projects/others/fast-lio/devel/setup.bash"
setup_ws2="/home/dongmyeong/Projects/others/interactive_slam/devel/setup.bash"

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
  # Start the first command in the background with its workspace sourced
  ( source $setup_ws1 && exec roslaunch fast_lio mapping_coda.launch --wait ) &
  PID1=$!

  # Start the second command in the background with its workspace sourced
  ( source $setup_ws2 && exec roslaunch odometry_saver fast_lio.launch \
      dataset:=coda \
      save_pose_only:=false \
      pose_file:=$dataset_dir/poses/fast_lio/$seq.txt \
      dst_directory:=$dataset_dir/fast-lio_results/$seq --wait ) &
  PID2=$!

  # Wait for both background processes to start
  sleep 3

  # Execute the third command in the foreground
  python $PROJECT_DIR/src/publish_data/publish_raw_data.py \
    --dataset CODa --dataset_dir $dataset_dir --scene $seq &
  wait $!

  echo "Rosbag play finished. Terminating background processes..."
  kill $PID1 $PID2
  wait $PID1 $PID2 2>/dev/null

  # echo "Converting odometry format..."
  # python $PROJECT_DIR/src/interactive_slam/convert_odom_format.py \
  #   --odom_dir $dataset_dir/fast_lio_results/$seq \
  #   --pc_outdir $dataset_dir/3d_comp/$seq \
  #   --pose_outfile $dataset_dir/poses/os1/$seq.txt \
  #   --prefix 3d_comp_os1_${seq}_
done
echo "All done!"

kill $PID
wait $PID 2>/dev/null