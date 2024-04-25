#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

#sequences=(0 1 2 3 4 5 6 7 9 10 11 12 13 16 17 18 19 20 21 22)
sequences=(1)
dataset_dir="/home/dongmyeong/Projects/datasets/CODa"

# Define the paths to your catkin workspace setup files
setup_ws1="/home/dongmyeong/Projects/others/Point-LIO/devel/setup.bash"
setup_ws2="/home/dongmyeong/Projects/interactive_slam/devel/setup.bash"

# Function to handle script termination
cleanup() {
  echo "Terminating background processes..."
  kill $PID1 $PID2
  wait $PID1 $PID2
  exit 1  # Exit script with a status indicating failure
}

# Trap SIGINT (Ctrl+C) and call the cleanup function
trap cleanup SIGINT

for seq in "${sequences[@]}"; do
  # Start Point-LIO
  ( source $setup_ws1 && exec roslaunch point_lio mapping_coda.launch --wait) &
  PID1=$!

  # Start odometry_saver
  ( source $setup_ws2 && exec roslaunch odometry_saver point_lio.launch \
      dataset:=coda dst_directory:=${dataset_dir}/point_lio_results/${seq} ) &
  PID2=$!

  # Wait for both background processes to start
  sleep 3

  # Execute the third command in the foreground
  python $PROJECT_DIR/py_scripts/publish_raw_coda.py --pc --imu --rate 10 --seq ${seq} &
  wait $!

  echo "Rosbag play finished. Terminating background processes..."
  kill $PID1 $PID2
  wait $PID1 $PID2

  # echo "Converting odometry format..."
  # python $PROJECT_DIR/py_scripts/interactive_slam/convert_odom_format.py \
  #   --odom_dir $dataset_dir/point_lio_results/$seq \
  #   --pc_outdir $dataset_dir/3d_comp/$seq \
  #   --pose_outfile $dataset_dir/poses/os1/$seq.txt
done
echo "All done!"