#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

dataset_dir="/home/dongmyeong/Projects/datasets/CODa"
sequences=(1 2 3 4 5 6 7 9 10 11 12 13 16 17 18 19 20 21 22)

pc_frame="os1"
pc_topic="/ouster_points"
imu_topic="/imu/data"

origin_frame="map"
odom_topic="/odom"
blind=3.0

# Define the paths to your catkin workspace setup files
setup_ws1="/home/dongmyeong/Projects/others/Point-LIO/devel/setup.bash"
setup_ws2="/home/dongmyeong/Projects/others/interactive_slam/devel/setup.bash"
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
  pkill -f "roscore"
  sleep 3
fi

# Start roscore
roscore & PID=$!
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
      pose_file:=$dataset_dir/poses/point-lio/$seq.txt --wait ) &
  PID2=$!

  # Wait for both background processes to start
  sleep 3

  # Execute the third command in the foreground
  python $PROJECT_DIR/src/publish_data/publish_raw_data.py \
    --dataset CODa --dataset_dir $dataset_dir --scene $seq \
    --pc_frame $pc_frame --pc_topic $pc_topic --imu_topic $imu_topic &
  wait $!

  echo "Rosbag play finished. Terminating background processes..."
  kill $PID1 $PID2
  wait $PID1 $PID2 2>/dev/null

  ##################################################
  # Post-process the odometry results
  ##################################################
  python $PROJECT_DIR/src/pointcloud_compensation/compensate_pointcloud_coda.py \
    --dataset_dir $dataset_dir --seq $seq \
    --ref_posefile $dataset_dir/poses/point-lio/$seq.txt \
    --out_posefile $dataset_dir/poses/point-lio_sync/$seq.txt &

  # Start odometry_saver
  ( source $setup_ws2 && exec roslaunch odometry_saver online.launch \
    dst_directory:=$dataset_dir/interactive_slam/$seq  \
    points_topic:=$pc_topic odom_topic:=$odom_topic \
    endpoint_frame:=$pc_frame origin_frame:=$origin_frame) &
  PID3=$!

  # Publish compensated pointcloud and odometry
  python $PROJECT_DIR/src/publish_data/publish_compensated_data.py \
    --dataset CODa --dataset_dir $dataset_dir --scene $seq --blind $blind \
    --origin_frame $origin_frame --pc_frame $pc_frame \
    --pc_topic $pc_topic --odom_topic $odom_topic -r 10 &
  wait $!

  kill $PID3
  wait $PID3 2>/dev/null
done
echo "All done!"

kill $PID
wait $PID 2>/dev/null