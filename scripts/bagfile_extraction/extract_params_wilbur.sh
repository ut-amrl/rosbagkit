#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

# frames
os_frame="wilbur/ouster_center_link"
cam_aux_frame="wilbur/multisense_forward/aux_camera_optical_frame"
imu_frame="wilbur/imu_link"

# topics
cam_aux_info="/wilbur/multisense_forward/aux/image_rect_color/camera_info"

dataset_dir="/home/dongmyeong/Projects/datasets/SARA/wilbur"
scenes=(
  mout-forest-loop-1_2024-04-10-10-29-03
)

trap "echo 'Script interrupted'; exit;" SIGINT

# Check if roscore is already running
if pgrep -x "roscore" > /dev/null; then
    # Kill existing roscore
    pkill -f "roscore"
    sleep 3
fi

# Start roscore
roscore & PID1=$!
sleep 1

for scene in "${scenes[@]}" ; do
    calibrations_dir=$dataset_dir/calibrations/$scene

    python $PROJECT_DIR/src/bagfile_extraction/extract_tf.py \
    --bagfile $dataset_dir/bagfiles/$scene.bag \
    --tf_specs "$os_frame:$cam_aux_frame:$calibrations_dir/os_to_cam_aux.yaml" \
               "$os_frame:$imu_frame:$calibrations_dir/os_to_imu.yaml" \

    python $PROJECT_DIR/src/bagfile_extraction/extract_camera_info.py \
        --bagfile $dataset_dir/bagfiles/$scene.bag \
        --info_topics $cam_aux_info \
        --outfiles $dataset_dir/calibrations/$scene/cam_aux_intrinsics.yaml
done

# Kill roscore
kill $PID1

echo "Finish extracting camera parameters."