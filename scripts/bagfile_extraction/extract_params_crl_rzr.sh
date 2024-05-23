#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

# frames
velodyne_frame="crl_rzr/velodyne_front_horiz_link"
cam_aux_frame="crl_rzr/multisense_front/aux_camera_optical_frame"
cam_left_frame="crl_rzr/multisense_front/left_camera_optical_frame"
cam_right_frame="crl_rzr/multisense_front/right_camera_optical_frame"
imu_frame="crl_rzr/base"

# camera info topics
cam_aux_info="/crl_rzr/multisense_front/aux/camera_info"
cam_left_info="/crl_rzr/multisense_front/left/camera_info"
cam_right_info="/crl_rzr/multisense_front/right/camera_info"

dataset_dir="/home/dongmyeong/Projects/datasets/SARA/crl_rzr"
scenes=(
    gq_TN_e3-baseline_rfv_250_remission_01_2024-02-09-14-39-36
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
    --bagfile=$dataset_dir/bagfiles/$scene.bag \
    --tf_specs "$velodyne_frame:$cam_aux_frame:$calibrations_dir/velodyne_to_cam_aux.yaml" \
               "$velodyne_frame:$cam_left_frame:$calibrations_dir/velodyne_to_cam_left.yaml" \
               "$velodyne_frame:$imu_frame:$calibrations_dir/velodyne_to_imu.yaml" \
               "$cam_left_frame:$cam_aux_frame:$calibrations_dir/cam_left_to_cam_aux.yaml"

    python $PROJECT_DIR/src/bagfile_extraction/extract_camera_info.py \
        --bagfile=$dataset_dir/bagfiles/$scene.bag \
        --info_topics $cam_aux_info $cam_left_info $cam_right_info \
        --outfile $dataset_dir/calibrations/$scene/cam_aux_intrinsics.yaml \
                  $dataset_dir/calibrations/$scene/cam_left_intrinsics.yaml \
                  $dataset_dir/calibrations/$scene/cam_right_intrinsics.yaml
done

# Kill roscore
kill $PID1

echo "Finish extracting camera parameters."