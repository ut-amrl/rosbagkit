#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

ROBOT=trevor
DATASET_DIR=$PROJECT_DIR/data/IRIS/$ROBOT
scenes=(
  2024-08-12-11-32-46
  2024-08-12-11-45-44
  2024-08-12-12-00-13
  2024-08-12-12-16-01
  2024-08-12-12-29-20
  2024-08-12-12-39-26
  2024-08-12-17-18-39
  2024-08-12-17-28-12
  2024-08-12-17-41-07
  2024-08-13-11-25-22
  2024-08-13-11-28-57
  2024-08-13-11-34-23
  2024-08-13-11-37-53
  2024-08-13-11-48-00
  2024-08-13-11-59-52
  2024-08-13-12-05-41
  2024-08-13-12-13-51
  2024-08-13-15-28-36
  2024-08-13-15-39-15
  2024-08-13-16-20-38
  2024-08-13-16-27-54
  2024-08-13-16-36-24
  2024-08-13-16-41-44
  2024-08-13-16-45-06
  2024-08-13-16-48-47
  2024-08-13-16-54-12
  2024-08-13-16-56-52
  2024-08-13-17-07-31
  2024-08-13-17-14-03
)

# frames
os_frame=$ROBOT/ouster_center_link
cam_aux_front_frame=$ROBOT/multisense_forward/aux_camera_optical_frame
cam_aux_rear_frame=$ROBOT/multisense_rear/aux_camera_optical_frame
cam_left_frame=$ROBOT/stereo_left_optical_frame
cam_right_frame=$ROBOT/stereo_right_optical_frame
imu_frame=$ROBOT/imu_link

# topics
info_topics=(
  /$ROBOT/multisense_forward/aux/camera_info
  /$ROBOT/multisense_rear/aux/camera_info
  /$ROBOT/stereo_left/camera_info
  /$ROBOT/stereo_right/camera_info
)
outfiles=(
  "cam_aux_front_intrinsics.yaml"
  "cam_aux_rear_intrinsics.yaml"
  "cam_left_intrinsics.yaml"
  "cam_right_intrinsics.yaml"
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
    calibrations_dir=$DATASET_DIR/calibrations/$scene
    full_outfiles=()
    for outfile in "${outfiles[@]}" ; do
        full_outfiles+=("$calibrations_dir/$outfile")
    done

    python $PROJECT_DIR/src/bagfile_extraction/extract_tf.py \
    --bagfile $DATASET_DIR/bagfiles/$scene.bag \
    --tf_specs "$os_frame:$cam_aux_front_frame:$calibrations_dir/os_to_cam_aux_front.yaml" \
               "$os_frame:$cam_aux_rear_frame:$calibrations_dir/os_to_cam_aux_rear.yaml" \
               "$os_frame:$cam_left_frame:$calibrations_dir/os_to_cam_left.yaml" \
               "$os_frame:$cam_right_frame:$calibrations_dir/os_to_cam_right.yaml" \
               "$os_frame:$imu_frame:$calibrations_dir/os_to_imu.yaml" \

    python $PROJECT_DIR/src/bagfile_extraction/extract_camera_info.py \
        --bagfile $DATASET_DIR/bagfiles/$scene.bag \
        --info_topics ${info_topics[@]} \
        --outfiles ${full_outfiles[@]}
done

# Kill roscore
kill $PID1

echo "Finish extracting camera parameters."