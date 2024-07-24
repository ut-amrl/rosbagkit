#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

ROBOT="trevor"
DATASET_DIR=$PROJECT_DIR/data/SARA/$ROBOT
scenes=(
  2024-07-18-18-40-08
)

# frames
os_frame=$ROBOT/ouster_center_link
cam_aux_front_frame=$ROBOT/multisense_forward/aux_camera_optical_frame
cam_aux_rear_frame=$ROBOT/multisense_rear/aux_camera_optical_frame
imu_frame=$ROBOT/imu_link

# topics
info_topics=(
  /$ROBOT/multisense_forward/aux/image_rect_color/camera_info
  /$ROBOT/multisense_rear/aux/image_rect_color/camera_info
)
outfiles=(
  "cam_aux_front_intrinsics.yaml"
  "cam_aux_rear_intrinsics.yaml"
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
               "$os_frame:$imu_frame:$calibrations_dir/os_to_imu.yaml" \

    python $PROJECT_DIR/src/bagfile_extraction/extract_camera_info.py \
        --bagfile $DATASET_DIR/bagfiles/$scene.bag \
        --info_topics ${info_topics[@]} \
        --outfiles ${full_outfiles[@]}
done

# Kill roscore
kill $PID1

echo "Finish extracting camera parameters."