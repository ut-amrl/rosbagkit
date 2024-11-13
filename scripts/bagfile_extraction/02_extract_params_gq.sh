#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

# input/output paths
BAGFILE_PATH=/robodata/ARL_SARA/GQ-dataset/bagfiles
OUTPUT_PATH=/robodata/dlee/Datasets/IRIS

# frames
OS_FRAME=trevor/ouster_center_link
IMU_FRAME=trevor/imu_link
CAM_FORWARD_FRAME=trevor/multisense_forward/aux_camera_optical_frame
CAM_REAR_FRAME=trevor/multisense_rear/aux_camera_optical_frame
CAM_LEFT_FRAME=trevor/stereo_left_optical_frame
CAM_RIGHT_FRAME=trevor/stereo_right_optical_frame

# camera info topics
CAMERA_INFO_TOPICS=(
  /trevor/multisense_forward/aux/camera_info
  /trevor/multisense_rear/aux/camera_info
  /trevor/stereo_left/camera_info
  /trevor/stereo_right/camera_info
)

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
  2024-08-15-12-40-46
  2024-08-15-12-44-16
  2024-08-15-12-53-53
  2024-08-15-12-56-35
  2024-08-15-13-08-57
  2024-08-15-13-23-03
  2024-08-15-13-27-41
  2024-08-15-13-33-25
  2024-08-15-13-37-11
  2024-08-15-13-43-35
  2024-08-15-13-48-29
  2024-08-15-13-52-10
  2024-08-15-13-59-13
)

outfiles=(
  "cam_aux_front_intrinsics.yaml"
  "cam_aux_rear_intrinsics.yaml"
  "cam_left_intrinsics.yaml"
  "cam_right_intrinsics.yaml"
)

trap "echo 'Script interrupted'; exit;" SIGINT


for scene in "${scenes[@]}" ; do
    calibrations_dir=$OUTPUT_PATH/calibrations/$scene
    full_outfiles=()
    for outfile in "${outfiles[@]}" ; do
        full_outfiles+=("$calibrations_dir/$outfile")
    done

    python $PROJECT_DIR/scripts/bagfile_extraction/02a_extract_tf.py \
    --bagfile $BAGFILE_PATH/$scene.bag \
    --tf_specs "$OS_FRAME:$CAM_FORWARD_FRAME:$calibrations_dir/os_to_cam_forward.yaml" \
               "$OS_FRAME:$CAM_REAR_FRAME:$calibrations_dir/os_to_cam_rear.yaml" \
               "$OS_FRAME:$CAM_LEFT_FRAME:$calibrations_dir/os_to_cam_left.yaml" \
               "$OS_FRAME:$CAM_RIGHT_FRAME:$calibrations_dir/os_to_cam_right.yaml" \
               "$OS_FRAME:$IMU_FRAME:$calibrations_dir/os_to_imu.yaml" \
               "$CAM_LEFT_FRAME:$CAM_RIGHT_FRAME:$calibrations_dir/cam_left_to_right.yaml"

    # python $PROJECT_DIR/src/bagfile_extraction/extract_camera_info.py \
    #     --bagfile $DATASET_DIR/bagfiles/$scene.bag \
    #     --info_topics ${info_topics[@]} \
    #     --outfiles ${full_outfiles[@]}
done

# Kill roscore
kill $PID1

echo "Finish extracting camera parameters."