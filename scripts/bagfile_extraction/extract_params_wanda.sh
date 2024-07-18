#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

# frames
os_frame="wanda/center_ouster_link"
cam_left_frame="wanda/stereo_left_optical_frame"
cam_right_frame="wanda/stereo_right_optical_frame"
imu_frame="wanda/imu_link"

# topics
info_topics=(
  "/wanda/stereo_left/camera_info"
  "/wanda/stereo_right/camera_info"
)
outfiles=(
  "cam_left_intrinsics.yaml"
  "cam_right_intrinsics.yaml"
)

DATASET_DIR=$PROJECT_DIR/data/SARA/wanda
scenes=(
  gq_appld_south_tour_01_2024-03-14-10-08-34
  gq_appld_wandagq_32_field_foresttrail_06_2024-03-15-11-17-44
  gq_appld_wandagq_32_forest_02_2024-03-15-12-02-37
  gq_appld_wandagq_32_forest_03_2024-03-15-12-16-36
  gq_appld_forest_mission_autonomous_deployment_01_2024-03-12-14-15-49
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
    --tf_specs "$os_frame:$cam_left_frame:$calibrations_dir/os_to_cam_left.yaml" \
               "$os_frame:$cam_right_frame:$calibrations_dir/os_to_cam_right.yaml" \
               "$os_frame:$imu_frame:$calibrations_dir/os_to_imu.yaml"

  python $PROJECT_DIR/src/bagfile_extraction/extract_camera_info.py \
    --bagfile $DATASET_DIR/bagfiles/$scene.bag \
    --info_topics ${info_topics[@]} \
    --outfile ${full_outfiles[@]}
done