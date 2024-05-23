#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

dataset_dir="/home/dongmyeong/Projects/datasets/SARA/crl_rzr"
scenes=(
  gq_TN_e3-baseline_rfv_250_remission_01_2024-02-09-14-39-36
)

pc_topic="/crl_rzr/velodyne_front_horiz_points"

trap "echo 'Script interrupted'; exit;" SIGINT

for scene in "${scenes[@]}"; do
  # Compensate pointcloud
  python $PROJECT_DIR/src/pointcloud_compensation/compensate_pointcloud_bagfile.py \
    --bagfile=$dataset_dir/bagfiles/$scene.bag \
    --pc_topic=$pc_topic \
    --dense_posefile=$dataset_dir/poses/$scene/fast_lio.txt \
    --out_pc_dir=$dataset_dir/3d_comp/$scene \
    --out_timestamps=$dataset_dir/timestamps/$scene/3d_comp.txt \
    --out_posefile=$dataset_dir/poses/$scene/os1.txt

  # Synchronize camera pose
  python $PROJECT_DIR/src/synchronization/sync_cam_pose_wanda.py \
    --dataset_dir=$dataset_dir --scene=$scene

  # Generate static map
  python $PROJECT_DIR/src/static_map_generation/generate_static_map_wanda.py \
    --dataset_dir=$dataset_dir --scene=$scene \
    --blind=20.0 --voxel_size=0.5 --nb_neighbors=10 --std_ratio=1.0
done