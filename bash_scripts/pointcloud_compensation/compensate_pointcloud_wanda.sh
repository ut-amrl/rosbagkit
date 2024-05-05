#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)
trap "echo 'Script interrupted'; exit;" SIGINT

scenes=(
  gq_appld_south_tour_01_2024-03-14-10-08-34
  gq_appld_wandagq_32_field_foresttrail_06_2024-03-15-11-17-44
  gq_appld_wandagq_32_forest_02_2024-03-15-12-02-37
  gq_appld_wandagq_32_forest_03_2024-03-15-12-16-36
  gq_TN_Menu_A_datacollect_02_2024-02-21-17-23-09
  gq_appld_forest_mission_autonomous_deployment_01_2024-03-12-14-15-49
)
dataset_dir="/home/dongmyeong/Projects/datasets/SARA"

lio_pose="point_lio.txt"
os1_pose="os1.txt"

for scene in "${scenes[@]}"; do
  # Compensate pointcloud
  python $PROJECT_DIR/py_scripts/pointcloud_compensation/compensate_pointcloud_bagfile.py \
    --bagfile $dataset_dir/bagfiles/$scene.bag \
    --pc_topic /wanda/lidar_points \
    --dense_posefile $dataset_dir/poses/$scene/$lio_pose \
    --out_pc_dir $dataset_dir/3d_comp/$scene \
    --out_timestamps $dataset_dir/timestamps/$scene/3d_comp.txt \
    --out_posefile $dataset_dir/poses/$scene/$os1_pose \

  # Synchronize camera pose
  python $PROJECT_DIR/py_scripts/synchronization/sync_cam_pose_wanda.py \
    --dataset_dir=${dataset_dir} --scene=${scene}

  # Generate static map
  python $PROJECT_DIR/py_scripts/static_map_generation/generate_static_map_wanda.py \
    --dataset_dir=${dataset_dir} --scene=${scene} \
    --blind 20.0 --voxel_size 1.0 --nb_neighbors 100 --std_ratio 1.0 --visualize

done