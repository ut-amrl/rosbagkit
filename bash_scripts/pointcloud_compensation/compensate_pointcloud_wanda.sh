#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

scenes=(
  gq_appld_south_tour_01_2024-03-14-10-08-34
  gq_appld_wandagq_32_field_foresttrail_06_2024-03-15-11-17-44
  gq_appld_wandagq_32_forest_02_2024-03-15-12-02-37
  gq_appld_wandagq_32_forest_03_2024-03-15-12-16-36
  gq_TN_Menu_A_datacollect_02_2024-02-21-17-23-09
  gq_appld_forest_mission_autonomous_deployment_01_2024-03-12-14-15-49
)
dataset_dir="/home/dongmyeong/Projects/datasets/SARA"

for scene in "${scenes[@]}"; do
    python $PROJECT_DIR/py_scripts/pointcloud_compensation/compensate_pointcloud_bagfile.py \
        --bagfile $dataset_dir/bagfiles/$scene.bag \
        --pc_topic /wanda/lidar_points \
        --dense_posefile $dataset_dir/poses/point_lio/$scene.txt \
        --outdir $dataset_dir/3d_comp/$scene \
        --ts_outfile $dataset_dir/timestamps/$scene/3d_comp.txt \
        --pose_outfile $dataset_dir/poses/os1/$scene.txt
done