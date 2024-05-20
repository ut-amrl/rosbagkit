#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

dataset_dir="/home/dongmyeong/Projects/datasets/SARA/wanda"
scenes=(
  gq_appld_wandagq_32_field_foresttrail_06_2024-03-15-11-17-44
  gq_appld_wandagq_32_forest_02_2024-03-15-12-02-37
  gq_appld_wandagq_32_forest_03_2024-03-15-12-16-36
  gq_appld_forest_mission_autonomous_deployment_01_2024-03-12-14-15-49
  gq_appld_south_tour_01_2024-03-14-10-08-34
  # gq_TN_Menu_A_datacollect_02_2024-02-21-17-23-09
)

trap "echo 'Script interrupted'; exit;" SIGINT

for scene in "${scenes[@]}" ; do
  python $PROJECT_DIR/py_scripts/depth_generation/generate_depth_wanda.py \
    --dataset_dir=${dataset_dir} --scene=${scene} --window=20 --debug
done