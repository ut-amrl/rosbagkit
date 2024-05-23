#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

dataset_dir="/home/dongmyeong/Projects/datasets/CODa"
# sequences=(0 1 2 3 4 5 6 7 9 10 11 12 13 16 17 18 19 20 21 22)
sequences=(0 1 2 3 4 5 6 7 9 10 11)

trap "echo 'Script interrupted'; exit;" SIGINT

for seq in "${sequences[@]}"; do
  python $PROJECT_DIR/src/pointcloud_compensation/compensate_pointcloud_coda.py \
    --dataset_dir $dataset_dir --seq $seq \
    --ref_posefile $dataset_dir/poses/point-lio/$seq.txt \
    --out_posefile $dataset_dir/poses/point-lio/sync_$seq.txt
done