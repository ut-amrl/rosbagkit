#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

DATASET_DIR=$PROJECT_DIR/data/CODa
# sequences=(0 1 2 3 4 5 6 7 9 10 11 12 13 16 17 18 19 20 21 22)
sequences=(2 3 4 5 6 7 9 10)

trap "echo 'Script interrupted'; exit;" SIGINT

for seq in "${sequences[@]}"; do
  python $PROJECT_DIR/src/pointcloud_compensation/compensate_pointcloud_coda.py \
    --DATASET_DIR $DATASET_DIR --seq $seq \
    --ref_posefile $DATASET_DIR/poses/ct-icp/$seq.txt \
    --out_posefile $DATASET_DIR/poses/ct-icp_sync/$seq.txt
done