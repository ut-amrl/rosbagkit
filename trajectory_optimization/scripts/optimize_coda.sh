#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../)

dataset_dir="/home/dongmyeong/Projects/datasets/CODa"
SEQ=${1:-0}  # Default sequence number is 0 if not provided

$PROJECT_DIR/build/interpolate_keyframes \
    --input_kf_file $dataset_dir/interactive_slam/key_graph/${SEQ}.txt \
    --input_odom_file $dataset_dir/poses/point-lio/sync_${SEQ}.txt \
    --output_file $dataset_dir/poses/${SEQ}.txt