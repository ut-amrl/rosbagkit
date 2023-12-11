#!/bin/bash

# Usage: ./interpolate_keyframes.sh [seq_number]
# desc: Interpolate LeGO-LOAM poses using INEKF odometry

SEQ=${1:-0}  # Default sequence number is 0 if not provided

build/interpolate_keyframes \
-input_kf_dir=/home/dongmyeong/Projects/AMRL/CODa/poses/ \
-input_odom_dir=/home/dongmyeong/Projects/AMRL/CODa/poses/inekfodom/sync/ \
-output_dir=/home/dongmyeong/Projects/AMRL/CODa/poses/dense/ \
-seq=$SEQ --logtostderr
