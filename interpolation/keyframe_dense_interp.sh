#!/bin/bash

# Usage: ./interpolate_keyframes.sh [seq_number]
# desc: Interpolate keyframes using odometry poses

SEQ=${1:-0}  # Default sequence number is 0 if not provided

build/interpolate_keyframes \
-input_kf_dir=/home/dongmyeong/Projects/AMRL/CODa/poses/global_keyframe/ \
-input_odom_dir=/home/dongmyeong/Projects/AMRL/CODa/poses/dense_keyframe/ \
-output_dir=/home/dongmyeong/Projects/AMRL/CODa/poses/ \
-seq=$SEQ --logtostderr
