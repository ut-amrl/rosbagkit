#!/bin/bash
PROJECT_DIR=$(realpath $(dirname "$0")/../..)

dataset_dir="/home/dongmyeong/Projects/datasets/CODa"
sequences=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22)
new_size="612 512"  # (1/2 of the original size)
# new_size="306 256" # (1/4 of the original size)

trap "echo 'Script interrupted'; exit;" SIGINT

for seq in "${sequences[@]}" ; do
    for cam in "cam0" "cam1" ; do
        echo "Resizing images for sequence $seq, camera $cam"
        python $PROJECT_DIR/src/miscellaneous/resize_images.py \
            --source_dir $dataset_dir/2d_raw/$cam/$seq \
            --target_dir $dataset_dir/2d_raw_resized_2/$cam/$seq \
            --new_size $new_size
    done
done