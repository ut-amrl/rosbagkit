#!/bin/bash

for i in {0..22}
do
  if [ "$i" -eq 14 ]; then
    echo "Skipping Sequence: $i"
    continue
  fi
  echo "Running Sequence: $i"
  python visualize_coda_rviz.py -d /home/dongmyeong/Projects/AMRL/CODa -s $i
  #python object_image_cropper.py -d /home/dongmyeong/Projects/AMRL/CODa_dev -s $i
done

