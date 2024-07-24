#!/bin/bash
DOCKER_IMAGE=domlee93/dataset-tools:latest
USER=$(whoami)
DATA_DIR=/media/dongmyeong/T7_Shield/SARA/wilbur

xhost +local:

# Run the Docker container with X11 forwarding enabled
docker run -it --rm \
  --gpus all \
  --user $USER \
  --network host \
  --privileged \
  --env DISPLAY=$DISPLAY \
  --env QT_X11_NO_MITSHM=1 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/home/$USER/dataset-tools \
  -v $DATA_DIR:/home/$USER/dataset-tools/data/SARA/wilbur \
  -w /home/$USER/dataset-tools \
  $DOCKER_IMAGE 
