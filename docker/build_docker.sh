#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-rosbagkit:latest}"
ROSBAGKIT_COMMIT="${ROSBAGKIT_COMMIT:-quattro}"

USER_UID="$(id -u)"
USER_GID="$(id -g)"

docker build \
  --build-arg USER_UID="${USER_UID}" \
  --build-arg USER_GID="${USER_GID}" \
  --build-arg ROSBAGKIT_COMMIT="${ROSBAGKIT_COMMIT}" \
  -t "${IMAGE_NAME}" \
  -f Dockerfile \
  .
