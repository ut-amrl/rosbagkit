#!/usr/bin/env bash
set -euo pipefail

# Run from within rosbagkit/docker

IMAGE_NAME="${IMAGE_NAME:-rosbagkit:latest}"

usage() {
  echo "Usage:"
  echo "  $0 /path/to/your/data"
  echo
  echo "Example:"
  echo "  $0 \$HOME/rosbags"
  echo
  echo "The provided host data directory will be mounted to:"
  echo "  /workspace/data"
  echo
}

if [ "$#" -ne 1 ]; then
  echo "Error: you must provide one argument that is the host data directory"
  echo
  usage
  exit 1
fi

HOST_DATA_DIR="$(realpath "$1")"

if [ ! -d "${HOST_DATA_DIR}" ]; then
  echo "Error: data directory does not exist:"
  echo "  ${HOST_DATA_DIR}"
  echo
  echo "Create it first"
  echo
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_CONFIG_DIR="${SCRIPT_DIR}/../config" # relative to rosbagkit/docker

docker run -it --rm \
  --name rosbagkit \
  -v "${HOST_DATA_DIR}:/workspace/data:rw" \
  -v "${HOST_CONFIG_DIR}:/workspace/rosbagkit/config:rw" \
  -w /workspace/rosbagkit \
  "${IMAGE_NAME}"
