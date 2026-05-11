#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-rosbagkit:latest}"

# Usage:
#   ./run_docker.sh /path/to/your/data
#
# Default host data dir:
#   ./data
HOST_DATA_DIR="${1:-${PWD}/data}"

mkdir -p "${HOST_DATA_DIR}"

docker run -it --rm \
  --name rosbagkit \
  -v "${HOST_DATA_DIR}:/workspace/data:rw" \
  -w /workspace/rosbagkit \
  "${IMAGE_NAME}"
