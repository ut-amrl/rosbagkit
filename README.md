# rosbagkit

`rosbagkit` is a lightweight toolkit for reading ROS bagfiles and extracting
common sensor streams such as RGB images, depth images, odometry, IMU data,
GPS data, and TF transforms.

The repository is organized around:

- a Python package in `src/rosbagkit`
- extraction scripts in `scripts/`
- YAML-driven extraction examples in `config/extract/`

## Requirements

- Python 3.12+
- ROS bagfiles supported by [`rosbags`](https://pypi.org/project/rosbags/)

## Install

```bash
uv sync
source .venv/bin/activate
```

[uv](https://docs.astral.sh/uv/getting-started/installation/) must be installed separately before running these commands.

## Usage

### Extract bag contents from a YAML config

Use one of the template configs in `config/extract/` as a starting point and
replace the placeholder paths, topics, and scene entries with your own data.

```bash
python scripts/extract_bagfile.py config/extract/UT-SARA-GQ.yaml
```

Each config follows this shape:

```yaml
bagfile_root: /path/to/bagfiles
output_root: /path/to/output

topics:
  /camera/color/image_raw:
    format: image
    outdir: image_raw
  /gps/fix:
    format: csv
    outpath: gps.csv

scenes:
  example_scene:
    bagfiles:
      - example_run/example_0.bag
    start: 0
    end: -1
```

Supported topic output formats currently include:

- `image`
- `depth`
- `pointcloud_depth`
- `csv`

Optional stereo rectification can be configured directly in the YAML file:

```yaml
rectification:
  enabled: true
  left_topic: /stereo/left/image_raw
  right_topic: /stereo/right/image_raw
  left_calib: /path/to/left_intrinsics.yaml
  right_calib: /path/to/right_intrinsics.yaml
  extrinsics: /path/to/left_to_right.yaml
  threshold: 0.005
  output_dir: 2d_rect
  left_subdir: cam_left
  right_subdir: cam_right
  timestamp_file: timestamps.txt
```

When enabled, the configured stereo topics are rectified in memory and only the final rectified images are written to disk. See [`config/extract/UT-SARA-GQ.yaml`](config/extract/UT-SARA-GQ.yaml) for a public GQ example.

### Extract a TF transform from a bagfile

```bash
python scripts/extract_tf.py /path/to/example.bag --src base_link --tgt camera_link
```

This writes a YAML file next to the bagfile containing the resolved transform chain.
