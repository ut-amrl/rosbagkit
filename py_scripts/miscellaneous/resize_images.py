"""
Author:      Dongmyeong Lee (domlee[at]utexas.edu)
Date:        Apr 29, 2024
Description: Resize images in a directory
"""

import pathlib
import argparse
from tqdm import tqdm
from PIL import Image


def main(args):
    # Resize images in a directory (png, jpg)
    img_files = list(args.source_dir.glob("*"))
    for img_file in tqdm(img_files, leave=False):
        if img_file.suffix not in [".png", ".jpg"]:
            continue
        img = Image.open(img_file)
        img_resized = img.resize(args.new_size, Image.LANCZOS)
        img_resized.save(args.target_dir / img_file.name)


def get_args():
    parser = argparse.ArgumentParser(description="Resize images in a directory")
    parser.add_argument("--source_dir", type=str, help="Source directory")
    parser.add_argument("--target_dir", type=str, help="Target directory")
    parser.add_argument("--new_size", nargs=2, type=int, help="New size (w, h)")
    args = parser.parse_args()

    args.source_dir = pathlib.Path(args.source_dir)
    args.target_dir = pathlib.Path(args.target_dir)
    args.target_dir.mkdir(parents=True, exist_ok=True)
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
