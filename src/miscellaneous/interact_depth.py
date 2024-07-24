import pathlib
from natsort import natsorted
import argparse
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt

from src.utils.depth import read_depth, show_depth


def format_coord(x, y, depth_image):
    numrows, numcols = depth_image.shape
    col = int(x + 0.5)
    row = int(y + 0.5)
    if col >= 0 and col < numcols and row >= 0 and row < numrows:
        z = depth_image[row, col]
        return f"x={x:.1f}, y={y:.1f}, depth={z:.3f} meters"
    else:
        return f"x={x:.1f}, y={y:.1f}"


def display_depth_image_with_hover(depth_image):
    fig, ax = plt.subplots()
    cax = ax.imshow(depth_image, cmap="viridis")
    ax.set_title("Hover over the image to see depth value")
    fig.colorbar(cax, ax=ax, label="Depth (meters)")

    ax.format_coord = lambda x, y: format_coord(x, y, depth_image)
    plt.show()


def main(args):
    depth_files = natsorted(args.dir.glob("*.png"))
    print(f"Found {len(depth_files)} depth images")

    for depth_file in tqdm(depth_files):
        depth_image = read_depth(depth_file)
        show_depth(depth_image)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, help="Path to the directory containing depth images"
    )
    args = parser.parse_args()

    args.dir = pathlib.Path(args.dir)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
