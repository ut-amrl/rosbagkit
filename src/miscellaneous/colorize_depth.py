import pathlib
from natsort import natsorted
import argparse
from tqdm import tqdm

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image


def main(args):
    depth_files = list(map(str, natsorted(pathlib.Path(args.dir).glob("*.png"))))

    for depth_file in tqdm(depth_files):
        depth_data = np.asarray(Image.open(depth_file))
        depth_img = depth_data / 1000

        plt.figure()
        # save the depth image
        plt.imshow(depth_img)
        plt.title(depth_file)

        outfile = args.out_dir / pathlib.Path(depth_file).name
        plt.savefig(outfile)
        plt.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Path to the dataset")
    args = parser.parse_args()

    args.dir = pathlib.Path(args.dir)

    args.out_dir = args.dir.parent.as_posix().replace("2d_depth", "2d_depth_color")
    args.out_dir = pathlib.Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
