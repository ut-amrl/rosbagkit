import argparse
import cv2

from src.utils.image import draw_epipolar_lines


def main(args):
    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)
    draw_epipolar_lines(img1, img2, args.outfile)


def get_args():
    parser = argparse.ArgumentParser(description="Draw epipolar lines")
    parser.add_argument("--img1", type=str, help="Path to the first image")
    parser.add_argument("--img2", type=str, help="Path to the second image")
    parser.add_argument(
        "--outfile", type=str, default=None, help="Path to the output image"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
