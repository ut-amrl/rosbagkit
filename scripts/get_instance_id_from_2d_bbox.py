"""
Author:      Dongmyeong Lee (domlee[at]utexas[dot]edu)
Date:        November 13, 2023
Description: Get instance ID of the object from 2D bounding box
"""
import pathlib
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Get instance ID for CODa")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/dongmyeong/Projects/AMRL/CODa",
        help="Path to the dataset",
    )

    return parser


def main(args):
    pass


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
