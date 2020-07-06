import argparse
import os
import sys
import cv2

from whitestripes import getLineStripes
from detector import detect

VERBOSE = True

def parser():
    """ Parse arguments """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("imput")
    parser.add_argument("N")
    parser.add_argument("output")

    return parser.parse_args()


def get_image(dirpath, imgname):
    path =  os.path.join(dirpath, str(imgname)+'.png')
    if os.path.exists(path): 
        return cv2.imread(path)
    else:
        sys.exit(f'ERROR: File {path} does not exist')


def doStuff(img):
    lines = getLineStripes(img, True)
    # TO DO


def main():
    args = parser()
    for i in range(int(args.N)):
        img = get_image(args.input, i)
        doStuff(img, args.output)


if __name__ == "__main__":
    main()