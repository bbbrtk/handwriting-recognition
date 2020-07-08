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
    parser.add_argument("input")
    parser.add_argument("N")
    parser.add_argument("output")

    return parser.parse_args()


def get_image(dirpath, imgname):
    path =  os.path.join(dirpath, str(imgname)+'.jpg')
    if os.path.exists(path): 
        return cv2.imread(path, 0)
    else:
        sys.exit(f'ERROR: File {path} does not exist')


def doStuff(img, name, output_dir):
    lines_and_masks, img_numered_lines, img_only_words = getLineStripes(img, verbose=True)

    output_words = f"{output_dir}/{name}-wyrazy.png"
    onlywords = f"onlywords/{name}-wyrazy.png"
    output_indexes = f"{output_dir}/{name}-indeksy.txt"

    cv2.imwrite(output_words, img_numered_lines)
    cv2.imwrite(onlywords, img_only_words)

    print(":: image saved as", output_words)

    indexes = detect(lines_and_masks)
    
    f = open(output_indexes, "a")
    for index in indexes:
        f.write(index)
    f.close()

    print(":: indexes saved as", output_indexes)

def main():
    args = parser()
    for i in range(1,int(args.N)+1):
        img = get_image(args.input, i)
        doStuff(img, i, args.output)


if __name__ == "__main__":
    main()