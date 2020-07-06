import cv2
import numpy as np
from scipy.signal import savgol_filter, find_peaks

def swap(a, b):
    return b, a

def findLines(img, vertical):
    kernel = np.ones((1, 7))
    dx, dy = 0, 1
    o1, o2 = 3, 7
    roll = 3
    ax = 0
    if vertical:
        kernel = np.ones((5, 1))
        dx, dy = swap(dx, dy)
        o1, o2 = swap(o1, o2)
        roll = 2
        ax = 1

    modified = np.abs(cv2.Sobel(img, -1, dx, dy, ksize=5))
    modified = cv2.morphologyEx(modified, cv2.MORPH_OPEN, kernel, iterations=3)
    modified = cv2.morphologyEx(modified, cv2.MORPH_CLOSE, np.ones((o1, o2)), iterations=1)
    modified = np.roll(modified, roll, axis=ax)

    return modified


def convertImage(img, verbose):
    # blurring image
    img = cv2.GaussianBlur(img, (5, 5), 7)
    # treshold to enhance lines
    img_treshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # fix (bold) lines - horizontally
    img_fix_hor = cv2.morphologyEx(img_treshold, cv2.MORPH_CLOSE, np.ones((3, 5)), iterations=1)
    # fix (bold) lines - vertically
    img_fix_ver = cv2.morphologyEx(img_treshold, cv2.MORPH_CLOSE, np.ones((5, 3)), iterations=1)

    # find and remove lines
    img_lines_removed = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 7)
    lines_hor = findLines(img_fix_hor, vertical=False)
    lines_ver = findLines(img_fix_ver, vertical=True)
    img_lines_removed[lines_hor > 0] = 0
    img_lines_removed[lines_ver > 0] = 0

    # noise reduction
    img_lines_removed = cv2.morphologyEx(img_lines_removed, cv2.MORPH_CLOSE, np.ones((7, 7)), iterations=5)
    img_lines_removed = cv2.morphologyEx(img_lines_removed, cv2.MORPH_OPEN, np.ones((3, 3)), iterations=1)
    
    if verbose:
        cv2.imshow('Lines Removed', img_lines_removed)
        cv2.waitKey(0)

    return img_lines_removed


def cutLines(img_lines_removed, img_original, verbose):
    rowsums = np.sum(img_lines_removed, axis=1)
    img_sum = np.ones(40)
    img_sum[0:11] = 0.5
    img_sum[-11:] = 0.5

    # random convolution with savgol
    img_conv = np.convolve(rowsums, img_sum, mode="valid")
    img_conv = savgol_filter(img_conv, 41, 3)

    # find lines which are image peaks
    list_of_peaks, _ = find_peaks(img_conv, threshold=0.4)
    lines_list = []
    for peak in list_of_peaks:
        if peak - 10 > 0 and peak + 40 < img_original.shape[0]:
            cut = img_original[peak - 10:peak + 40, :]
            lines_list.append(cut)

            if verbose:
                cv2.imshow('pre', cut)
                cv2.waitKey(0)

    return lines_list


def getLineStripes(filepath, verbose):
    img = cv2.imread(filepath, 0)
    img_original = img.copy() 

    img_modified = convertImage(img, verbose)
    lines = cutLines(img_modified, img_original, verbose)

    return lines

