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
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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
        cv2.imshow('img_lines_removed', img_lines_removed)
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
    masks_list = []
    iter = 1

    img_mask = img_lines_removed.copy()
    img_lines_removed[img_lines_removed == 255] = 1
    for peak in list_of_peaks:
        if peak - 15 > 0 and peak + 40 < img_original.shape[0]:
            lr = peak - 15
            pr = peak + 40

            cut = img_original[ lr : pr, :]
            img_mask[ lr : pr, :] *= 100
            img_lines_removed[ lr : pr, :] *= iter

            lines_list.append(cut)

            mini_mask = img_mask[ lr : pr, :].copy()
            mini_mask[mini_mask == 255] = 0
            mini_mask[mini_mask > 10] = 255
            masks_list.append(mini_mask)

            iter += 1

            cv2.imwrite(f"aftercut/{peak}.png", cut)
            cv2.imwrite(f"aftercut/{peak}_mask.png", mini_mask)

            if verbose:
                cv2.imshow('cut', cut)
                cv2.waitKey(0)

    img_lines_removed[img_lines_removed == 255] = 0
    img_mask[img_mask == 255] = 0

    if verbose:
        cv2.imshow('img_mask', img_mask)
        cv2.waitKey(0)

    return [lines_list, masks_list], img_lines_removed, img_mask


def getLineStripes(img, verbose):
    img_original = img.copy() 
    img_modified = convertImage(img, verbose)

    lines_and_masks, img_numered_lines, image_mask = cutLines(img_modified, img_original, verbose)
    img_only_words = cv2.bitwise_and(img_original, img_original, mask = image_mask)

    if verbose:
        cv2.imshow('img_only_words', img_only_words)
        cv2.waitKey(0)

    return lines_and_masks, img_numered_lines, img_only_words

