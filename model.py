import cv2
import numpy as np

from scipy.signal import savgol_filter, find_peaks
from whitestripes import findLines



def cropIndex(img, mask, tresholded=False, verbose=False):
    ''' crop index out of a text line '''

    im2 = img.copy()
    # try to remove lines in trsholded image by OTSU tresh
    img = cv2.GaussianBlur(img, (5, 5), 7)
    et,img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

    # remove artefacts - big white dots
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, None, None, None, 8, cv2.CV_32S)
    sizes = stats[1:, -1] #get CC_STAT_AREA component
    img = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if sizes[i] >= 60:   #filter small dotted regions
            img[labels == i + 1] = 255

    # optional color inverter
    # img = cv2.bitwise_not(img)

    # soft erosion
    kernel = np.ones((1,1),np.uint8)
    img = cv2.erode(img,kernel,iterations = 1)

    if verbose:
        cv2.imshow('img_lines_removed', img)
        cv2.waitKey(0)

    # strong opening to make contour big and full
    kernelmask = np.ones((15,15),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelmask)

    # strong dilatation to make it even bigger
    kerneldil = np.ones((25,25),np.uint8)
    mask = cv2.dilate(mask, kerneldil, iterations = 1)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if verbose:
        cv2.imshow('img_lines_removed', mask)
        cv2.waitKey(0)

    # optional crop
    # cropped = cv2.bitwise_and(img, img, mask = mask)

    # add 30px black border to ease the process of finding contours
    img = cv2.copyMakeBorder(img, 30, 30, 30, 30, cv2.BORDER_CONSTANT) 
    mask = cv2.copyMakeBorder(mask, 30, 30, 30, 30, cv2.BORDER_CONSTANT) 
    im2 = cv2.copyMakeBorder(im2, 30, 30, 30, 30, cv2.BORDER_CONSTANT) 

    # find contours
    edged = cv2.Canny(mask, 30, 200)   
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    
    if verbose:
        print("Number of Contours found = " + str(len(contours))) 

    # find contour which lays on the right side of image
    cur_max = 0
    cur_cont = 0
    for index, cont in enumerate(contours):
        if max(cont[:, 0][:,0]) > cur_max:
            cur_max = max(cont[:, 0][:,0])
            cur_cont = index

    if verbose:
        cv2.drawContours(img, contours, cur_cont, (255, 255, 255), 3) 
        cv2.imshow('Cont', img) 
        cv2.waitKey(0) 

    # crop rectangle of contour from image
    x,y,w,h = cv2.boundingRect(contours[cur_cont])
    if tresholded:
        # black-white image after tresholds 
        img_rect_cropped = img[y:y+h, x:x+w]
    else:
        # normal image
        img_rect_cropped = im2[y:y+h, x:x+w]

    if verbose:
        cv2.imshow('Cont', img_rect_cropped) 
        cv2.waitKey(0) 

    return img_rect_cropped


path = []
path.append('aftercut/511')
path.append('aftercut/1052')
path.append('aftercut/970')
path.append('aftercut/1269')
path.append('aftercut/1118')
path.append('aftercut/739')

for p in path:
    img = cv2.imread(f"{p}.png", 0)
    mask = cv2.imread(f"{p}_mask.png")
    
    img_index = cropIndex(img, mask, tresholded=False, verbose=False)
    cv2.imshow('Cont', img_index) 
    cv2.waitKey(0) 
    img = np.array(img_index)
    img = cv2.GaussianBlur(img, (3, 3), 5)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    img = np.pad(img, ((0, 0), (0, 5)), 'constant')
    cv2.imshow('Cont', img) 
    cv2.waitKey(0) 

    height, width = img.shape
    window_width = int(0.7 * height)
    move_window_step = int(0.14 * height)
    start = 0
    end = window_width
    consecutive = 0
    previous = -1
    index_text = ''

    # move from left to right to find number
    while end <= width:
        th = img[:, start:end]
        size_image = cv2.resize(th, (34, 34))
        sample = size_image / 255
        sample = np.array(sample).reshape(34, 34, 1)
        sample = np.expand_dims(sample, axis=0)
        cv2.imshow('Cont', th) 
        cv2.waitKey(0)
        start += move_window_step
        end += move_window_step