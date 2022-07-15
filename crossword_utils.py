import cv2
import numpy as np
from utils import *
from sys import exit

def show_wait_destroy(winname, img):
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 500, 0)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyWindow(winname)
        exit('Pressed q - exiting ...')

    cv2.destroyWindow(winname)

def removeBorder(thresh, borderThickness = 2):
    
    h, w = thresh.shape
    blank = np.zeros(thresh.shape[:2], np.uint8)
    
    x1 = borderThickness
    y1 = borderThickness
    x2 = int(w - 2 * borderThickness)
    y2 = int(h - 2 * borderThickness)
    mask = cv2.rectangle(blank, (x1, y1), (x2, y2), 255, -1)
    masked = cv2.bitwise_and(thresh, thresh, mask=mask)

    return masked

def morphological_closing(gray):
    # https://stackoverflow.com/questions/10561222/how-do-i-equalize-contrast-brightness-of-images-using-opencv
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11)) # originally 11,11
    close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel1)
    div = np.float32(gray)/(close)
    output = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
    return output

def extract_largest_contour(input, output):
    mask = np.zeros((input.shape), np.uint8)
    contours, hierarchy = cv2.findContours(input, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find the biggest contour
    max_area = 0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt

    cv2.drawContours(mask, [best_cnt], 0, 255, -1) # full color (255) inverted
    cv2.drawContours(mask, [best_cnt], 0, 0, 2)    # no color (0) thickness 2

    # increase mask size so we don't cut away the lines when bitwising
    mask = cv2.dilate(mask, None, iterations=3)

    output = cv2.bitwise_and(output, mask)
    return output

def extractVerticalLines(input):
    # Finding Vertical lines
    kernel1X = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))

    dx = cv2.Sobel(input, cv2.CV_8U, dx=2, dy=0) # originally CV_16S
    dx = cv2.convertScaleAbs(dx)
    cv2.normalize(dx, dx, 0, 255, cv2.NORM_MINMAX)
    ret, close = cv2.threshold(dx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernel1X, iterations = 1)

    contours, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h/w > 20: # originally 5
            cv2.drawContours(close, [cnt], 0, 255, -1)  # full color (255)
        else:
            cv2.drawContours(close, [cnt], 0, 0, -1)    # no color (0)

    close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, None, iterations = 2)
    closeX = close.copy()
    return closeX

def extractHorizontalLines(input):
    # Finding Horizontal Lines
    kernel1Y = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    dy = cv2.Sobel(input, cv2.CV_8U, dx=0, dy=2) # originally CV_16S
    dy = cv2.convertScaleAbs(dy)
    cv2.normalize(dy, dy, 0, 255, cv2.NORM_MINMAX)
    ret, close = cv2.threshold(dy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernel1Y)

    contours, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w/h > 20: # originally 5
            cv2.drawContours(close, [cnt], 0, 255, -1)  # full color (255)
        else:
            cv2.drawContours(close, [cnt], 0, 0, -1)   # no color (0)

    close = cv2.morphologyEx(close, cv2.MORPH_DILATE, None, iterations = 2)
    closeY = close.copy()
    return closeY

def writeArrayToDisk(arr, out = 'array_out.txt'):
    dim = arr.ndim 
        
    with open(out, 'w') as outfile:    
        outfile.write('# Array shape: {0}\n'.format(arr.shape))
        
        if dim == 1 or dim == 2:
            np.savetxt(outfile, arr, fmt='%10.3f')
            # output as CSV
            # np.savetxt(outfile, arr, delimiter=",", fmt="%10.5f")

        elif dim == 3:
            for i, arr2d in enumerate(arr):
                outfile.write('# {0}-th channel\n'.format(i))
                np.savetxt(outfile, arr2d, fmt='%10.3f')
                
        elif dim == 4:
            for j, arr3d in enumerate(arr):
                outfile.write('\n# {0}-th Image\n'.format(j))
                for i, arr2d in enumerate(arr3d):
                    outfile.write('# {0}-th channel\n'.format(i))
                    np.savetxt(outfile, arr2d, fmt='%10.3f')

        else:
            print("Out of dimension!")