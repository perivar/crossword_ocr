import cv2
import numpy as np

def removeOuterBorder(image, thresh):
    # https://stackoverflow.com/questions/58084229/remove-borders-from-image-but-keep-text-written-on-borders-preprocessing-before
    removed = image.copy()

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(removed, [c], -1, (255,255,255), 5)

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(removed, [c], -1, (255,255,255), 5)

    # Repair kernel
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    removed = 255 - removed
    dilate = cv2.dilate(removed, repair_kernel, iterations=5)
    dilate = cv2.cvtColor(dilate, cv2.COLOR_BGR2GRAY)
    pre_result = cv2.bitwise_and(dilate, thresh)

    result = cv2.morphologyEx(pre_result, cv2.MORPH_CLOSE, repair_kernel, iterations=5)
    final = cv2.bitwise_and(result, thresh)

    invert_final = 255 - final

    return invert_final

def get_cropped_image(image, x, y, w, h):
    cropped_image = image[ y:y+h, x:x+w ]
    return cropped_image

def get_ROI(image, horizontal, vertical, left_line_index, right_line_index, top_line_index, bottom_line_index, offset=4):
    # https://levelup.gitconnected.com/text-extraction-from-a-table-image-using-pytesseract-and-opencv-3342870691ae
    x1 = vertical[left_line_index][2] + offset
    y1 = horizontal[top_line_index][3] + offset
    x2 = vertical[right_line_index][2] - offset
    y2 = horizontal[bottom_line_index][3] - offset
    
    w = x2 - x1
    h = y2 - y1
    
    cropped_image = get_cropped_image(image, x1, y1, w, h)
    
    return cropped_image, (x1, y1, w, h)

def extractVerticalLines(input):
    # https://stackoverflow.com/questions/10196198/how-to-remove-convexity-defects-in-a-sudoku-square

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
    # https://stackoverflow.com/questions/10196198/how-to-remove-convexity-defects-in-a-sudoku-square
    
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