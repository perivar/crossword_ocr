import cv2
import numpy as np
from utils import *
from sys import exit
import json

# https://github.com/hmallen/numpyencoder/blob/master/numpyencoder/numpyencoder.py
# https://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    
        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)

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
    (cnts, _) = cv2.findContours(input, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find the biggest contour
    # max_area = 0
    # best_cnt = None
    # for cnt in cnts:
    #     area = cv2.contourArea(cnt)
    #     if area > max_area:
    #         max_area = area
    #         best_cnt = cnt

    # find the biggest contour, cannot use the 4 side version, since the grid can be skewed
    best_cnt, _ = biggestContour(cnts, False, 0) 

    cv2.drawContours(mask, [best_cnt], 0, 255, -1) # full color (255) inverted
    cv2.drawContours(mask, [best_cnt], 0, 0, 2)    # no color (0) thickness 2

    # increase mask size so we don't cut away the lines when bitwising
    mask = cv2.dilate(mask, None, iterations=3)

    output = cv2.bitwise_and(output, mask)
    return output

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

def writeListToDisk(list, jsonFile = 'list_out.json'):
    with open(jsonFile, 'w') as fp:
        # for numpy_data in list:
        #     json.dump(numpy_data, fp, indent=4, sort_keys=True,
        #       separators=(', ', ': '), ensure_ascii=False,
        #       cls=NumpyEncoder)
        #     fp.write("\n")
        json.dump(list, fp, indent=4, sort_keys=True,
            separators=(', ', ': '), ensure_ascii=False,
            cls=NumpyEncoder)
        
def readListFromDisk(jsonFile = 'list_out.json'):
    # for reading also binary mode is important
    with open(jsonFile, 'rb') as fp:
        n_list = json.load(fp)
        return n_list

def removeNoise(thresh, minArea = 5000):
    (cnts, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < minArea:
            cv2.drawContours(thresh, [c], -1, (0,0,0), -1)


