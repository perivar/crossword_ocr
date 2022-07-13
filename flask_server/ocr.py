import pytesseract
import requests
from PIL import Image
from PIL import ImageFilter
from io import StringIO
import cv2
import os
import numpy as np

def process_image(url):
    print("--call process_image--")
    image = _get_image(url)
    image.filter(ImageFilter.SHARPEN)
    return pytesseract.image_to_string(image)


def process_image2(image):
    print("--call process_image2--")
    image.filter(ImageFilter.SHARPEN)
    return pytesseract.image_to_string(image)

def process_image3(image):
    print("--call process_image3--")
    image.filter(ImageFilter.SHARPEN)
    image.filter(ImageFilter.EDGE_ENHANCE)
    image.filter(ImageFilter.FIND_EDGES)
    return pytesseract.image_to_string(image)

def process_image4(image):
    print("--call process_image4--")
    path_output_dir = "images"
    imgPath = os.path.join(path_output_dir, 'capture.png')
    os.makedirs(path_output_dir, exist_ok=True) 
    print('Saving snapshot to ', imgPath)

    # convert from PIL image format to CV2 format
    cv2Image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imwrite(imgPath, cv2Image)
    
    # image.save(imgPath, format="png") 
    return pytesseract.image_to_string(image)

def _get_image(url):
    return Image.open(StringIO(requests.get(url).content))
