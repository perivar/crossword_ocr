import pytesseract
import requests
from PIL import Image
from PIL import ImageFilter
from io import StringIO


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

def _get_image(url):
    return Image.open(StringIO(requests.get(url).content))
