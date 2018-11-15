from PIL import Image
import numpy as np


def read_img(filename):
    return np.array(Image.open(filename)) / 255


def save_img(filename, img):
    img *= 255
    Image.fromarray(img.astype('uint8')).save(filename)
