from PIL import Image
import numpy as np


def read_img(filename):
    return np.array(Image.open(filename))


def save_img(filename, img):
    im = Image.fromarray(img)
    im.save(filename)
