from PIL import Image
import numpy as np


def read_img(filename):
    return np.array(Image.open(filename)) / 255


def save_img(img, filename):
    Image.fromarray((img * 255).astype('uint8')).save(filename)
