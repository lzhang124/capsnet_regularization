from PIL import Image, ImageDraw
import numpy as np


def read_img(filename):
    return np.array(Image.open(filename)) / 255


def save_img(img, filename):
    if img.shape[-1] == 1:
        mode = 'L'
        img = img[..., 0]
    else:
        mode = 'RGB'
    Image.fromarray((img * 255).astype('uint8'), mode).save(filename)
