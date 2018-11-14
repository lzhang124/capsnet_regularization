from PIL import Image


def save_image(filename, image):
    im = Image.fromarray(image)
    im.save(filename)
