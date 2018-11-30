from PIL import Image, ImageDraw
import numpy as np


def read_img(filename):
    return np.array(Image.open(filename)) / 255


def save_img(img, filename):
    Image.fromarray((img * 255).astype('uint8')).save(filename)


def rotation_matrix(x1, x2, x3):
    # Fast Random Rotation Matrices, James Arvo
    theta = 2 * np.pi * x1 # rotation about the pole
    phi = 2 * np.pi * x2 # direction to deflect the pole
    z = x3 # amount of pole deflection

    R = np.array([[np.cos(theta), np.sin(theta), 0],
                   [-np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])
    v = np.array([[np.cos(phi) * np.sqrt(z)],
                   [np.sin(phi) * np.sqrt(z)],
                   [np.sqrt(1 - z)]])
    H = np.eye(3) - 2 * np.dot(v, v.T)
    M = -np.dot(H, R)
    return M


def draw_cube(rotation, image_size=32):
    CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    FACES = [[0, 1, 3, 2], [0, 2, 6, 4], [0, 4, 5, 1], [7, 6, 4, 5], [7, 5, 1, 3], [7, 3, 2, 6]]
    COLORS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]

    im = Image.new('RGB', (image_size, image_size), color=(0, 0, 0))
    d = ImageDraw.Draw(im)

    corners = np.dot(CORNERS, rotation)
    faces_i = sorted(range(6), key=lambda i: np.mean(corners[FACES[i]][:,2]), reverse=True)[:3]
    for i in faces_i:
        face = (corners[FACES[i]][:,:2] + 2) * image_size / 4
        d.polygon(list(map(tuple, face)), fill=COLORS[i])

    return np.array(im)
