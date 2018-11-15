from keras.utils import Sequence
from PIL import Image, ImageDraw
import numpy as np


CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
FACES = [[0, 1, 3, 2], [0, 2, 6, 4], [0, 4, 5, 1], [7, 6, 4, 5], [7, 5, 1, 3], [7, 3, 2, 6]]
COLORS = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]


def random_rotation():
    # Fast Random Rotation Matrices, James Arvo
    x1, x2, x3 = np.random.rand(3)
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
    return theta, phi, z, M


def draw_cube(image_size, rotation):
    im = Image.new('RGB', (image_size, image_size), color=(1, 1, 1))
    d = ImageDraw.Draw(im)

    corners = np.dot(CORNERS, rotation)
    faces_i = sorted(range(6), key=lambda i: np.mean(corners[FACES[i]][:,2]), reverse=True)[:3]
    for i in faces_i:
        face = (corners[FACES[i]][:,:2] + 2) * image_size / 4
        d.polygon(list(map(tuple, face)), fill=COLORS[i])

    return np.array(im)


class CubeGenerator(Sequence):
    def __init__(self, n=1, image_size=256, batch_size=1, label_type=None):
        self.n = n
        self.image_size = image_size
        self.batch_size = batch_size
        self.label_type = label_type

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, i):
        if i >= len(self):
            raise ValueError('Asked to retrieve element {}, but the Sequence has length {}'.format(i, len(self)))
        return self._generate_batch()

    def _generate_batch(self):
        batch = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        poses = np.zeros((self.batch_size, 3))
        for i in range(self.batch_size):
            theta, phi, z, rot = random_rotation()
            batch[i] = draw_cube(self.image_size, rot)
            poses[i] = theta, phi, z

        if self.label_type == 'pose':
            return (batch, poses)
        if self.label_type == 'input':
            return (batch, batch)
        return batch
