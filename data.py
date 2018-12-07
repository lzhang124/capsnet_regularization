from keras.datasets import mnist
from keras.utils import Sequence, to_categorical
import numpy as np
import util


class Generator(Sequence):
    def __init__(self, n, batch_size, label_type, shuffle):
        self.n = n
        self.batch_size = batch_size
        self.label_type = label_type
        self.shuffle = shuffle
        self.samples = []
        self.index_array = np.arange(n)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __getitem__(self, i):
        if i >= len(self):
            raise ValueError(f'Asked to retrieve element {i}, but the Sequence has length {len(self)}')
        return np.array(self.samples[self.index_array[self.batch_size*i:self.batch_size*(i+1)]])

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.index_array)


class CubeGenerator(Generator):
    def __init__(self, n=1000, batch_size=1, label_type=None, shuffle=True, image_size=32):
        super().__init__(n, batch_size, label_type, shuffle)
        self.image_size = image_size
        self.samples = [self._generate_sample() for i in range(n)]

    def _generate_sample(self):
        pose = np.random.rand(3)
        rot = util.rotation_matrix(*pose)
        sample = util.draw_cube(rot, image_size=self.image_size)

        if self.label_type == 'pose':
            return (sample, pose)
        if self.label_type == 'input':
            return (sample, sample)
        return sample


class MNISTGenerator(Generator):
    def __init__(self, n=None, batch_size=1, label_type=None, shuffle=True, partition='train'):
        (x_train_all, y_train_all), (x_test, y_test) = mnist.load_data()
        split_index = len(x_train_all) * 9 // 10
        if partition == 'train':
            x = x_train_all[:split_index] / 255
            y = y_train_all[:split_index]
        elif partition == 'val':
            x = x_train_all[split_index:] / 255
            y = y_train_all[split_index:]
        elif partition == 'test':
            x = x_test / 255
            y = y_test
        else:
            raise ValueError(f'Partition {partition} not valid.')

        super().__init__(len(x), batch_size, label_type, shuffle)

        x = np.pad(x, ((0,0), (2,2), (2,2)), 'constant')[...,np.newaxis] # pad with 0s to 32 x 32
        y = to_categorical(y, num_classes=10)

        if self.label_type == 'digit':
            self.samples = list(zip(x, y))
        elif self.label_type == 'input':
            self.samples = list(zip(x, x))
        else:
            self.samples = x
