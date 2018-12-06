from keras.datasets import mnist
from keras.utils import Sequence, to_categorical
import numpy as np
import util


class Generator(Sequence):
    def __init__(self, batch_size, label_type, shuffle, capsule=False):
        self.batch_size = batch_size
        self.label_type = label_type
        self.shuffle = shuffle
        self.capsule = capsule
        self.batches = []

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, i):
        if i >= len(self):
            raise ValueError(f'Asked to retrieve element {i}, but the Sequence has length {len(self)}')
        batch = self.batches[self.index_array[i]]
        if self.capsule:
            batch = [b[np.newaxis,...] for b in batch]
        return batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.index_array)


class CubeGenerator(Generator):
    def __init__(self, batch_size=1, label_type=None, shuffle=True, capsule=False, n=1000, image_size=32):
        super().__init__(batch_size, label_type, shuffle, capsule)
        self.n = n
        self.image_size = image_size
        self.index_array = np.arange(len(self))
        self.batches = [self._generate_batch() for i in range(len(self))]

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def _generate_batch(self):
        batch = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        poses = np.zeros((self.batch_size, 3))
        for i in range(self.batch_size):
            x1, x2, x3 = np.random.rand(3)
            rot = util.rotation_matrix(x1, x2, x3)
            batch[i] = util.draw_cube(rot, image_size=self.image_size)
            poses[i] = x1, x2, x3

        if self.label_type == 'pose':
            return (batch, poses)
        if self.label_type == 'input':
            return (batch, batch)
        return batch


class MNISTGenerator(Generator):
    def __init__(self, batch_size=1, label_type=None, shuffle=True, capsule=False, partition='train'):
        super().__init__(batch_size, label_type, shuffle, capsule)
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

        self.n = len(x)
        x = np.pad(x, ((0,0), (2,2), (2,2)), 'constant')[...,np.newaxis] # pad with 0s to 32 x 32
        y = to_categorical(y, num_classes=10)

        num_batches = int(np.ceil(len(x) / self.batch_size))
        batches = np.array_split(x, num_batches)
        digits = np.array_split(y, num_batches)

        if self.label_type == 'digit':
            self.batches = list(zip(batches, digits))
        elif self.label_type == 'input':
            self.batches = list(zip(batch, batch))
        else:
            self.batches = batches

        self.index_array = np.arange(len(self))

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))
