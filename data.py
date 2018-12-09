from keras.datasets import cifar10, mnist
from keras.utils import Sequence, to_categorical
import numpy as np
import util


class Generator(Sequence):
    def __init__(self, n, batch_size, decoder, include_label, shuffle):
        self.n = n
        self.batch_size = batch_size
        self.decoder = decoder
        self.include_label = include_label
        self.shuffle = shuffle
        self.samples = []
        self.labels = []
        self.index_array = np.arange(n)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __getitem__(self, i):
        if i >= len(self):
            raise ValueError(f'Asked to retrieve element {i}, but the Sequence has length {len(self)}')
        indices = self.index_array[self.batch_size*i:self.batch_size*(i+1)]
        sample = self.samples[indices]
        label = self.labels[indices]

        if self.decoder:
            new_sample = (sample, label)
            label = (label, sample)
            sample = new_sample

        if self.include_label:
            return (sample, label)
        return sample

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.index_array)


class CIFARGenerator(Generator):
    def __init__(self, partition, batch_size=10, decoder=False, include_label=True, shuffle=False):
        (x_train_all, y_train_all), (x_test, y_test) = cifar10.load_data()
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

        super().__init__(len(x), batch_size, decoder, include_label, shuffle)

        self.samples = x
        self.labels = to_categorical(y, num_classes=10)


class MNISTGenerator(Generator):
    def __init__(self, partition, batch_size=10, decoder=False, include_label=True, shuffle=False):
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

        super().__init__(len(x), batch_size, decoder, include_label, shuffle)

        self.samples = np.pad(x, ((0,0), (2,2), (2,2)), 'constant')[...,np.newaxis] # pad with 0s to 32 x 32
        self.labels = to_categorical(y, num_classes=10)
