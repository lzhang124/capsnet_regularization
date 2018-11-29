from keras.utils import Sequence
from keras.datasets import mnist
import numpy as np


class MNISTGenerator(Sequence):
    def __init__(self, partition='train', batch_size=1, label_type=None, shuffle=True):
        self.batches = []
        self.batch_size = batch_size
        self.label_type = label_type
        self.shuffle = shuffle

        (x_train_all, y_train_all), (x_test, y_test) = mnist.load_data()

        num_training = len(x_train_all)
        split_index = num_training*9//10
        if partition == 'train':
            x = x_train_all[:split_index]/255
            y = y_train_all[:split_index]
        elif partition == 'validation':
            x = x_train_all[split_index:]/255
            y = y_train_all[split_index:]
        elif partition == 'test':
            x = x_test/255
            y = y_test
        else:
            raise ValueError(f'Partition {partition} not valid')

        self.n = len(x)
        x = np.pad(x, ((0,0), (2,2), (2,2)), 'constant')[...,np.newaxis] # pad with 0s to 32 x 32

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

    def __getitem__(self, i):
        if i >= len(self):
            raise ValueError(f'Asked to retrieve element {i}, but the Sequence has length {len(self)}')
        return self.batches[self.index_array[i]]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.index_array)
