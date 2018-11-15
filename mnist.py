from keras.utils import Sequence
from keras.datasets import mnist
import numpy as np

class MNISTGenerator(Sequence):
    def __init__(self, batch_size=1, label_type=None, partition='train'):
        self.batches = []
        self.batch_size = batch_size
        self.label_type = label_type

        (x_train_all, y_train_all), (x_test, y_test) = mnist.load_data()

        num_training = len(x_train_all)
        split_index = num_training*9//10
        if partition == 'train':
            x_train = x_train_all[:split_index]
            y_train = y_train_all[:split_index]
            num_batches = int(np.ceil(len(x_train) / self.batch_size))
            self.batches = np.array_split(x_train, num_batches)
        elif partition == 'validation':
            x_val = x_train_all[split_index:]
            y_val = y_train_all[split_index:]
            num_batches = int(np.ceil(len(x_val) / self.batch_size))
            self.batches = np.array_split(x_val, num_batches)
        elif partition == 'test':
            num_batches = int(np.ceil(len(x_test) / self.batch_size))
            self.batches = np.array_split(x_test, num_batches)
        else:
            raise ValueError('Partition {} not valid'.format(partition))

        self.index_array = np.arange(len(self))

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        if i >= len(self):
            raise ValueError('Asked to retrieve element {}, but the Sequence has length {}'.format(i, len(self)))
        return self.batches[self.index_array[i]]

    def on_epoch_end(self):
        np.random.shuffle(self.index_array)
