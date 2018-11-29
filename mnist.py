from keras.utils import Sequence
from keras.datasets import mnist
import numpy as np

class MNISTGenerator(Sequence):
    def __init__(self, batch_size=1, label_type=None, partition='train', shuffle=True):
        self.batches = []
        self.batch_size = batch_size
        self.label_type = label_type
        self.shuffle = shuffle

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
            raise ValueError(f'Partition {partition} not valid')

        self.index_array = np.arange(len(self))

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        if i >= len(self):
            raise ValueError(f'Asked to retrieve element {i}, but the Sequence has length {len(self)}')
        return self.batches[self.index_array[i]]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.index_array)
