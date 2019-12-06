
import numpy as np
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from keras.datasets import cifar10


class Dataset:

    def __init__(self):
        self.data = None
        self.label = None
        self.real = None
        self.data_shape = None
        self.sample_number = None
        self.test_data = None

        self.load_dataset()

    def load_dataset(self):
        # load dataset
        (trainX, trainY), (testX, testY) = cifar10.load_data()
        # (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
        x = np.concatenate([trainX, testX], axis=0)
        y = np.concatenate([trainY, testY], axis=0)
        # expand to 3d, e.g. add channels
        # x = np.expand_dims(x, axis=-1)
        # convert from ints to floats
        x = x.astype('float32')
        # scale from [0,255] to [-1,1]
        x = (x - 127.5) / 127.5

        self.data = x
        self.label = y
        self.real = np.ones(shape=(len(x), ), dtype=np.int)
        self.data_shape = x[0].shape
        self.sample_number = len(x)

    def iter_samples(self, sample_number):
        start = 0
        while start < self.sample_number:
            end = start + sample_number
            yield self.data[start:end], self.label[start:end], self.real[start:end]
            start = end

    def split_test_data(self, test_class=0):
        test_indices = np.where(self.label == test_class)[0]
        test_size = len(test_indices)
        self.test_data = self.data[test_indices]

        mask = np.ones(self.sample_number, bool)
        mask[test_indices] = False

        self.data = self.data[mask]
        self.label = self.label[mask]
        self.real = self.real[mask]

        self.sample_number -= test_size
