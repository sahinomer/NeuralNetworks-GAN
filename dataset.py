
import numpy as np
from keras.datasets import cifar10
import caltech256


class Dataset:

    def __init__(self, dataset='cifar10'):
        """
        Initialize dataset class, assign data shape, load dataset
        :param dataset: Dataset name 'cifar10' or 'caltech256'
        """
        self.data = None
        self.label = None
        self.data_shape = None
        self.sample_number = None
        self.test_data = None

        self.load_dataset(dataset=dataset)

    def load_dataset(self, dataset='cifar10'):
        """
        Load given dataset and scale from [0,255] to [-1,1]
        :param dataset: Dataset name 'cifar10' or 'caltech256'
        """

        if dataset == 'cifar10':
            (trainX, trainY), (testX, testY) = cifar10.load_data()
            x = np.concatenate([trainX, testX], axis=0)
            y = np.concatenate([trainY, testY], axis=0)

        elif dataset == 'caltech256':
            x, y = caltech256.load_data(width=128, height=128)

        else:
            raise Exception('Unknown dataset!')

        # convert from ints to floats
        x = x.astype('float32')
        # scale from [0,255] to [-1,1]
        x = (x - 127.5) / 127.5

        self.data = x
        self.label = y
        self.data_shape = x[0].shape
        self.sample_number = len(x)

    def iter_samples(self, sample_number):
        """
        Iterate data up to sample number
        :param sample_number: Batch size for iteration
        :return: Image data and labels
        """
        start = 0
        while start < self.sample_number:
            end = start + sample_number
            yield self.data[start:end], self.label[start:end]
            start = end

    def split_test_data(self, test_sample=-1):
        """
        Split test data and assign to test attribute of the class
        :param test_sample: Number of test samples or test class.
                            If test_sample less than 10, select an image class as the test set,
                            otherwise, select the first test_sample samples as the test set.
        """

        if test_sample > 10:  # more than class number
            # test_indices = np.random.randint(0, self.sample_number, size=test_sample)
            test_indices = np.arange(test_sample)

        else:
            test_indices = np.where(self.label == test_sample)[0]

        test_size = len(test_indices)
        self.test_data = self.data[test_indices]

        mask = np.ones(self.sample_number, bool)
        mask[test_indices] = False

        self.data = self.data[mask]
        self.label = self.label[mask]

        self.sample_number -= test_size
