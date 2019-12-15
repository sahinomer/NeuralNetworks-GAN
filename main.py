import tensorflow as tf
from keras import backend as K

from dataset import Dataset
from dn_gan import DenoiseGAN
from siamese_dn_gan import SiameseDenoiseGAN

K.clear_session()
tf.keras.backend.clear_session()

if __name__ == '__main__':

    dataset = Dataset(dataset='cifar10')
    dataset.split_test_data(test_sample=0)
    # gan = DenoiseGAN(data_shape=dataset.data_shape)
    gan = SiameseDenoiseGAN(data_shape=dataset.data_shape)
    gan.train(dataset=dataset, batch_size=64, epochs=20)
