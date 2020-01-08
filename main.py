import tensorflow as tf
from keras import backend as K

from dataset import Dataset
from dn_gan import DenoiseGAN
from siamese_dn_gan import SiameseDenoiseGAN


def clear_session():
    """
    Clear Keras and Tensorflow session
    """
    K.clear_session()
    tf.keras.backend.clear_session()


if __name__ == '__main__':
    # Load dataset and split test data
    dataset = Dataset(dataset='caltech256')
    dataset.split_test_data(test_sample=2000)

    # Denoise GAN
    clear_session()
    gan = DenoiseGAN(data_shape=dataset.data_shape)
    gan.train(dataset=dataset, batch_size=32, epochs=20)

    # Siamese Denoise GAN
    clear_session()
    gan = SiameseDenoiseGAN(data_shape=dataset.data_shape)
    gan.train(dataset=dataset, batch_size=32, epochs=20)
