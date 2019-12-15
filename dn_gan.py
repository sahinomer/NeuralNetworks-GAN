import os
from datetime import datetime

import numpy as np

from dataset import Dataset
from keras import Input, Model
from keras.layers import Dense, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, \
    Flatten, Dropout, Activation
from keras.optimizers import Adam

from noisy_samples import NoisySamples
from utils import performance


def build_adversarial(generator_model, discriminator_model):
    discriminator_model.trainable = False

    # generator.output -> discriminator.input
    gan_output = discriminator_model(generator_model.output)

    model = Model(generator_model.input, gan_output)

    model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
                  loss='binary_crossentropy')

    return model


def build_generator(input_shape):
    noisy_input = Input(shape=input_shape)

    gen = Conv2DTranspose(128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                          kernel_initializer='glorot_normal')(noisy_input)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Dense(256)(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Dense(512)(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Dense(512)(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Dense(128)(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv2DTranspose(3, kernel_size=(3, 3), strides=(1, 1), padding='same',
                          kernel_initializer='glorot_normal')(gen)

    gen = Activation('tanh')(gen)

    model = Model(noisy_input, gen)
    return model


def build_discriminator(input_shape):
    input_data = Input(shape=input_shape)

    cnn = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(input_data)
    cnn = LeakyReLU(alpha=0.2)(cnn)
    cnn = Dropout(0.4)(cnn)

    cnn = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(cnn)
    cnn = LeakyReLU(alpha=0.2)(cnn)
    cnn = Dropout(0.4)(cnn)

    cnn = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(cnn)
    cnn = LeakyReLU(alpha=0.2)(cnn)
    cnn = Dropout(0.4)(cnn)

    cnn = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(cnn)
    cnn = LeakyReLU(alpha=0.2)(cnn)
    cnn = Dropout(0.4)(cnn)

    cnn = Flatten()(cnn)

    out = Dense(1, activation='sigmoid')(cnn)

    model = Model(input_data, out)
    model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
                  loss='binary_crossentropy')

    return model


#######################################################################################################################


class DenoiseGAN:

    def __init__(self, data_shape):
        self.data_shape = data_shape
        self.discriminator = None
        self.generator = None
        self.adversarial = None

        self.define_gan()
        self.noisy_samples = NoisySamples(shape=self.data_shape, noise_type='s&p')

        self.performance_output_path = 'performance/siamese_dn_gan_' + str(datetime.now().date())

    def define_gan(self):
        self.generator = build_generator(input_shape=self.data_shape)
        self.discriminator = build_discriminator(input_shape=self.data_shape)

        self.adversarial = build_adversarial(generator_model=self.generator,
                                             discriminator_model=self.discriminator)

    def train(self, dataset, epochs=100, batch_size=64):

        for e in range(epochs):
            print('Epochs: %3d/%d' % (e, epochs))
            self.single_epoch(dataset, batch_size)
            performance(model=self, epoch=e, test_data=dataset.test_data)

    def single_epoch(self, dataset, batch_size):
        half_batch_size = int(batch_size / 2)
        trained_samples = 0

        for realX, _ in dataset.iter_samples(batch_size):
            noisy = self.noisy_samples.add_noise(real_samples=realX)
            fakeX = self.generator.predict(noisy)
            X = np.vstack([realX, fakeX])

            realY = np.ones(shape=(len(realX),))
            fakeY = np.zeros(shape=(len(fakeX),))
            Y = np.hstack([realY, fakeY])

            discriminator_loss = self.discriminator.train_on_batch(X, Y)

            noisy = self.noisy_samples.add_noise(realX)
            act_real = np.ones(shape=(len(noisy),))

            gan_loss = self.adversarial.train_on_batch(noisy, act_real)

            trained_samples = min(trained_samples+half_batch_size, dataset.sample_number)
            print('     %5d/%d -> Discriminator Loss: %f, Gan Loss: %f'
                  % (trained_samples, dataset.sample_number, discriminator_loss, gan_loss))


if __name__ == '__main__':
    dataset = Dataset(dataset='caltech256')
    dataset.split_test_data(test_sample=100)
    gan = DenoiseGAN(data_shape=dataset.data_shape)
    gan.train(dataset=dataset, batch_size=32, epochs=20)
