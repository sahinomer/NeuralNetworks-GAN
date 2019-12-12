import os

import numpy as np
from matplotlib import pyplot

from dataset import Dataset
from keras import Input, Model, Sequential
from keras.layers import Dense, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, \
    Flatten, Reshape, Dropout, Activation, subtract, multiply, dot, concatenate
from keras.optimizers import Adam

from noisy_samples import NoisySamples

import keras.backend as K


def triplet_loss(y_true, y_pred):
    _alpha = 0.9 * y_true
    embeddings = K.reshape(y_pred, (-1, 3, 9408))

    positive_distance = K.mean(K.square(embeddings[:, 0] - embeddings[:, 1]), axis=-1)
    negative_distance = K.mean(K.square(embeddings[:, 0] - embeddings[:, 2]), axis=-1)
    return K.mean(K.maximum(0.0, positive_distance - negative_distance + _alpha))


def build_adversarial(generator_model, discriminator_model):
    discriminator_model.trainable = False

    real_input = Input(shape=(28, 28, 1))
    noisy_input = Input(shape=(28, 28, 1))

    # generator.output -> discriminator.input
    gan_output = discriminator_model([real_input, generator_model.output, noisy_input])

    model = Model([real_input, generator_model.input, noisy_input], gan_output)

    model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
                  loss=triplet_loss)

    return model


def build_generator(input_shape):
    noisy_input = Input(shape=input_shape)

    gen = Conv2DTranspose(64, kernel_size=(5, 5), strides=(1, 1), padding='same',
                          kernel_initializer='glorot_normal')(noisy_input)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Flatten()(gen)

    gen = Dense(28*28*1)(gen)

    gen = Reshape((28, 28, 1))(gen)

    gen = Conv2DTranspose(64, kernel_size=(5, 5), strides=(1, 1), padding='same',
                          kernel_initializer='glorot_normal')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv2DTranspose(1, kernel_size=(5, 5), strides=(1, 1), padding='same',
                          kernel_initializer='glorot_normal')(gen)

    gen = Activation('tanh')(gen)

    model = Model(noisy_input, gen)
    return model


def build_discriminator(input_shape):
    real_input = Input(shape=input_shape)
    fake_input = Input(shape=input_shape)
    noisy_input = Input(shape=input_shape)

    cnn = Sequential(layers=[
        Conv2D(24, kernel_size=(3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(alpha=0.2),
        Dropout(0.4),

        Conv2D(48, kernel_size=(3, 3), strides=(1, 1), padding='same'),
        LeakyReLU(alpha=0.2),
        Dropout(0.4),

        Conv2D(96, kernel_size=(3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(alpha=0.2),
        Dropout(0.4),

        Conv2D(192, kernel_size=(3, 3), strides=(1, 1), padding='same'),
        LeakyReLU(alpha=0.2),
        Dropout(0.4),

        Flatten()
    ])

    real_cnn = cnn(real_input)
    fake_cnn = cnn(fake_input)
    noisy_cnn = cnn(noisy_input)

    out = concatenate([real_cnn, fake_cnn, noisy_cnn])

    # subtract_inputs = subtract([real_cnn, fake_cnn])
    # multiply_inputs = multiply([real_cnn, fake_cnn])
    # dot_inputs = dot([real_cnn, fake_cnn], axes=1, normalize=True)
    #
    # merged = concatenate([dot_inputs, subtract_inputs, multiply_inputs])
    #
    # dense = BatchNormalization()(merged)
    # dense = Dense(128, activation='relu')(dense)
    # dense = BatchNormalization()(dense)
    # dense = Dropout(0.4)(dense)
    # dense = Dense(128, activation='relu')(dense)
    # dense = BatchNormalization()(dense)
    # dense = Dropout(0.4)(dense)
    # out = Dense(1, activation='sigmoid')(dense)

    model = Model([real_input, fake_input, noisy_input], out)
    model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
                  loss=triplet_loss)

    return model


#######################################################################################################################


class SiameseGAN:

    def __init__(self, data_shape):
        self.data_shape = data_shape
        self.discriminator = None
        self.generator = None
        self.adversarial = None

        self.define_gan()
        self.noisy_samples = NoisySamples(generator=self.generator)

        self.performance_output_path = 'performance/temp/'
        if not os.path.exists(self.performance_output_path):
            os.makedirs(self.performance_output_path)

    def define_gan(self):
        self.generator = build_generator(input_shape=self.data_shape)
        self.discriminator = build_discriminator(input_shape=self.data_shape)

        self.adversarial = build_adversarial(generator_model=self.generator,
                                             discriminator_model=self.discriminator)

    def train(self, dataset, epochs=100, batch_size=64):

        for e in range(epochs):
            print('Epochs: %3d/%d' % (e, epochs))
            self.single_epoch(dataset, batch_size)
            self.performance(step=e, test_data=dataset.test_data)

    def single_epoch(self, dataset, batch_size):
        half_batch_size = int(batch_size / 2)
        trained_samples = 0

        for realX, _, realY in dataset.iter_samples(half_batch_size):
            Y = np.ones(shape=(len(realX),))

            fakeX, fakeY, noiseX = self.noisy_samples.denoise_samples(real_samples=realX)

            Y = np.zeros(shape=(len(realX),))
            discriminator_loss = self.discriminator.train_on_batch([realX, fakeX, noiseX], Y)

            noisy_input = self.noisy_samples.add_noise(realX)
            act_real = np.ones(shape=(len(noisy_input),))

            gan_loss = self.adversarial.train_on_batch([realX, noisy_input, noisy_input], act_real)

            trained_samples = min(trained_samples+half_batch_size, dataset.sample_number)
            print('     %5d/%d -> Discriminator Loss: %f, Gan Loss: %f'
                  % (trained_samples, dataset.sample_number, discriminator_loss, gan_loss))

    def performance(self, step, test_data):
        # prepare fake examples
        generated, _, _ = self.noisy_samples.denoise_samples(real_samples=test_data)
        # scale from [-1,1] to [0,1]
        generated = (generated + 1) / 2.0
        # plot images
        for i in range(100):
            # define subplot
            pyplot.subplot(10, 10, 1 + i)
            # turn off axis
            pyplot.axis('off')
            # plot raw pixel data
            pyplot.imshow(generated[i, :, :, 0], cmap='gray_r')
        # save plot to file
        fig_file = self.performance_output_path + 'generated_plot_%04d.png' % (step + 1)
        pyplot.savefig(fig_file)
        pyplot.close()
        # save the generator model
        model_file = self.performance_output_path + 'model_%04d.h5' % (step + 1)
        self.generator.save(model_file)
        print('>Saved: %s and %s' % (fig_file, model_file))


def plot_images(images):
    images = (images + 1) / 2.0
    for i in range(100):
        # define subplot
        pyplot.subplot(10, 10, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i, :, :, 0], cmap='gray_r')
        # save plot to file
    pyplot.show()


if __name__ == '__main__':
    dataset = Dataset()
    dataset.split_test_data(test_size=100)
    gan = SiameseGAN(data_shape=(28, 28, 1))
    gan.train(dataset=dataset, batch_size=64, epochs=50)
