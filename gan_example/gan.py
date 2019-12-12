import os

import numpy as np
from matplotlib import pyplot

from dataset import Dataset
from fake_samples import FakeSamples
from keras import Input, Model
from keras.layers import Dense, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, \
    Flatten, Reshape, Dropout, Activation
from keras.optimizers import Adam

from keras.initializers import RandomNormal


def build_adversarial(generator_model, discriminator_model):
    discriminator_model.trainable = False

    # generator.output -> discriminator.input
    gan_output = discriminator_model(generator_model.output)

    model = Model(generator_model.input, gan_output)

    model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
                  loss='binary_crossentropy')

    return model


def build_generator(latent_size):
    noise_input = Input(shape=(latent_size,))

    gen = Dense(3 * 3 * 384, activation='relu')(noise_input)
    gen = Reshape((3, 3, 384))(gen)

    # 3*3*384 -> 7*7*192
    gen = Conv2DTranspose(filters=192, kernel_size=(5, 5), strides=(1, 1), padding='valid',
                          kernel_initializer=RandomNormal(stddev=0.02))(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    # 7*7*192 -> 14*14*96
    gen = Conv2DTranspose(filters=96, kernel_size=(5, 5), strides=(2, 2), padding='same',
                          kernel_initializer=RandomNormal(stddev=0.02))(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    # 14*14*96 -> 28*28*48
    gen = Conv2DTranspose(filters=48, kernel_size=(5, 5), strides=(2, 2), padding='same',
                          kernel_initializer=RandomNormal(stddev=0.02))(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    # 28*28*48 -> 28*28*1
    gen = Conv2DTranspose(filters=1, kernel_size=(5, 5), strides=(1, 1), padding='same',
                          kernel_initializer=RandomNormal(stddev=0.02))(gen)

    gen = Activation('tanh')(gen)

    model = Model(noise_input, gen)
    return model


def build_discriminator(input_shape):
    input_data = Input(shape=input_shape)

    cnn = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same')(input_data)
    cnn = LeakyReLU(alpha=0.2)(cnn)
    cnn = Dropout(0.4)(cnn)

    cnn = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(cnn)
    cnn = LeakyReLU(alpha=0.2)(cnn)
    cnn = Dropout(0.4)(cnn)

    cnn = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(cnn)
    cnn = LeakyReLU(alpha=0.2)(cnn)
    cnn = Dropout(0.4)(cnn)

    cnn = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(cnn)
    cnn = LeakyReLU(alpha=0.2)(cnn)
    cnn = Dropout(0.4)(cnn)

    cnn = Flatten()(cnn)

    out = Dense(1, activation='sigmoid')(cnn)

    model = Model(input_data, out)
    model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
                  loss='binary_crossentropy')

    return model


#######################################################################################################################


class GAN:

    def __init__(self, data_shape, latent_size):
        self.data_shape = data_shape
        self.latent_size = latent_size
        self.discriminator = None
        self.generator = None
        self.adversarial = None

        self.define_gan()
        self.fake_samples = FakeSamples(generator=self.generator, latent_size=latent_size)

        self.performance_output_path = 'performance/temp/'
        if not os.path.exists(self.performance_output_path):
            os.makedirs(self.performance_output_path)

    def define_gan(self):
        self.generator = build_generator(latent_size=self.latent_size)
        self.discriminator = build_discriminator(input_shape=self.data_shape)

        self.adversarial = build_adversarial(generator_model=self.generator,
                                             discriminator_model=self.discriminator)

    def train(self, dataset, epochs=100, batch_size=64):

        for e in range(epochs):
            print('Epochs: %3d/%d' % (e, epochs))
            self.single_epoch(dataset, batch_size)
            self.performance(step=e)

    def single_epoch(self, dataset, batch_size):
        half_batch_size = int(batch_size / 2)
        trained_samples = 0

        for realX, _, realY in dataset.iter_samples(half_batch_size):
            fakeX, fakeY = self.fake_samples.generate_fake_samples(sample_number=half_batch_size)

            X = np.vstack([realX, fakeX])
            Y = np.hstack([realY, fakeY])

            discriminator_loss = self.discriminator.train_on_batch(X, Y)

            noise_input = self.fake_samples.generate_latent_points(batch_size)
            act_real = np.ones(shape=(batch_size,))

            gan_loss = self.adversarial.train_on_batch(noise_input, act_real)

            trained_samples = min(trained_samples+half_batch_size, dataset.sample_number)
            print('     %5d/%d -> Discriminator Loss: %f, Gan Loss: %f'
                  % (trained_samples, dataset.sample_number, discriminator_loss, gan_loss))

    def performance(self, step):
        # prepare fake examples
        generated, _ = self.fake_samples.generate_fake_samples(sample_number=100)
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


if __name__ == '__main__':
    dataset = Dataset()
    gan = GAN(data_shape=(28, 28, 1), latent_size=100)
    gan.train(dataset=dataset, batch_size=64, epochs=50)
