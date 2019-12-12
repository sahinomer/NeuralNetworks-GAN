import os

import numpy as np
from matplotlib import pyplot

from dataset import Dataset
from keras import Input, Model
from keras.layers import Dense, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, \
    Flatten, Dropout, Activation
from keras.optimizers import Adam

from noisy_samples import NoisySamples


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

    gen = Conv2DTranspose(64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                          kernel_initializer='glorot_normal')(noisy_input)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Dense(128)(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Dense(256)(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Dense(512)(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Dense(64)(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv2DTranspose(1, kernel_size=(3, 3), strides=(1, 1), padding='same',
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


class Img2ImgGAN:

    def __init__(self, data_shape):
        self.data_shape = data_shape
        self.discriminator = None
        self.generator = None
        self.adversarial = None

        self.define_gan()
        self.noisy_samples = NoisySamples(generator=self.generator, shape=self.data_shape, noise_type='s&p')

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
            fakeX, fakeY, _ = self.noisy_samples.denoise_samples(real_samples=realX)
            X = np.vstack([realX, fakeX])
            Y = np.hstack([realY, fakeY])

            discriminator_loss = self.discriminator.train_on_batch(X, Y)

            noisy_input = self.noisy_samples.add_noise(realX)
            act_real = np.ones(shape=(len(noisy_input),))

            gan_loss = self.adversarial.train_on_batch(noisy_input, act_real)

            trained_samples = min(trained_samples+half_batch_size, dataset.sample_number)
            print('     %5d/%d -> Discriminator Loss: %f, Gan Loss: %f'
                  % (trained_samples, dataset.sample_number, discriminator_loss, gan_loss))

    def performance(self, step, test_data):

        sub_test_data = test_data[step*50:(step+1)*50]

        # prepare fake examples
        generated, _, noise = self.noisy_samples.denoise_samples(real_samples=sub_test_data)

        # save plot to file
        fig_file = self.performance_output_path + 'epoch-%04d_plot.png' % (step + 1)
        data_triplet = np.concatenate([sub_test_data, noise, generated], axis=2)
        plot_images(data_triplet, path=fig_file)

        # save the generator model
        model_file = self.performance_output_path + 'model_%04d.h5' % (step + 1)
        self.generator.save(model_file)
        print('>Saved: %s and %s' % (fig_file, model_file))


def plot_images(images, path=None):
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    for i in range(50):
        # define subplot
        pyplot.subplot(10, 5, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i, :, :, 0], cmap='gray_r')
        # save plot to file

    if path:
        pyplot.savefig(path)
        pyplot.close()
    else:
        pyplot.show()


if __name__ == '__main__':
    dataset = Dataset()
    dataset.split_test_data(test_size=500)
    gan = Img2ImgGAN(data_shape=(28, 28, 1))
    gan.train(dataset=dataset, batch_size=64, epochs=10)
