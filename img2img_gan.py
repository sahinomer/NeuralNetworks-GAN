import numpy as np
from matplotlib import pyplot

from dataset import Dataset
from keras import Input, Model
from keras.layers import Dense, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, \
    Flatten, Reshape, Dropout
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

    cnn = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same')(noisy_input)
    cnn = LeakyReLU(alpha=0.2)(cnn)

    cnn = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(cnn)
    cnn = LeakyReLU(alpha=0.2)(cnn)

    # upsample to (14, 14, ...)
    gen = Conv2DTranspose(32, kernel_size=(5, 5), strides=(2, 2), padding='same',
                          activation='relu',
                          kernel_initializer='glorot_normal')(cnn)
    gen = BatchNormalization()(gen)

    # upsample to (28, 28, ...)
    gen = Conv2DTranspose(1, kernel_size=(5, 5), strides=(2, 2), padding='same',
                          activation='tanh',
                          kernel_initializer='glorot_normal')(gen)

    model = Model(noisy_input, gen)
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


class Img2ImgGAN:

    def __init__(self, data_shape):
        self.data_shape = data_shape
        self.discriminator = None
        self.generator = None
        self.adversarial = None

        self.define_gan()
        self.noisy_samples = NoisySamples(generator=self.generator)

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
            fakeX, fakeY = self.noisy_samples.denoise_samples(real_samples=realX)

            X = np.vstack([realX, fakeX])
            Y = np.hstack([realY, fakeY])

            discriminator_loss = self.discriminator.train_on_batch(X, Y)

            noisy_input = self.noisy_samples.add_noise(realX)
            act_real = np.ones(shape=(batch_size, 1))

            gan_loss = self.adversarial.train_on_batch(noisy_input, act_real)

            trained_samples += half_batch_size
            print('     %5d/%d -> Discriminator Loss: %f, Gan Loss: %f'
                  % (trained_samples, dataset.sample_number, discriminator_loss, gan_loss))

    def performance(self, step, test_data, path='temp/'):
        # prepare fake examples
        generated, _ = self.noisy_samples.denoise_samples(real_samples=test_data)
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
        fig_file = path + 'generated_plot_%04d.png' % (step + 1)
        pyplot.savefig(fig_file)
        pyplot.close()
        # save the generator model
        model_file = path + 'model_%04d.h5' % (step + 1)
        self.generator.save(model_file)
        print('>Saved: %s and %s' % (fig_file, model_file))


if __name__ == '__main__':
    dataset = Dataset()
    gan = Img2ImgGAN(data_shape=(28, 28, 1))
    gan.train(dataset=dataset, batch_size=64, epochs=50)
