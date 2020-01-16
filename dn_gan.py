import os
from datetime import datetime

import numpy as np

from keras import Input, Model
from keras.layers import Dense, Conv2D, LeakyReLU, BatchNormalization, \
    Flatten, Dropout, Activation
from keras.optimizers import Adam

from noise_maker import NoiseMaker
from utils import measure_and_plot, mean_ssim


def build_adversarial(generator_model, discriminator_model):
    """
    Adversarial Model builder
        Input -> Generator -> Discriminator(static, untrainable) -> Real/Fake
    """
    discriminator_model.trainable = False

    # generator.output -> discriminator.input
    gan_output = discriminator_model(generator_model.output)

    model = Model(generator_model.input, gan_output)

    model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
                  loss='binary_crossentropy')

    return model


def build_generator(input_shape):
    """
    Generator Model Builder
        Input(noisy image) -> Convolution2D(128) -> Dense(128)
                    -> Dense(256) -> Dense(512) -> Dense(64) -> Convolution2D(3)
    """
    noisy_input = Input(shape=input_shape)

    gen = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same',
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

    gen = Conv2D(3, kernel_size=(3, 3), strides=(1, 1), padding='same',
                 kernel_initializer='glorot_normal')(gen)

    gen = Activation('tanh')(gen)

    model = Model(noisy_input, gen)
    return model


def build_discriminator(input_shape):
    """
    Build Discriminator Model
        Input -> Convolution2D(32) -> Convolution2D(64) -> Convolution2D(128)
                    -> Convolution2D(256) -> Output(1) : Real or Fake
    """
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
        """
        Initialize DN-GAN
        """
        self.data_shape = data_shape
        self.discriminator = None
        self.generator = None
        self.adversarial = None

        self.define_gan()
        self.noise_maker = NoiseMaker(shape=self.data_shape, noise_type='s&p')

        self.performance_output_path = 'performance/dn_gan_' + str(datetime.now().date())

    def define_gan(self):
        """
        Build Models
            Generative model
            Discriminative model
            Adversarial model
        """
        self.generator = build_generator(input_shape=self.data_shape)
        self.discriminator = build_discriminator(input_shape=self.data_shape)

        self.adversarial = build_adversarial(generator_model=self.generator,
                                             discriminator_model=self.discriminator)

    def train(self, dataset, epochs=20, batch_size=64):
        """
        Train model
        """
        for e in range(epochs):
            print('Epochs: %3d/%d' % (e+1, epochs))
            self.single_epoch(dataset, batch_size)
            self.performance(epoch=e, test_data=dataset.test_data)

    def single_epoch(self, dataset, batch_size):
        """
        Single iteration/epoch
            Iterate dataset as batch size
        """
        trained_samples = 0

        for realX, _ in dataset.iter_samples(batch_size):
            # Add noise to images and denoise them
            noisy = self.noise_maker.add_noise(real_samples=realX)
            fakeX = self.generator.predict(noisy)
            X = np.vstack([realX, fakeX])

            realY = np.ones(shape=(len(realX),))    # Real images with label 1
            fakeY = np.zeros(shape=(len(fakeX),))   # Denoised images with label 0
            Y = np.hstack([realY, fakeY])

            # Train discriminative model with real and denoised images
            discriminator_loss = self.discriminator.train_on_batch(X, Y)

            # Add noise to images
            noisy = self.noise_maker.add_noise(realX)
            act_real = np.ones(shape=(len(noisy),))

            # Train adversarial model with denoised images that are labeled like real images
            gan_loss = self.adversarial.train_on_batch(noisy, act_real)

            trained_samples = min(trained_samples+batch_size, dataset.sample_number)
            print('     %5d/%d -> Discriminator Loss: %f, Gan Loss: %f'
                  % (trained_samples, dataset.sample_number, discriminator_loss, gan_loss))

    def performance(self, epoch, test_data):
        """
        Measure performance of model at each iteration
        """
        path = self.performance_output_path + '/epoch-%04d' % (epoch + 1)
        if not os.path.exists(path):
            os.makedirs(path)

        # Average SSIM index on test set
        mean_ssim(epoch, test_data, self.noise_maker, self.generator, self.performance_output_path + '/result.txt')

        test_data = test_data[epoch * 100:(epoch + 1) * 100]

        # Generate denoised samples - add noise test images and denoise
        noisy = self.noise_maker.add_noise(real_samples=test_data)
        generated = self.generator.predict(noisy)

        # Save the generator model
        model_file = path + '/model_%04d.h5' % (epoch + 1)
        self.generator.save(model_file)

        # Save figures
        fig_file = path + '/plot_%04d' % ((epoch + 1))
        measure_and_plot(original_images=test_data, noisy_images=noisy, generated_images=generated, path=fig_file)

        print('>Saved model and figures to', path)
