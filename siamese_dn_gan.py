import os
from datetime import datetime

import numpy as np

from keras import Input, Model, Sequential
from keras.layers import Dense, Conv2D,  LeakyReLU, BatchNormalization, \
    Flatten, Dropout, Activation, Lambda
from keras.optimizers import Adam

from keras import backend as K

from noise_maker import NoiseMaker
from utils import measure_and_plot, mean_ssim


def euclidean_distance(vectors):
    """
    Euclidean distance between two vectors
    """
    x, y = vectors
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    """
    Output shape of the Euclidean distance
    """
    shape1, shape2 = shapes
    return (shape1[0], 1)


def build_adversarial(generator_model, discriminator_model):
    """
    Adversarial Model builder
                            Input(real image)
                                            \
        Input(noisy image) -> Generator -> Discriminator(static, untrainable) -> Close/Far
    """
    discriminator_model.trainable = False

    real_input = Input(shape=discriminator_model.input_shape[0][1:])
    fake_input = generator_model.output  # generator.output -> discriminator.input

    gan_output = discriminator_model([real_input, fake_input])

    model = Model([real_input, generator_model.input], gan_output)

    model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
                  # loss='binary_crossentropy')
                  loss='mean_squared_error')
                  # loss=contrastive_loss)

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
                Input(real)  Input(noisy)
                    |              |
        [Convolution2D(32) -> Convolution2D(64)
            -> Convolution2D(128) -> Convolution2D(256)]
                    |              |
            Real_Features   Fake_Features
                    \_____________/
                   Euclidean Distance
                            |
                        Output(1) : Close or Not
    """
    real_input = Input(shape=input_shape)
    fake_input = Input(shape=input_shape)

    cnn = Sequential(layers=[
        Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(alpha=0.2),
        Dropout(0.4),

        Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same'),
        LeakyReLU(alpha=0.2),
        Dropout(0.4),

        Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(alpha=0.2),
        Dropout(0.4),

        Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'),
        LeakyReLU(alpha=0.2),
        Dropout(0.4)
    ])

    real_cnn = cnn(real_input)
    fake_cnn = cnn(fake_input)

    real_vector = Flatten()(real_cnn)
    fake_vector = Flatten()(fake_cnn)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([real_vector, fake_vector])

    out = Dense(1, activation='sigmoid')(distance)

    model = Model([real_input, fake_input], out)

    model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
                  # loss='binary_crossentropy')
                  loss='mean_squared_error')
                  # loss=contrastive_loss)

    return model


#######################################################################################################################


class SiameseDenoiseGAN:

    def __init__(self, data_shape):
        """
        Initialize SiDN-GAN
        """
        self.data_shape = data_shape
        self.discriminator = None
        self.generator = None
        self.adversarial = None

        self.define_gan()
        self.noise_maker = NoiseMaker(shape=self.data_shape, noise_type='s&p')

        self.performance_output_path = 'performance/siamese_dn_gan_' + str(datetime.now().date())

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

            # Train discriminator for real vs fake, far samples
            Y = np.ones(shape=(len(realX),))
            discriminator_loss_rf = self.discriminator.train_on_batch([realX, fakeX], Y)

            # Train discriminator for fake vs noisy, close samples
            Y = np.zeros(shape=(len(realX),))
            discriminator_loss_fn = self.discriminator.train_on_batch([fakeX, noisy], Y)

            # Add noise to images
            noisy = self.noise_maker.add_noise(realX)
            act_real = np.zeros(shape=(len(noisy),))

            # Train adversarial model with denoised images that are labeled like close to real images
            gan_loss = self.adversarial.train_on_batch([realX, noisy], act_real)

            trained_samples = min(trained_samples+batch_size, dataset.sample_number)
            print('     %5d/%d -> Discriminator Loss: [RvsF: %f, FvsN: %f], Gan Loss: %f'
                  % (trained_samples, dataset.sample_number, discriminator_loss_rf, discriminator_loss_fn, gan_loss))

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
