# from keras import Input, Model
# from keras.layers import Dense, Conv2D, Conv2DTranspose, Activation, LeakyReLU, BatchNormalization, \
#     Flatten, Reshape, Dropout
# from keras.optimizers import Adam
#
#
# def build_adversarial(generator_model, discriminator_model):
#
#     discriminator_model.trainable = False
#
#     # generator.output -> discriminator.input
#     gan_output = discriminator_model(generator_model.output)
#
#     model = Model(generator_model.input, gan_output)
#
#     model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
#                   loss='binary_crossentropy')
#
#     return model
#
#
# def build_generator(latent_size):
#     noise_input = Input(shape=(latent_size,))
#
#     gen = Dense(7 * 7 * 384, input_dim=latent_size, activation='relu')(noise_input)
#     gen = Reshape((7, 7, 384))(gen)
#
#     # # upsample to (7, 7, ...)
#     # gen = Conv2DTranspose(192, 5, strides=1, padding='valid',
#     #                       activation='relu',
#     #                       kernel_initializer='glorot_normal')(gen)
#     # gen = BatchNormalization()(gen)
#
#     # upsample to (14, 14, ...)
#     gen = Conv2DTranspose(192, kernel_size=(5, 5), strides=(2, 2), padding='same',
#                           activation='relu',
#                           kernel_initializer='glorot_normal')(gen)
#     gen = BatchNormalization()(gen)
#
#     # upsample to (28, 28, ...)
#     gen = Conv2DTranspose(1, kernel_size=(5, 5), strides=(2, 2), padding='same',
#                           activation='tanh',
#                           kernel_initializer='glorot_normal')(gen)
#
#     model = Model(noise_input, gen)
#     return model
#
#
# def build_discriminator(input_shape):
#     in_image = Input(shape=input_shape)
#
#     cnn = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same')(in_image)
#     cnn = LeakyReLU(alpha=0.2)(cnn)
#     cnn = Dropout(0.4)(cnn)
#
#     cnn = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(cnn)
#     cnn = LeakyReLU(alpha=0.2)(cnn)
#     cnn = Dropout(0.4)(cnn)
#
#     cnn = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(cnn)
#     cnn = LeakyReLU(alpha=0.2)(cnn)
#     cnn = Dropout(0.4)(cnn)
#
#     cnn = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(cnn)
#     cnn = LeakyReLU(alpha=0.2)(cnn)
#     cnn = Dropout(0.4)(cnn)
#
#     cnn = Flatten()(cnn)
#
#     out = Dense(1, activation='sigmoid')(cnn)
#
#     model = Model(in_image, out)
#     model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
#                   loss='binary_crossentropy')
#
#     return model
