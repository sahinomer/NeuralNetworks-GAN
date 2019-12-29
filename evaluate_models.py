import os

import tensorflow as tf
from keras import backend as K


from keras.engine.saving import load_model
from matplotlib import pyplot

from dataset import Dataset
from noise_maker import NoiseMaker


def clear_session():
  K.clear_session()
  tf.keras.backend.clear_session()


def plot_model_result(path, original_images, noisy_images, generated_dn_images, generated_sdn_images):

    for i, (original, noisy, generated_dn, generated_sdn) in \
            enumerate(zip(original_images, noisy_images, generated_dn_images, generated_sdn_images)):

        x = 64
        y = 144

        fig = pyplot.figure(figsize=(16, 8))

        pyplot.subplot(1, 4, 1)
        pyplot.axis('off')
        pyplot.imshow(noisy, interpolation='none')
        pyplot.text(x, y, '(a) Input Image', size=12, ha='center')

        pyplot.subplot(1, 4, 2)
        pyplot.axis('off')
        pyplot.imshow(generated_dn, interpolation='none')
        pyplot.text(x, y, '(b) DN-GAN', size=12, ha='center')

        pyplot.subplot(1, 4, 3)
        pyplot.axis('off')
        pyplot.imshow(generated_sdn, interpolation='none')
        pyplot.text(x, y, '(c) SiDN-GAN', size=12, ha='center')

        pyplot.subplot(1, 4, 4)
        pyplot.axis('off')
        pyplot.imshow(original, interpolation='none')
        pyplot.text(x, y, '(d) Ground Truth', size=12, ha='center')

        img_path = path + '/image-%04d.png' % (i+1)
        pyplot.savefig(img_path)
        pyplot.close()


if __name__ == '__main__':

    dataset = Dataset(dataset='caltech256')
    dataset.split_test_data(test_sample=2000)
    noise_maker = NoiseMaker(shape=dataset.data_shape, noise_type='s&p')

    # dataset_name = 'cifar10-32x32'
    # dataset_name = 'caltech256-64x64'
    dataset_name = 'caltech256-128x128'

    model_folder = 'C:/PycharmProjects/NeuralNetworks-GAN/performance/'

    path = model_folder + dataset_name + '_evaluate'
    if not os.path.exists(path):
        os.makedirs(path)

    best_dn_gan_path = model_folder + dataset_name + '-dn_gan_2019-12-24' + '/epoch-0018/model_0018.h5'
    best_sdn_gan_path = model_folder + dataset_name + '-siamese_dn_gan_2019-12-21' + '/epoch-0019/model_0019.h5'

    # generate fake examples
    noisy = noise_maker.add_noise(real_samples=dataset.test_data)

    clear_session()
    best_dn_generator = load_model(best_dn_gan_path)
    generated_dn = best_dn_generator.predict(noisy)

    clear_session()
    best_sdn_generator = load_model(best_sdn_gan_path)
    generated_sdn = best_sdn_generator.predict(noisy)

    plot_model_result(path=path, original_images=dataset.test_data, noisy_images=noisy,
                      generated_dn_images=generated_dn, generated_sdn_images=generated_sdn)
