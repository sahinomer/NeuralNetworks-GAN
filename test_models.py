import os

from keras.engine.saving import load_model

from dataset import Dataset
from noise_maker import NoiseMaker
from utils import mean_ssim, measure_and_plot


def performance(generator, noise_maker, epoch, test_data, model_folder):

    path = model_folder + '_reproduced/epoch-%04d' % (epoch + 1)
    if not os.path.exists(path):
        os.makedirs(path)

    mean_ssim(epoch, test_data, noise_maker, generator, model_folder + '_reproduced/result.txt')

    test_data = test_data[epoch * 100:(epoch + 1) * 100]

    # generate fake examples
    noisy = noise_maker.add_noise(real_samples=test_data)
    generated = generator.predict(noisy)

    # save the generator model
    model_file = path + '/model_%04d.h5' % (epoch + 1)
    generator.save(model_file)

    fig_file = path + '/plot_%04d' % ((epoch + 1))
    measure_and_plot(original_images=test_data, noisy_images=noisy, generated_images=generated, path=fig_file)

    print('>Saved model and figures to', path)


if __name__ == '__main__':

    dataset = Dataset(dataset='caltech256')
    dataset.split_test_data(test_sample=2000)
    noise_maker = NoiseMaker(shape=dataset.data_shape, noise_type='s&p')

    model_folder = 'C:/PycharmProjects/NeuralNetworks-GAN/performance/caltech256-64x64-siamese_dn_gan_2019-12-17'

    for epoch in range(20):
        generator_path = model_folder + '/epoch-%04d' % (epoch + 1) + '/model_%04d.h5' % (epoch + 1)
        generator = load_model(generator_path)

        performance(generator, noise_maker, epoch, dataset.test_data, model_folder)
