import os

import numpy as np
from matplotlib import pyplot

from skimage.measure import compare_ssim


# def performance(model, epoch, test_data):
#
#     test_data = test_data[epoch * 100:(epoch + 1) * 100]
#
#     # generate fake examples
#     noisy = model.noisy_samples.add_noise(real_samples=test_data)
#     generated = model.generator.predict(noisy)
#
#     path = model.performance_output_path + '/epoch-%04d' % (epoch + 1)
#     if not os.path.exists(path):
#         os.makedirs(path)
#
#     # save the generator model
#     model_file = path + '/model_%04d.h5' % (epoch + 1)
#     model.generator.save(model_file)
#
#     fig_file = path + '/plot_%04d' % ((epoch + 1))
#     measure_and_plot(original_images=test_data, noisy_images=noisy, generated_images=generated, path=fig_file)
#
#     print('>Saved model and figures to', path)


def measure_and_plot(original_images, noisy_images, generated_images, path):

    # scale from [-1,1] to [0,1]
    original_images = (original_images + 1) / 2.0
    noisy_images = (noisy_images + 1) / 2.0
    generated_images = (generated_images + 1) / 2.0

    for i, (original, noisy, generated) in enumerate(zip(original_images, noisy_images, generated_images)):

        ssim = compare_ssim(original, generated, multichannel=True)

        fig = pyplot.figure(figsize=(16, 8))
        fig.suptitle('SSIM:' + str(ssim), fontsize=12, fontweight='bold')

        pyplot.subplot(1, 3, 1)
        pyplot.axis('off')
        pyplot.imshow(original, interpolation='none')

        pyplot.subplot(1, 3, 2)
        pyplot.axis('off')
        pyplot.imshow(noisy, interpolation='none')

        pyplot.subplot(1, 3, 3)
        pyplot.axis('off')
        pyplot.imshow(generated, interpolation='none')

        img_path = path + '-%04d.png' % (i+1)
        pyplot.savefig(img_path)
        pyplot.close()


def mean_ssim(epoch, original_images, noise_maker, generator, path):
    noisy = noise_maker.add_noise(real_samples=original_images)
    generated_images = generator.predict(noisy)

    # scale from [-1,1] to [0,1]
    original_images = (original_images + 1) / 2.0
    generated_images = (generated_images + 1) / 2.0

    ssim_total = 0
    for original, generated in zip(original_images, generated_images):
        ssim_total += compare_ssim(original, generated, multichannel=True)

    avg_ssim = ssim_total / len(original_images)

    with open(path, mode='a+', encoding='utf8') as result:
        print('%d\t%f' % (epoch+1, avg_ssim), file=result)
