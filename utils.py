import os

import numpy as np
from matplotlib import pyplot

from skimage.measure import compare_ssim


def performance(model, epoch, test_data):

    test_data = test_data[epoch * 100:(epoch + 1) * 100]

    # generate fake examples
    noisy = model.noisy_samples.add_noise(real_samples=test_data)
    generated = model.generator.predict(noisy)

    path = model.performance_output_path + '/epoch-%04d' % (epoch + 1)
    if not os.path.exists(path):
        os.makedirs(path)

    # save the generator model
    model_file = path + '/model_%04d.h5' % (epoch + 1)
    model.generator.save(model_file)

    fig_file = path + '/plot_%04d' % ((epoch + 1))
    measure_and_plot(original_images=test_data, noisy_images=noisy, generated_images=generated, path=fig_file)

    print('>Saved model and figures to', path)


def measure_and_plot(original_images, noisy_images, generated_images, path=None):

    for i, (original, noisy, generated) in enumerate(zip(original_images, noisy_images, generated_images)):

        ssim = compare_ssim(original, generated, multichannel=True)

        # scale from [-1,1] to [0,1]
        original = (original + 1) / 2.0
        noisy = (noisy + 1) / 2.0
        generated = (generated + 1) / 2.0

        fig = pyplot.figure()
        fig.suptitle('SSIM:' + str(ssim), fontsize=12, fontweight='bold')

        pyplot.subplot(1, 3, 1)
        pyplot.axis('off')
        pyplot.imshow(original)

        pyplot.subplot(1, 3, 2)
        pyplot.axis('off')
        pyplot.imshow(noisy)

        pyplot.subplot(1, 3, 3)
        pyplot.axis('off')
        pyplot.imshow(generated)

        img_path = path + '-%04d.png' % (i+1)
        pyplot.savefig(img_path)
        pyplot.close()
