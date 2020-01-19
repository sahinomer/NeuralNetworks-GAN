
from matplotlib import pyplot
from skimage.measure import compare_ssim


def measure_and_plot(original_images, noisy_images, generated_images, path):
    """
    Measure SSIM between real and denoised images and plot results
    :param original_images: Real images
    :param noisy_images: Noisy images
    :param generated_images: Denoised images
    :param path: Path to figures that saved
    """

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
    """
    Calculate average SSIM on the test set
    :param epoch: Current epoch for indexing
    :param original_images: Real images
    :param noise_maker: Noise maker object
    :param generator: Generative model for denoising
    :param path: Path to SSIM index that saved
    """
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
