import numpy as np


def generate_noise(noise_type, data_shape, samples):
    """
    Generate and add noise to images
    :param noise_type: Noise type
    :param data_shape: Image shape
    :param samples: Real image samples
    :return: Noisy images
    """
    sample_number = len(samples)
    # Gaussian Noise
    if noise_type == 'gauss':
        data_shape = (sample_number, data_shape[0], data_shape[1], data_shape[2])
        mean = 0
        var = 0.04
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, data_shape)
        noisy = samples + gauss
        noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min())
        noisy = (noisy - 0.5) / 0.5
        return noisy

    # Salt&Pepper Noise
    elif noise_type == "s&p":
        size = data_shape[0] * data_shape[1]
        s_vs_p = 0.5
        amount = 0.16
        out = np.copy(samples)
        # Generate Salt '1' noise
        num_salt = int(np.ceil(amount * size * s_vs_p))
        x = np.random.randint(data_shape[0], size=num_salt)
        y = np.random.randint(data_shape[1], size=num_salt)
        out[:, x, y, :] = 1
        # Generate Pepper '-1' noise
        num_pepper = int(np.ceil(amount * size * (1. - s_vs_p)))
        x = np.random.randint(data_shape[0], size=num_pepper)
        y = np.random.randint(data_shape[1], size=num_pepper)
        out[:, x, y, :] = -1
        return out

    # Poisson Noise
    elif noise_type == "poisson":
        vals = len(np.unique(samples))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(samples * vals) / float(vals)
        return noisy

    # Speckle Noise
    elif noise_type == "speckle":
        gauss = np.random.randn(sample_number, data_shape[0], data_shape[1], data_shape[2]) / 2
        noisy = samples + gauss
        noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min())
        noisy = (noisy - 0.5) / 0.5
        return noisy


class NoiseMaker:

    def __init__(self, shape, noise_type):
        """
        Initialize noise maker class
        :param shape: Image shape
        :param noise_type: Noise type
        """
        self.shape = shape
        self.noise_type = noise_type

    def add_noise(self, real_samples):
        """
        Add noise to images
        :param real_samples: Real images without noise
        :return: Noisy images
        """
        noisy = generate_noise(noise_type=self.noise_type, data_shape=self.shape, samples=real_samples)
        return noisy
