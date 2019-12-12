import numpy as np


class FakeSamples:

    def __init__(self, generator, latent_size, class_number=0):
        self.generator = generator
        self.latent_size = latent_size
        self.class_number = class_number

    def generate_latent_points(self, sample_number):

        noise = np.random.randn(sample_number, self.latent_size)

        if self.class_number > 0:
            labels = np.random.randint(0, self.class_number, sample_number)
            return noise, labels

        return noise

    def generate_fake_samples(self, sample_number, set_labels=None):
        fake_label = np.zeros((sample_number,))
        if self.class_number > 0:
            noise, labels = self.generate_latent_points(sample_number)

            if set_labels is not None:
                labels = set_labels

            fake_data = self.generator.predict([noise, labels])
            return fake_data, labels, fake_label

        else:
            noise = self.generate_latent_points(sample_number)
            fake_data = self.generator.predict(noise)
            return fake_data, fake_label
