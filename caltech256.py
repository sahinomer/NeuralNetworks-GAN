import os

import numpy as np
from sklearn.utils import shuffle

from keras.utils import get_file
from keras_preprocessing.image import load_img, img_to_array
from skimage.transform import resize


def load_data(width=128, height=128, verbose=False):

    img_res = str(width) + 'x' + str(height)
    try:
        image_data = np.load(file='caltech256_' + img_res + '.images.npy')
        image_label = np.load(file='caltech256_' + img_res + '.labels.npy')
        image_data, image_label = shuffle(image_data, image_label, random_state=0)
        return image_data, image_label
    except IOError:
        print('Cached images not found! Loading images...')

    dirname = '256_ObjectCategories'
    origin = 'http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar'
    path = get_file(dirname, origin=origin, untar=True)

    num_samples = 30607
    image_data = np.empty((num_samples, width, height, 3), dtype='uint8')
    image_label = np.empty((num_samples,), dtype='uint8')

    category_list = os.listdir(path)
    sample = 0
    for category in category_list:

        if verbose:
            print(category + '...')

        category_path = path + '\\' + category + '\\'
        image_list = os.listdir(category_path)

        for image in image_list:
            if image.lower().endswith('.jpg'):
                img = load_img(category_path + image)
                img = img_to_array(img)
                img = resize(img, (width, height))

                image_data[sample, :, :, :] = img
                image_label[sample] = int(category.split('.')[0])
                sample += 1

    image_data = image_data[:sample, :, :, :]
    image_label = image_label[:sample]

    # Save images and labels to cache file
    np.save(file='caltech256_' + img_res + '.images', arr=image_data)
    np.save(file='caltech256_' + img_res + '.labels', arr=image_label)

    image_data, image_label = shuffle(image_data, image_label, random_state=0)
    return image_data, image_label


if __name__ == '__main__':
    caltech256images = load_data(verbose=True)
    print(caltech256images[0].shape)
    print(caltech256images[1].shape)

