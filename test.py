import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from matplotlib import pyplot as plt

from fid import FrechetInceptionDistance


def make_interpolation(generator, latent_dim, count, filename='interpolation.png'):
    fig, ax = plt.subplots(count, count, figsize=(20, 20))

    for i in range(count):
        x1 = np.random.randn(latent_dim)
        x2 = np.random.randn(latent_dim)

        ti = [i / (count - 1) for i in range(count)]
        x = np.array([x1 * t + x2 * (1 - t) for t in ti])
        images = (generator(x) + 1) * 0.5

        for j, img in enumerate(images):
            ax[i, j].axis("off")
            ax[i, j].imshow(img, aspect="auto")

    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def make_examples(generator, latent_dim, count, filename='examples.png'):
    images = (generator(tf.random.normal(shape=(count * count, latent_dim))) + 1) * 0.5
    fig, ax = plt.subplots(count, count, figsize=(50, 50))

    for i, image in enumerate(images):
        ax[i // count, i % count].axis("off")
        ax[i // count, i % count].imshow(image, aspect="auto")

    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def evaluate_fid(generator, latent_dim, count, dataset_path):
    files = [f for f in os.listdir(dataset_path)][:count]
    images1 = [np.array(image.load_img(dataset_path + f)) for f in files]
    images1 = np.array(images1).astype('float32')

    noise = tf.random.normal(shape=(count, latent_dim))
    images2 = (generator(noise).numpy() + 1) * 127.5

    fid = FrechetInceptionDistance()
    print('fid:', fid.evaluate(images1, images2))


def main():
    latent_dim = 128
    interpolation_count = 10
    fid_count = 1024
    examples_count = 16

    generator_path = "models/generator_epoch205.h5"
    images_path = "C:/Users/dronp/Desktop/ImageRecognition/cats/"

    generator = tf.keras.models.load_model(generator_path)

    make_interpolation(generator, latent_dim, interpolation_count)
    make_examples(generator, latent_dim, examples_count)
    evaluate_fid(generator, latent_dim, fid_count, images_path)


if __name__ == '__main__':
    main()