import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


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


def main():
    latent_dim = 128
    count = 10
    filename = "models/generator_epoch205.h5"

    generator = tf.keras.models.load_model(filename)

    make_interpolation(generator, latent_dim, count)
    make_examples(generator, latent_dim, 16)


if __name__ == '__main__':
    main()