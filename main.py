import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
import numpy as np
import os

import losses
from gan import GAN


def load_data(path: str, n: int = -1):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))][:n]
    imgs = [np.array(image.load_img(path + file)) for file in files]

    return (np.array(imgs).astype('float32') - 127.5) / 127.5


def main():
    dataset_path = "C:/Users/dronp/Desktop/ImageRecognition/cats/"
    images_path = "generated/"
    models_path = "models/"

    image_shape = (64, 64, 3)

    latent_dim = 256
    batch_size = 16
    iterations = 300000

    save_period = 1500
    save_loss_period = 500
    examples_count = 8

    init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    generator = keras.Sequential([
        layers.Dense(4 * 4 * 1024, kernel_initializer=init, input_shape=(latent_dim,)),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((4, 4, 1024)),

        layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='tanh'),
    ], name="generator")

    discriminator = keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, input_shape=image_shape),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),

        layers.Flatten(),
        layers.Dense(1, kernel_initializer=init),
    ], name="discriminator")

    g_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    d_optimizer = keras.optimizers.Adam(learning_rate=0.00015, beta_1=0.5)

    images = load_data(dataset_path)

    if models_path and not os.path.exists(models_path):
        os.mkdir(models_path)

    if images_path and not os.path.exists(images_path):
        os.mkdir(images_path)

    loss_fn = losses.RaLSGANLoss()
    gan = GAN(latent_dim, discriminator, generator, d_optimizer, g_optimizer, loss_fn)
    gan.summary()
    gan.train(images, iterations, batch_size, models_path, images_path, examples_count, save_period, save_loss_period)


if __name__ == '__main__':
    main()