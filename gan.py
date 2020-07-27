import tensorflow as tf
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt


class GAN(tf.keras.models.Model):
    def __init__(self, latent_dim, discriminator, generator, d_optimizer, g_optimizer):
        super(tf.keras.models.Model, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.test_noise = None

        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.4)

        self.gan = keras.models.Sequential()
        self.gan.add(generator)
        self.gan.add(discriminator)

    def summary(self):
        print("Generator:")
        self.generator.summary()

        print("Discriminator:")
        self.discriminator.summary()

        print("Gan:")
        self.gan.summary()

    @tf.function
    def generator_loss(self, real_output, fake_output):
        real_part = tf.reduce_mean((real_output - tf.reduce_mean(fake_output, 0) + tf.ones_like(real_output)) ** 2, 0)
        fake_part = tf.reduce_mean((fake_output - tf.reduce_mean(real_output, 0) - tf.ones_like(real_output)) ** 2, 0)

        return (real_part + fake_part) / 2.

    @tf.function
    def discriminator_loss(self, real_output, fake_output):
        real_part = self.bce(tf.ones_like(real_output), real_output)
        fake_part = self.bce(tf.zeros_like(fake_output), fake_output)

        return real_part + fake_part

    @tf.function
    def train_step(self, real_images, batch_size):
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_images = self.generator(noise, training=True)
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)

            g_loss = self.generator_loss(real_output, fake_output)
            d_loss = self.discriminator_loss(real_output, fake_output)

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        return g_loss[0], d_loss

    def save_plot(self, path, epoch, n):
        if self.test_noise is None:
            self.test_noise = tf.random.normal(shape=(n*n, self.latent_dim))

        images = (self.generator(self.test_noise, training=False) + 1) * 0.5
        fig, ax = plt.subplots(n, n, figsize=(20, 20))

        for i, image in enumerate(images):
            ax[i // n, i % n].axis("off")
            ax[i // n, i % n].imshow(image, aspect="auto")

        plt.subplots_adjust(wspace=.05, hspace=.05)
        plt.savefig(f'{path}/{epoch}.png', bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def save_losses(self, path, epoch):
        fig, ax = plt.subplots()

        epochs = [i for i in range(epoch + 1)]
        ax.plot(epochs, self.losses_info["g"], label='g loss')
        ax.plot(epochs, self.losses_info["d"], label='d loss')
        ax.legend()
        plt.savefig(f'{path}/losses.jpg')
        plt.close()

    def print_metrics(self, epoch):
        print(f'epoch {epoch}', end=' ')
        print(f'g_loss: {self.losses_info["g"][-1]},', end=' ')
        print(f'd_loss: {self.losses_info["d"][-1]},')

    def train(self, images, epochs, batch_size, models_path, images_path, num_img=8, save_period=5):
        self.losses_info = {"g": [], "d": []}
        n_batches = images.shape[0] // batch_size

        for epoch in range(epochs):
            d_loss_avg = 0
            g_loss_avg = 0

            for batch in range(n_batches):
                train_images = images[np.random.randint(0, images.shape[0], batch_size)]
                g_loss, d_loss = self.train_step(train_images, batch_size)
                g_loss_avg += g_loss
                d_loss_avg += d_loss
                print(f'epoch {epoch}, batch {batch}, g_loss: {g_loss}, d_loss: {d_loss}', end='\r')

            self.losses_info["g"].append(g_loss_avg / n_batches)
            self.losses_info["d"].append(d_loss_avg / n_batches)

            self.print_metrics(epoch)
            self.save_losses(images_path, epoch)

            if epoch < 10 or epoch % save_period == 0:
                self.save_plot(images_path, epoch, num_img)
                self.generator.save(f'{models_path}/generator_epoch{epoch}.h5')
                self.discriminator.save(f'{models_path}/discriminator_epoch{epoch}.h5')
