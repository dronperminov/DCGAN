import tensorflow as tf
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt


class GAN:
    def __init__(self, latent_dim, discriminator, generator, d_optimizer, g_optimizer, loss_fn='binary_crossentropy'):
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.test_noise = None

        self.discriminator.compile(optimizer=d_optimizer, loss=loss_fn, metrics=['accuracy'])
        self.discriminator.trainable = False

        self.gan = keras.models.Sequential()
        self.gan.add(generator)
        self.gan.add(discriminator)
        self.gan.compile(optimizer=g_optimizer, loss=loss_fn)

    def summary(self):
        print("Generator:")
        self.generator.summary()

        print("Discriminator:")
        self.discriminator.summary()

        print("Gan:")
        self.gan.summary()

    def generate_latent(self, size):
        return tf.random.normal(shape=(size, self.latent_dim))

    def generate_images(self, size):
        random_latent_vectors = self.generate_latent(size)
        return self.generator(random_latent_vectors)

    def train_discriminator(self, train_images, batch_size):
        real_images = train_images[np.random.randint(0, train_images.shape[0], batch_size // 2)]
        real_labels = tf.random.uniform((batch_size // 2, 1), 0.0, 0.1)

        fake_images = self.generate_images(batch_size // 2)
        fake_labels = tf.random.uniform((batch_size // 2, 1), 0.9, 1.0)

        d_loss_real, _ = self.discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake, _ = self.discriminator.train_on_batch(fake_images, fake_labels)

        return d_loss_real, d_loss_fake

    def train_generator(self, batch_size):
        random_latent_vectors = self.generate_latent(batch_size)
        misleading_labels = tf.zeros((batch_size, 1))

        return self.gan.train_on_batch(random_latent_vectors, misleading_labels)

    def train_step(self, images, batch_size, epoch):
        g_loss_avg = 0
        d_loss_real_avg = 0
        d_loss_fake_avg = 0
        batches_count = images.shape[0] // batch_size

        for i in range(batches_count):
            d_loss_real, d_loss_fake = self.train_discriminator(images, batch_size)
            g_loss = self.train_generator(batch_size)

            g_loss_avg += g_loss
            d_loss_real_avg += d_loss_real
            d_loss_fake_avg += d_loss_fake

            print(f'epoch {epoch} batch {i}, g_loss: {g_loss}, d_loss_real: {d_loss_real}, d_loss_fake: {d_loss_fake}', end='\r')

        return g_loss_avg / batches_count, d_loss_real_avg / batches_count, d_loss_fake_avg / batches_count

    def save_plot(self, path, epoch, n):
        if self.test_noise is None:
            self.test_noise = self.generate_latent(n*n)

        images = (self.generator(self.test_noise) + 1) * 0.5
        fig, ax = plt.subplots(n, n, figsize=(20, 20))

        for i, image in enumerate(images):
            ax[i // n, i % n].axis("off")
            ax[i // n, i % n].imshow(image, aspect="auto")

        plt.subplots_adjust(wspace=.05, hspace=.05)
        plt.savefig(path + '/epoch%d.png' % epoch, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def save_losses(self, path, epoch, g_losses, d_losses_real, d_losses_fake):
        fig, ax = plt.subplots()

        epochs = [i for i in range(epoch + 1)]
        ax.plot(epochs, g_losses, label='g loss')
        ax.plot(epochs, d_losses_real, label='d loss (real)')
        ax.plot(epochs, d_losses_fake, label='d loss (fake)')
        ax.legend()
        plt.savefig(path + '/losses.jpg')
        plt.close()

    def train(self, images, epochs, batch_size, models_path, images_path, num_img=8, save_period=5, init_epoch=0):
        g_losses = []
        d_losses_real = []
        d_losses_fake = []

        for epoch in range(init_epoch, epochs):
            g_loss, d_loss_real, d_loss_fake = self.train_step(images, batch_size, epoch)
            g_losses.append(g_loss)
            d_losses_real.append(d_loss_real)
            d_losses_fake.append(d_loss_fake)

            self.save_losses(images_path, epoch, g_losses, d_losses_real, d_losses_fake)

            if epoch < 10 or epoch % save_period == 0:
                self.save_plot(images_path, epoch, num_img)
                self.generator.save(models_path + f'/generator_epoch{epoch}.h5')
                self.discriminator.save(models_path + f'/discriminator_epoch{epoch}.h5')

            print(f'epoch {epoch}, g_loss: {g_losses[-1]}, d_loss_real: {d_losses_real[-1]}, d_loss_fake: {d_losses_fake[-1]}')

