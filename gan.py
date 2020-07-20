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

        self.losses["g"].append(g_loss_avg / batches_count)
        self.losses["d_real"].append(d_loss_real_avg / batches_count)
        self.losses["d_fake"].append(d_loss_fake_avg / batches_count)

    def test_accuracy(self, images, n):
        real_images = images[np.random.randint(0, images.shape[0], n)]
        real_labels = tf.zeros((n, 1))
        fake_images = self.generate_images(n)
        fake_labels = tf.ones((n, 1))

        real_loss, real_accuracy = self.discriminator.evaluate(real_images, real_labels, verbose=0)
        fake_loss, fake_accuracy = self.discriminator.evaluate(fake_images, fake_labels, verbose=0)

        self.accuracies["real"].append(real_accuracy)
        self.accuracies["fake"].append(fake_accuracy)

    def save_plot(self, path, epoch, n):
        if self.test_noise is None:
            self.test_noise = self.generate_latent(n*n)

        images = (self.generator(self.test_noise) + 1) * 0.5
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
        ax.plot(epochs, self.losses["g"], label='g loss')
        ax.plot(epochs, self.losses["d_real"], label='d loss (real)')
        ax.plot(epochs, self.losses["d_fake"], label='d loss (fake)')
        ax.legend()
        plt.savefig(f'{path}/losses.jpg')
        plt.close()

    def save_accuracies(self, path, epoch):
        fig, ax = plt.subplots()

        epochs = [i for i in range(epoch + 1)]
        ax.plot(epochs, self.accuracies["real"], label='real accuracy')
        ax.plot(epochs, self.accuracies["fake"], label='fake accuracy')
        ax.legend()
        plt.savefig(f'{path}/accuracies.jpg')
        plt.close()

    def print_metrics(self, epoch):
        print(f'epoch {epoch}', end=' ')
        print(f'g_loss: {self.losses["g"][-1]},', end=' ')
        print(f'd_loss_real: {self.losses["d_real"][-1]},', end=' ')
        print(f'd_loss_fake: {self.losses["d_fake"][-1]},', end=' ')
        print(f'real accuracy: {self.accuracies["real"][-1]},', end=' ')
        print(f'fake accuracy: {self.accuracies["fake"][-1]}')

    def train(self, images, epochs, batch_size, models_path, images_path, num_img=8, test_acc_num=128, save_period=5, init_epoch=0):
        self.losses = {"g": [], "d_real": [], "d_fake": []}
        self.accuracies = {"real": [], "fake": []}

        for epoch in range(init_epoch, epochs):
            self.train_step(images, batch_size, epoch)
            self.test_accuracy(images, test_acc_num)

            self.save_losses(images_path, epoch)
            self.save_accuracies(images_path, epoch)
            self.print_metrics(epoch)

            if epoch < 10 or epoch % save_period == 0:
                self.save_plot(images_path, epoch, num_img)
                self.generator.save(f'{models_path}/generator_epoch{epoch}.h5')
                self.discriminator.save(f'{models_path}/discriminator_epoch{epoch}.h5')
