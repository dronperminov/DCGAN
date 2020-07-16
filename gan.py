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

        gan_input = keras.Input(shape=(latent_dim,))
        gan_output = discriminator(generator(gan_input))
        self.gan = keras.models.Model(gan_input, gan_output)
        self.gan.compile(optimizer=g_optimizer, loss=loss_fn)

    def summary(self):
        print("Generator:")
        self.generator.summary()

        print("Discriminator:")
        self.discriminator.summary()

    def generate_latent(self, size):
        return tf.random.normal(shape=(size, self.latent_dim))

    def generate_images(self, size):
        random_latent_vectors = self.generate_latent(size)
        return self.generator(random_latent_vectors)

    def train_discriminator(self, real_images):
        batch_size = tf.shape(real_images)[0]

        generated_images = self.generate_images(batch_size)
        combined_images = tf.concat([generated_images, real_images], axis=0)

        labels = tf.concat([tf.zeros((batch_size, 1)), tf.ones((batch_size, 1)) * 0.7], axis=0)
        labels += 0.3 * tf.random.uniform(tf.shape(labels))

        self.discriminator.trainable = True
        d_loss, _ = self.discriminator.train_on_batch(combined_images, labels)

        return d_loss

    def train_generator(self, batch_size):
        random_latent_vectors = self.generate_latent(batch_size)
        misleading_labels = tf.ones((batch_size, 1))

        self.discriminator.trainable = False
        g_loss = self.gan.train_on_batch(random_latent_vectors, misleading_labels)

        return g_loss

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        d_loss = self.train_discriminator(real_images)
        g_loss = self.train_generator(tf.shape(real_images)[0])

        return {"d_loss": d_loss, "g_loss": g_loss}

    def save_plot(self, path, epoch, n):
        if self.test_noise is None:
            self.test_noise = self.generate_latent(n*n)

        images = self.generator(self.test_noise)
        loss, accuracy = self.discriminator.evaluate(images, tf.ones((n*n, 1)), verbose=0)
        print("accuracy:", accuracy)
        images = (images + 1) / 2

        fig, axes = plt.subplots(n, n, figsize=(20, 20))

        for i, image in enumerate(images):
            axes[i // n, i % n].axis("off")
            axes[i // n, i % n].imshow(image, aspect="auto")

        plt.subplots_adjust(wspace=.05, hspace=.05)

        filename = path + '/epoch%03d.png' % epoch
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def train(self, images, epochs, batch_size, models_path, images_path, num_img=10, save_period=5, init_epoch=0):
        dataset = tf.data.Dataset.from_tensor_slices(images)
        dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(32)

        for epoch in range(init_epoch, epochs):
            d_loss = 0
            g_loss = 0
            losses_count = 0

            for i, batch in enumerate(dataset):
                losses = self.train_step(batch)
                d_loss += losses["d_loss"]
                g_loss += losses["g_loss"]
                losses_count += 1

                print(f'epoch {epoch} batch {i}, d_loss: {d_loss / losses_count}, g_loss: {g_loss / losses_count}', end='\r')

            d_loss /= losses_count
            g_loss /= losses_count

            if epoch < 10 or epoch % save_period == 0:
                self.save_plot(images_path, epoch, num_img)
                self.generator.save(models_path + '/generator_epoch{epoch}.h5'.format(epoch=epoch))
                self.discriminator.save(models_path + '/discriminator_epoch{epoch}.h5'.format(epoch=epoch))
            else:
                print(f'epoch {epoch}, d_loss: {d_loss}, g_loss: {g_loss}')

