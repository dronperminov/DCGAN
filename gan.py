import tensorflow as tf
import numpy as np
from tensorflow import keras
from matplotlib import pyplot


class GAN:
    def __init__(self, latent_dim, discriminator, generator, d_optimizer, g_optimizer, loss_fn='binary_crossentropy'):
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

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

        labels = tf.concat([tf.ones((batch_size, 1)) * 0.9, tf.zeros((batch_size, 1))], axis=0)
        labels += 0.1 * tf.random.uniform(tf.shape(labels))

        self.discriminator.trainable = True
        d_loss, _ = self.discriminator.train_on_batch(combined_images, labels)

        return d_loss

    def train_generator(self, batch_size):
        random_latent_vectors = self.generate_latent(batch_size)
        misleading_labels = tf.zeros((batch_size, 1))

        self.discriminator.trainable = False
        g_loss = self.gan.train_on_batch(random_latent_vectors, misleading_labels)

        return g_loss

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        d_loss = self.train_discriminator(real_images)
        g_loss = self.train_generator(tf.shape(real_images)[0])

        return {"d_loss": d_loss, "g_loss": g_loss}

    def save_plot(self, path, epoch, n=8):
        examples = self.generate_images(n*n)
        examples = (examples + 1) / 2

        loss, accuracy = self.discriminator.evaluate(examples, tf.zeros((n*n, 1)), verbose=0)
        print("accuracy:", accuracy)

        for i in range(n * n):
            pyplot.subplot(n, n, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(examples[i])

        filename = path + '/epoch%03d.png' % epoch
        pyplot.savefig(filename, dpi=400)
        pyplot.close()

    def train(self, dataset, epochs, models_path, images_path, num_img=8, save_period=5, init_epoch=0):
        for epoch in range(init_epoch, epochs):
            for i, batch in enumerate(dataset):
                losses = self.train_step(batch)
                print('epoch {epoch} batch {i}, d_loss: {d_loss}, g_loss: {g_loss}'.format(epoch=epoch, i=i, d_loss=losses["d_loss"], g_loss=losses["g_loss"]), end='\r')

            if epoch < 10 or epoch % save_period == 0:
                self.save_plot(images_path, epoch, num_img)
                self.generator.save(models_path + '/generator_epoch{epoch}.h5'.format(epoch=epoch))
                self.discriminator.save(models_path + '/discriminator_epoch{epoch}.h5'.format(epoch=epoch))
            else:
                print('')
