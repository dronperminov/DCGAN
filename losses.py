import tensorflow as tf


class GANLoss:
    def __init__(self, from_logits=True, smoothing=0.4):
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=smoothing)

    def discriminator_loss(self, real_output, fake_output):
        real_part = self.bce(tf.ones_like(real_output), real_output)
        fake_part = self.bce(tf.zeros_like(fake_output), fake_output)

        return real_part + fake_part

    def generator_loss(self, real_output, fake_output):
        return self.bce(tf.ones_like(fake_output), fake_output)


class LSGANLoss:
    def discriminator_loss(self, real_output, fake_output):
        real_part = tf.reduce_mean(tf.square(real_output), 0)
        fake_part = tf.reduce_mean(tf.square(fake_output - tf.ones_like(fake_output)), 0)

        return (real_part + fake_part) / 2

    def generator_loss(self, real_output, fake_output):
        return tf.reduce_mean(tf.square(fake_output), 0)


class RaLSGANLoss:
    def discriminator_loss(self, real_output, fake_output):
        real_part = tf.reduce_mean(tf.square(real_output - tf.reduce_mean(fake_output, 0) - tf.ones_like(real_output)), 0)
        fake_part = tf.reduce_mean(tf.square(fake_output - tf.reduce_mean(real_output, 0) + tf.ones_like(real_output)), 0)

        return (real_part + fake_part) / 2

    def generator_loss(self, real_output, fake_output):
        real_part = tf.reduce_mean(tf.square(real_output - tf.reduce_mean(fake_output, 0) + tf.ones_like(real_output)), 0)
        fake_part = tf.reduce_mean(tf.square(fake_output - tf.reduce_mean(real_output, 0) - tf.ones_like(real_output)), 0)

        return (real_part + fake_part) / 2
