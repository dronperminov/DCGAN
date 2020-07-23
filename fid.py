import numpy as np
from scipy.linalg import sqrtm
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
import gc


class FrechetInceptionDistance:
    def __init__(self):
        self.shape = (299, 299, 3)
        self.model = InceptionV3(include_top=False, pooling='avg', input_shape=self.shape)

    def scale_images(self, images):
        return np.asarray([resize(image, self.shape, 0) for image in images])

    def calculate_fid(self, mu1, sigma1, mu2, sigma2):
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = sqrtm(sigma1.dot(sigma2))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        return ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    def evaluate(self, images1, images2):
        images1 = preprocess_input(self.scale_images(images1))
        images2 = preprocess_input(self.scale_images(images2))

        act1 = self.model.predict(images1)
        act2 = self.model.predict(images2)

        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

        return self.calculate_fid(mu1, sigma1, mu2, sigma2)

    def evaluate_models(self, real_images, generator_paths, latent_dim):
        images = preprocess_input(self.scale_images(real_images))
        act1 = self.model.predict(images)
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)

        noise = tf.random.normal(shape=(real_images.shape[0], latent_dim))

        for path in generator_paths:
            generator = tf.keras.models.load_model(path)

            images2 = (generator(noise).numpy() + 1) * 127.5
            images2 = preprocess_input(self.scale_images(images2))
            act2 = self.model.predict(images2)
            mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

            print(path + ":", self.calculate_fid(mu1, sigma1, mu2, sigma2))
            tf.keras.backend.clear_session()
            gc.collect()
