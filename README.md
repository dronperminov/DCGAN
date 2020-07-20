# DCGAN
Implementation of DCGAN for generation 64x64 RGB images with Tensorflow and Keras

## Examples

### Anime-faces <a href='https://www.kaggle.com/soumikrakshit/anime-faces'>dataset</a>
<table>
  <tr>
    <td width='33%'><img src='https://github.com/dronperminov/DCGAN/blob/master/examples/anime_faces_examples.png' /></td>
    <td width='33%'><img src='https://github.com/dronperminov/DCGAN/blob/master/examples/anime_faces_interpolation.png' /></td>
    <td width='33%'><img src='https://github.com/dronperminov/DCGAN/blob/master/examples/anime_faces_losses.jpg' /></td>
  </tr>
</table>

### Cats-faces <a href='https://www.kaggle.com/spandan2/cats-faces-64x64-for-generative-models'>dataset</a>
<table>
  <tr>
    <td width='33%'><img src='https://github.com/dronperminov/DCGAN/blob/master/examples/cats_examples.png' /></td>
    <td width='33%'><img src='https://github.com/dronperminov/DCGAN/blob/master/examples/cats_interpolation.png' /></td>
    <td width='33%'><img src='https://github.com/dronperminov/DCGAN/blob/master/examples/cats_losses.jpg' /></td>
  </tr>
</table>

## Architecture

### Generator
```python
generator = keras.Sequential([
    layers.Dense(4 * 4 * 1024, input_shape=(latent_dim,)),
    layers.LeakyReLU(alpha=0.2),
    layers.Reshape((4, 4, 1024)),

    layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init),
    layers.LeakyReLU(alpha=0.2),

    layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init),
    layers.LeakyReLU(alpha=0.2),

    layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init),
    layers.LeakyReLU(alpha=0.2),

    layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='tanh'),
], name="generator")
```
Total params: 13130627

### Discriminator
```python
discriminator = keras.Sequential([
    layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init),
    layers.LeakyReLU(alpha=0.2),

    layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init),
    layers.LeakyReLU(alpha=0.2),

    layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init),
    layers.LeakyReLU(alpha=0.2),

    layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init),
    layers.LeakyReLU(alpha=0.2),

    layers.Flatten(),
    layers.Dense(1, activation='sigmoid'),
], name="discriminator")
```
Total params: 4314753

## Implemented "hacks"
* noisy labels: 0.0...0.1 for real and 0.9...1.0 for fake images
* weights init from random normal distribution with mean=0 and std=0.02
* different batches for fake and real images in discriminator training
* adam optimizer with beta1=0.5
* tanh activation in generator