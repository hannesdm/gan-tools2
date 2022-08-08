import tensorflow as tf
from tensorflow.keras import layers

from core import gan, vis


def mnist_generator_model():
    generator = tf.keras.Sequential()
    generator.add(layers.Dense(512, input_dim=100))
    generator.add(layers.LeakyReLU(0.2))
    generator.add(layers.Dense(512))
    generator.add(layers.LeakyReLU(0.2))
    generator.add(layers.Dense(512))
    generator.add(layers.LeakyReLU(0.2))
    generator.add(layers.Dense(784, activation='tanh'))
    return generator


def mnist_discriminator_model():
    discriminator = tf.keras.Sequential()
    discriminator.add(layers.Dense(512, input_dim=784))
    discriminator.add(layers.LeakyReLU(0.2))
    discriminator.add(layers.Dense(512))
    discriminator.add(layers.LeakyReLU(0.2))
    discriminator.add(layers.Dense(512))
    discriminator.add(layers.LeakyReLU(0.2))
    discriminator.add(layers.Dense(1))
    return discriminator



def mnist_gan():
    return gan.WGAN(discriminator=mnist_discriminator_model(), generator=mnist_generator_model(), n_critic=10, clip_value=0.01)


if __name__ == '__main__':
    (X_train_mnist, Y_train_mnist), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train_mnist = X_train_mnist.reshape((-1, 28 * 28))
    X_train_mnist = X_train_mnist.astype('float32') / 127.5 - 1
    gan = mnist_gan()
    gan.train(X_train_mnist, steps=20000, batch_size=64, plot_interval=500, image_shape=(28, 28))
    vis.show_gan_image_predictions(gan, 32, image_shape=(28, 28))
