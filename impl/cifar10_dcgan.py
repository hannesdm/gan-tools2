import tensorflow as tf
from tensorflow.keras import layers
from core import gan, vis

cifar10_channels = 3
cifar10_img_shape = (32, 32, cifar10_channels)
cifar10_noise_input_shape = (100,)


def cifar10_dcgan_generator_model(z_dim):
    model = tf.keras.Sequential(name='generator')
    model.add(layers.Dense(8 * 8 * 256, input_shape=z_dim))
    model.add(layers.Reshape((8, 8, 256)))
    model.add(layers.Conv2DTranspose(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same', activation='tanh'))
    return model


def cifar10_dcgan_discriminator_model(output_activation=None):
    model = tf.keras.Sequential(name='discriminator')
    model.add(layers.Input(shape=cifar10_img_shape))
    model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation=output_activation))
    return model


def cifar10_dcgan():
    return gan.GAN(discriminator=cifar10_dcgan_discriminator_model(), generator=cifar10_dcgan_generator_model(z_dim=cifar10_noise_input_shape))


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    model_class = 1
    (X_train_original, Y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
    X_train_single_class = X_train_original[np.where(np.squeeze(Y_train) == model_class)]
    X_train_single_class = X_train_single_class / 127.5 - 1.
    gan = cifar10_dcgan()
    gan.train(X_train_single_class, batch_size=32, steps=20000)
    vis.show_gan_image_predictions(gan, 32)

    # Plot the final loss curves
    def moving_average(a, n=10):
        s = np.cumsum(a, dtype=float)
        s[n:] = s[n:] - s[:-n]
        return s[n - 1:] / n

    plt.figure(figsize=(16, 12))
    plt.plot(moving_average(gan.d_losses), c="blue", label="D Loss")
    plt.plot(moving_average(gan.g_losses), c="red", label="G Loss")
    plt.plot(moving_average(gan.d_accs), c="green", label="D Accuracy")
    plt.plot(moving_average(gan.g_accs), c="yellow", label="G Accuracy")
    plt.legend(loc="upper left")
    plt.show()
