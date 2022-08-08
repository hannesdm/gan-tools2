from keras import Sequential
from keras.datasets import mnist
from keras.layers import Conv2D, LeakyReLU, Dropout, BatchNormalization, Dense, Reshape, UpSampling2D, Activation, \
    Flatten
import numpy as np
from core import gan, vis

mnist_channels = 1
mnist_img_shape = (28, 28, mnist_channels)
mnist_noise_input_shape = (100,)


def mnist_generator_model():
    model = Sequential(name='generator')

    model.add(Dense(128 * 7 * 7, activation="relu", input_shape=mnist_noise_input_shape))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(mnist_channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))
    return model


def mnist_discriminator_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=mnist_img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    #     model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model


def mnist_dcgan():
    return gan.GAN(discriminator=mnist_discriminator_model(), generator=mnist_generator_model())


if __name__ == '__main__':
    (X_train, _), (_, _) = mnist.load_data()
    X_train = X_train / 127.5 - 1.
    X_train = np.expand_dims(X_train, -1)
    gan = mnist_dcgan()
    gan.train(X_train, batch_size=32)
    vis.show_gan_image_predictions(gan, 32)
