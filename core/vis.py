import numpy as np
import matplotlib.pyplot as plt


def image_grid(array, columns):
    if len(array.shape) == 3:
        nr, height, width = array.shape
        channels = 1
    else:
        nr, height, width, channels = array.shape
    rows = nr // columns
    assert nr == rows * columns  # otherwise not a rectangle
    result = array.reshape(rows, columns, height, width, channels) \
        .swapaxes(1, 2) \
        .reshape(height * rows, width * columns, channels)
    return result


def show_image_grid(array, columns):
    grid = image_grid(array, columns)
    plt.imshow(grid)


def show_gan_image_predictions(gan, nr, columns=8, image_shape=None):
    images = gan.sample_generator(nr)
    if image_shape is not None:
        if len(image_shape) == 2:
            image_shape = image_shape + (1,)
        images = images.reshape(-1, *image_shape[0:])
    grid = image_grid(images, columns)
    grid = 0.5 * grid + 0.5
    plt.imshow(np.squeeze(grid))
    plt.show()
