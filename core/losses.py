import tensorflow as tf

cross_entropy_from_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def cross_entropy_discriminator_loss(real_output, fake_output, logits=True):
    if logits:
        real_loss = cross_entropy_from_logits(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy_from_logits(tf.zeros_like(fake_output), fake_output)
    else:
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = (real_loss + fake_loss) / 2
    return total_loss


def cross_entropy_generator_loss(fake_output, logits=True):
    if logits:
        loss = cross_entropy_from_logits(tf.ones_like(fake_output), fake_output)
    else:
        loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return loss


def generator_accuracy(fake_output, logits=True):
    if logits:
        fake_output = tf.nn.sigmoid(fake_output)
    acc = tf.reduce_mean(tf.cast(tf.cast(fake_output > 0.5, float) == tf.ones_like(fake_output), float))
    return acc


def discriminator_accuracy(real_output, fake_output, logits=True):
    if logits:
        fake_output = tf.nn.sigmoid(fake_output)
        real_output = tf.nn.sigmoid(real_output)

    acc_real = tf.reduce_mean(tf.cast(tf.cast(real_output > 0.5, float) == tf.ones_like(real_output), float))
    acc_fake = tf.reduce_mean(tf.cast(tf.cast(fake_output > 0.5, float) == tf.zeros_like(fake_output), float))
    return (acc_real + acc_fake) / 2.

