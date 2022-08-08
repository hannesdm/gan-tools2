
import tensorflow as tf
from tqdm.auto import tqdm

from core import losses, vis


class GAN:

    def __init__(self, generator, discriminator, generator_optimizer=None,
                 discriminator_optimizer=None):

        self.generator = generator
        self.discriminator = discriminator
        if generator_optimizer is None:
            generator_optimizer = tf.keras.optimizers.Adam(beta_1=0.5,
                                                           learning_rate=0.0002)
        self.generator_optimizer = generator_optimizer
        if discriminator_optimizer is None:
            discriminator_optimizer = tf.keras.optimizers.Adam(beta_1= 0.5,
                                                               learning_rate=0.0002)
        self.discriminator_optimizer = discriminator_optimizer
        self.z_dim = generator.input_shape[1]

        # training statistics
        self.d_losses = []
        self.g_losses = []
        self.d_accs = []
        self.g_accs = []


    def sample_generator(self, nr, batch_size=None):
        noise = tf.random.normal([nr, self.z_dim])
        return self.generator.predict(noise, batch_size=batch_size, verbose=0)


    @tf.function
    def train_step(self, batch_size, x_batches):
        noise = tf.random.normal([batch_size, self.z_dim])
        real_batch = x_batches

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_batch = self.generator(noise)

            real_y = self.discriminator(real_batch)
            fake_y = self.discriminator(generated_batch)

            gen_loss = losses.cross_entropy_generator_loss(fake_y)
            disc_loss = losses.cross_entropy_discriminator_loss(real_y, fake_y)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        gen_acc = losses.generator_accuracy(fake_y)
        dis_acc = losses.discriminator_accuracy(real_y, fake_y)

        return gen_loss, disc_loss, gen_acc, dis_acc


    def train(self, X, batch_size=32, steps=20000, log_interval=1, plot_interval=1000, image_shape=None):

        dataset = tf.data.Dataset.from_tensor_slices(X).repeat().shuffle(X.shape[0]).batch(batch_size).take(steps).prefetch(tf.data.AUTOTUNE)

        with tqdm(total=steps) as pbar:
            for i, x_batches in enumerate(dataset):
                gen_loss, disc_loss, gen_acc, dis_acc = self.train_step(batch_size, x_batches=x_batches)
                gen_loss, disc_loss, gen_acc, dis_acc = gen_loss.numpy(), disc_loss.numpy(), gen_acc.numpy(), dis_acc.numpy()

                if log_interval != 0 and (i % log_interval == 0):
                    pbar.set_description("Batch " + str(i + 1) + ",  " + " Discriminator loss: " + str(round(disc_loss, 6)) +
                                             " Discriminator acc: " + str(round(dis_acc, 6)) +
                                             " Generator loss: " + str(round(gen_loss, 6)) +
                                             " Generator acc: " + str(round(gen_acc, 6)))
                    self.d_losses.append(disc_loss)
                    self.g_losses.append(gen_loss)
                    self.d_accs.append(dis_acc)
                    self.g_accs.append(gen_acc)
                if plot_interval != 0 and (i % plot_interval == 0):
                    vis.show_gan_image_predictions(self, 32, image_shape=image_shape)

                pbar.update()


class WGAN(GAN):

    def __init__(self, generator, discriminator, generator_optimizer=None,
                 discriminator_optimizer=None, n_critic=5, clip_value=0.01):
        super().__init__(generator, discriminator, generator_optimizer, discriminator_optimizer)

        self.n_critic = n_critic
        self.clip_value = clip_value


    def update_discriminator(self, noise, real_batch):
        generated_batch = self.generator(noise)
        with tf.GradientTape() as disc_tape:
            real_y = self.discriminator(real_batch)
            fake_y = self.discriminator(generated_batch)
            disc_loss = tf.reduce_mean(real_y) - tf.reduce_mean(fake_y)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        for w in self.discriminator.trainable_variables:
            w.assign(tf.clip_by_value(w, -self.clip_value, self.clip_value))

        return disc_loss

    def update_generator(self, noise):
        with tf.GradientTape() as gen_tape:
            generated_batch = self.generator(noise)
            fake_y = self.discriminator(generated_batch)
            gen_loss = tf.reduce_mean(fake_y)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return gen_loss


    @tf.function
    def train_step(self, batch_size, x_batches):
        disc_loss = 0.0
        for real_batch in x_batches:
            noise = tf.random.normal([batch_size, self.z_dim])
            disc_loss += self.update_discriminator(noise, real_batch)

        noise = tf.random.normal([batch_size, self.z_dim])
        gen_loss = self.update_generator(noise)
        return gen_loss, disc_loss / self.n_critic


    def train(self, X, batch_size=32, steps=20000, log_interval=1, plot_interval=1000, image_shape=None):

        dataset = tf.data.Dataset.from_tensor_slices(X).repeat().shuffle(X.shape[0]).batch(batch_size).take(steps*self.n_critic).prefetch(tf.data.AUTOTUNE)
        dataset = iter(dataset)

        with tqdm(total=steps) as pbar:
            for i in range(steps):
                x_batches = [dataset.get_next() for _ in range(self.n_critic)]

                gen_loss, disc_loss = self.train_step(batch_size, x_batches=x_batches)
                gen_loss, disc_loss = gen_loss.numpy(), disc_loss.numpy()

                if log_interval != 0 and (i % log_interval == 0):
                    pbar.set_description("Batch " + str(i + 1) + ",  " + " Discriminator loss: " + str(round(disc_loss, 6)) +
                                             " Generator loss: " + str(round(gen_loss, 6)))
                    self.d_losses.append(disc_loss)
                    self.g_losses.append(gen_loss)
                if plot_interval != 0 and (i % plot_interval == 0):
                    vis.show_gan_image_predictions(self, 32, image_shape=image_shape)

                pbar.update()



class WGAN_GP(WGAN):

    def __init__(self, generator, discriminator, generator_optimizer=None, discriminator_optimizer=None,n_critic=5, lam=10.0):
        super().__init__(generator, discriminator, generator_optimizer, discriminator_optimizer, n_critic)

        self.lam = lam


    def gradient_penalty(self, x, x_gen):
        epsilon = tf.random.uniform([x.shape[0]] + [1] * (len(x.shape) - 1), 0.0, 1.0)
        x_hat = epsilon * x + (1 - epsilon) * x_gen
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.discriminator(x_hat)
        gradients = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=list(range(1, len(x.shape)))))  # L2
        d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)  # expectation
        return d_regularizer

    def update_discriminator(self, noise, real_batch):
        generated_batch = self.generator(noise, training=True)
        with tf.GradientTape() as disc_tape:
            real_y = self.discriminator(real_batch, training=True)
            fake_y = self.discriminator(generated_batch, training=True)
            d_regularizer = self.gradient_penalty(real_batch, generated_batch)
            disc_loss = tf.reduce_mean(real_y) - tf.reduce_mean(fake_y) + self.lam * d_regularizer
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return disc_loss
