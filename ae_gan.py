import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import ones, zeros
from numpy.random import randint
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.engine import data_adapter

import architecture
import utils
from config import LEARNING_RATE, MODEL_NAME

# TODO: add csv headers automatically


def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return X, y


def summarize_performance(epoch, g_model, d_model, dataset, n_samples=100):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, n_samples, dataset)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print(">Accuracy real: %.0f%%, fake: %.0f%%" % (acc_real * 100, acc_fake * 100))
    with open(f"./{MODEL_NAME}/data/accuracy_metrics_{MODEL_NAME}.csv", "a") as f:
        f.write(f"{acc_real},{acc_fake}\n")
    # save plot
    utils.save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = f"./{MODEL_NAME}/models/generator_model_{MODEL_NAME}_{epoch+1}.h5"
    g_model.save(filename)
    filename = f"./{MODEL_NAME}/models/discriminator_model_{MODEL_NAME}_{epoch+1}.h5"
    d_model.save(filename)


def generate_fake_samples(vae_model, n_samples, dataset):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    x_input = dataset[ix]
    # use VAE to reconstruct some dataset images
    X = vae_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))

    return X, y


def train(ae_model, d_model, gan_model, dataset, n_epochs=100, n_batch=256):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss_real, _ = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(ae_model, half_batch, dataset)
            # update discriminator model weights
            d_loss_fake, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            ix = randint(0, dataset.shape[0], n_batch)
            X_gan = dataset[ix]
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan, return_dict=True)
            g_loss = g_loss["loss"]
            # TODO: accuracy should be evaluated on evey batch
            # summarize loss on this batch
            print(
                f">{i+1}, {j+1}/{bat_per_epo}, d_loss_real={d_loss_real:.3f}, d_loss_fake={d_loss_fake:.3f}, g={g_loss:.3f}"
            )
            # epoch, batch, d_loss_real, d_loss_fake, g_loss
            general_metrics = f"{i+1},{j+1},{d_loss_real},{d_loss_fake},{g_loss}\n"
            with open(
                f"./{MODEL_NAME}/data/general_metrics_{MODEL_NAME}.csv", "a"
            ) as f:
                f.write(general_metrics)
        # evaluate the model performance, sometimes
        # if (i + 1) % 10 == 0:
        summarize_performance(i, ae_model, d_model, dataset)


def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False

    model = VAEGAN(g_model)
    model.add(g_model)
    model.add(d_model)
    # compile model
    opt = Adam(learning_rate=LEARNING_RATE, beta_1=0.5)
    # TODO: scalar should increase with lower reconstruction loss
    model.compile(my_loss=loss_wapper(g_model, 1), optimizer=opt)

    return model


def loss_wapper(g_model, alpha):
    mse = MeanSquaredError()
    bce = BinaryCrossentropy()

    def loss(x, y_true, y_pred):
        # run data through generator
        y = g_model(x)
        # calculate typical loss functions
        ae = mse(x, y)
        gan = bce(y_true, y_pred)

        # scale ae loss and invert
        # ae_loss = tf.math.scalar_mul(alpha, ae)
        # ae_loss_inverted = 1 / ae_loss
        df = pd.read_csv(f"./{MODEL_NAME}/data/alpha_beta_loss_{MODEL_NAME}.csv")
        beta = 0
        if len(df.index) > 50:
            derivative = df["ae_loss"].diff(periods=50) / df[
                "ae_loss"
            ].index.to_series().diff(periods=50)
            beta_vec = abs(-0.0000002 / derivative)
            beta_vec = beta_vec.apply(lambda x: min(x, 0.01)).fillna(0)
            beta = beta_vec.iloc[-1]

        # gan_loss should = gan_loss * 0.0005 * (1/ae_loss) Hopefully that will allow recovery from convergence failure
        gan_loss_scaled = tf.math.scalar_mul(beta, gan)
        # gan_loss = tf.math.multiply(gan_loss_scaled, ae_loss_inverted)
        # record results for analysis
        with open(f"./{MODEL_NAME}/data/alpha_beta_loss_{MODEL_NAME}.csv", "a") as f:
            f.write(f"{ae},{gan},{beta}\n")

        return ae + gan_loss_scaled

    return loss


class VAEGAN(tf.keras.Sequential):
    def compile(self, optimizer, my_loss, run_eagerly=True):
        super().compile(optimizer, run_eagerly=True)
        self.my_loss = my_loss

    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss_value = self.my_loss(x, y, y_pred)

        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {"loss": loss_value}


def main():
    utils.add_dirs()
    utils.add_csv_headers()

    dataset = utils.load_real_samples()
    d_model = architecture.discriminator()
    ae_model = architecture.ae()  # AE model is generator

    gan_model = define_gan(ae_model, d_model)

    train(ae_model, d_model, gan_model, dataset)


if __name__ == "__main__":
    main()
