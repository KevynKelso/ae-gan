from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Dense, Dropout,
                                     Flatten, Input, Lambda, LeakyReLU,
                                     Reshape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from config import LEARNING_RATE

latent_dim = 100

# AE VERSION 2, based almost entirely on discriminator / generator
def ae(in_shape=(80, 80, 3)):
    model = Sequential(name="autoencoder")
    # normal
    model.add(Conv2D(128, (5, 5), padding="same", input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 40x40
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 20x30
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 10x10
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 5x5
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dense(latent_dim, activation="relu"))
    # foundation for 5x5 feature maps
    n_nodes = 100 * 5 * 5
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((5, 5, 100)))
    # upsample to 10x10
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 20x20
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 40x40
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 80x80
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # output layer 80x80x3
    model.add(Conv2D(3, (5, 5), activation="tanh", padding="same"))

    return model


def encoder(in_shape=(80, 80, 3)):
    model = Sequential(name="encoder")
    # normal
    model.add(Conv2D(128, (5, 5), padding="same", input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 40x40
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 20x30
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 10x10
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 5x5
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dense(latent_dim, activation="relu"))

    return model


def decoder():
    model = Sequential(name="decoder")
    n_nodes = 100 * 5 * 5
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((5, 5, 100)))
    # upsample to 10x10
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 20x20
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 40x40
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 80x80
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # output layer 80x80x3
    model.add(Conv2D(3, (5, 5), activation="tanh", padding="same"))

    return model


def discriminator(in_shape=(80, 80, 3)):
    model = Sequential(name="discriminator")
    # normal
    model.add(Conv2D(128, (5, 5), padding="same", input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 40x40
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 20x30
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 10x10
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # downsample to 5x5
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation="sigmoid"))
    # compile model
    opt = Adam(learning_rate=LEARNING_RATE, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model
