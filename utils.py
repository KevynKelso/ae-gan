import os
import platform
import sys
from os.path import isdir, isfile

import cv2
import matplotlib as mpl
import numpy as np
from PIL import Image

from architecture import decoder, encoder

if platform.system() != "Darwin":
    mpl.use("Agg")  # Disable the need for X window environment
from matplotlib import pyplot as plt

from config import MODEL_NAME


def add_dirs():
    if not isdir(f"./{MODEL_NAME}"):
        os.system(f"mkdir {MODEL_NAME}")
        # data dir
        if not isdir(f"./{MODEL_NAME}/data"):
            os.system(f"mkdir {MODEL_NAME}/data")
        # images dir
        if not isdir(f"./{MODEL_NAME}/images"):
            os.system(f"mkdir {MODEL_NAME}/images")
            os.system(f"mkdir {MODEL_NAME}/images/training")
            os.system(f"mkdir {MODEL_NAME}/images/validation")
        # models dir
        if not isdir(f"./{MODEL_NAME}/models"):
            os.system(f"mkdir {MODEL_NAME}/models")
        # results dir
        if not isdir(f"./{MODEL_NAME}/results"):
            os.system(f"mkdir {MODEL_NAME}/results")


def add_csv_headers():
    with open(f"./{MODEL_NAME}/data/alpha_beta_loss_{MODEL_NAME}.csv", "w") as f:
        f.write("ae_loss,gan_loss\n")
    with open(f"./{MODEL_NAME}/data/general_metrics_{MODEL_NAME}.csv", "w") as f:
        f.write("epoch,batch,d_loss_real,d_loss_fake,g_loss\n")
    with open(f"./{MODEL_NAME}/data/accuracy_metrics_{MODEL_NAME}.csv", "w") as f:
        f.write("acc_real,acc_fake\n")


def save_plot(examples, epoch, n=10, filename="", show=False):
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis("off")
        # plot raw pixel data
        plt.imshow(examples[i])
    # save plot to file
    if filename == "":
        filename = f"./{MODEL_NAME}/images/training/generated_plot_e{epoch+1}.png"
    plt.savefig(filename)
    if show:
        plt.show()
    plt.close()


def _variance_of_laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


def get_blur_factor(img_pil):
    img_cv = np.array(img_pil)
    img_cv = img_cv[:, :, ::-1].copy()
    return int(_variance_of_laplacian(img_cv))


def get_average_blur(imgs):
    summation = 0
    for img in imgs:
        img_pil = Image.fromarray(((img + 1) / 2.0 * 255).astype(np.uint8))
        summation += get_blur_factor(img_pil)

    return summation / len(imgs)


def load_real_samples(filename="img_align_celeba.npz"):
    # load the face dataset
    data = np.load(filename)
    X = data["arr_0"]
    # convert from unsigned ints to floats
    X = X.astype("float32")
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X


def fatal_check_is_file(fname):
    if not isfile(fname):
        print(f"NOT FILE: {fname}")
        sys.exit(1)
    return True


# NOTE: Changing the model architecture, might have to change this function.
# Encoder and decoder models will def need to be changed.
def split_ae_generator_v2(generator_model):
    m_encoder = encoder()
    m_decoder = decoder()
    i = 0
    # generator_model.summary()
    for i in range(len(m_encoder.layers)):
        m_encoder.layers[i].set_weights(generator_model.layers[i].get_weights())
    k = 0
    for j in range(i + 1, i + 1 + len(m_decoder.layers)):
        m_decoder.layers[k].set_weights(generator_model.layers[j].get_weights())
        k += 1

    return m_encoder, m_decoder
