import os
import platform
from os.path import isdir

import cv2
import matplotlib as mpl
import numpy as np
from PIL import Image

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
        # models dir
        if not isdir(f"./{MODEL_NAME}/models"):
            os.system(f"mkdir {MODEL_NAME}/models")
        # results dir
        if not isdir(f"./{MODEL_NAME}/results"):
            os.system(f"mkdir {MODEL_NAME}/results")


def add_csv_headers():
    with open(f"./{MODEL_NAME}/data/alpha_beta_loss_{MODEL_NAME}.csv", "w") as f:
        f.write("ae_loss,gan_loss")
    with open(f"./{MODEL_NAME}/data/general_metrics_{MODEL_NAME}.csv", "w") as f:
        f.write("epoch,batch,d_loss_real,d_loss_fake,g_loss")
    with open(f"./{MODEL_NAME}/data/accuracy_metrics_{MODEL_NAME}.csv", "w") as f:
        f.write("acc_real,acc_fake")


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
        filename = f"./{MODEL_NAME}/images/generated_plot_e{epoch+1}.png"
    if show:
        plt.show()
    plt.savefig(filename)
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


def load_real_samples():
    # load the face dataset
    data = np.load("img_align_celeba.npz")
    X = data["arr_0"]
    # convert from unsigned ints to floats
    X = X.astype("float32")
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X
