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


def add_dirs(model_name):
    if not isdir(f"./{model_name}"):
        os.system(f"mkdir {model_name}")
        # data dir
        if not isdir(f"./{model_name}/data"):
            os.system(f"mkdir {model_name}/data")
        # images dir
        if not isdir(f"./{model_name}/images"):
            os.system(f"mkdir {model_name}/images")
        # models dir
        if not isdir(f"./{model_name}/models"):
            os.system(f"mkdir {model_name}/models")
        # results dir
        if not isdir(f"./{model_name}/results"):
            os.system(f"mkdir {model_name}/results")


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
