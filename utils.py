import os
import platform
from os.path import isdir

import matplotlib as mpl
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


def mpl_init():
    if platform.system() == "Darwin":
        return
    mpl.use("Agg")  # Disable the need for X window environment
