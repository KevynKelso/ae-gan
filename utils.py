import os
from os.path import isdir

from matplotlib import pyplot as plt


def add_dirs(model_name):
    if not isdir(f"./{model_name}"):
        os.system(f"mkdir {model_name}")
        if not isdir(f"./{model_name}/data"):
            os.system(f"mkdir {model_name}/data")
        if not isdir(f"./{model_name}/images"):
            os.system(f"mkdir {model_name}/images")
        if not isdir(f"./{model_name}/models"):
            os.system(f"mkdir {model_name}/models")


def save_plot(model_name, examples, epoch, n=10, filename="", show=False):
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
        filename = f"./{model_name}/images/generated_plot_e{epoch+1}.png"
    if show:
        plt.show()
    plt.savefig(filename)
    plt.close()
