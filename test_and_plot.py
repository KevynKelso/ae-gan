import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from ae_gan import load_real_samples
from utils import save_plot

# from pyfzf.pyfzf import FzfPrompt

# fzf = FzfPrompt()

MODEL_NAME = "aegan_inverse1"


def test_model(epoch):
    plt.clf()
    dataset = load_real_samples()
    save_plot(dataset, 0, n=3, filename="dataset_orig.png", show=True)
    mname = f"./{MODEL_NAME}/models/generator_model_{MODEL_NAME}_{epoch}.h5"
    print(f"Loaded model: {mname}")
    generator_model = load_model(mname)
    y = generator_model(dataset)
    save_plot(y, 0, n=3, filename=f"{MODEL_NAME}_e{epoch}_output.png", show=True)


def plot_losses():
    plt.clf()
    df = pd.read_csv(f"./{MODEL_NAME}/data/alpha_beta_loss_{MODEL_NAME}.csv")
    plt.title("")
    plt.xlabel("1/2 batch")
    plt.ylabel("loss")
    print(f"max = {max(df['ae_loss'])}")
    print(f"min = {min(df['ae_loss'])}")
    test = np.linspace(0.3, 0, 1500)
    plt.plot(test, label="Reconstruction Loss")
    plt.plot(df["gan_loss"][:1500] * 0.0005 * (1 / test), label="GAN loss")
    plt.legend()
    plt.show()
    plt.savefig(f"./{MODEL_NAME}/results/ae_loss_generator.png")

    plt.clf()
    df = pd.read_csv(f"./{MODEL_NAME}/data/general_metrics_{MODEL_NAME}.csv")
    plt.title("Discriminator loss")
    plt.xlabel("1/2 batch")
    plt.ylabel("loss")
    plt.plot(df["d_loss_real"][:1500], label="Real images")
    plt.plot(df["d_loss_fake"][:1500], label="Fake images")
    plt.legend()
    plt.show()
    plt.savefig(f"./{MODEL_NAME}/results/ae_loss_discriminator.png")


def plot_discriminator_accuracy():
    plt.clf()
    df = pd.read_csv(f"./{MODEL_NAME}/data/accuracy_metrics_{MODEL_NAME}.csv")
    plt.title("Discriminator Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.plot(df["acc_real"] * 100, label="Real images")
    plt.plot(df["acc_fake"] * 100, label="Fake images")
    plt.legend()
    plt.show()


def main():
    # test_model(23)
    plot_losses()
    plot_discriminator_accuracy()


if __name__ == "__main__":
    main()

# Script for adding fzf searching to project
# Script for adding basic mpl plot
