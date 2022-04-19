import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model

from utils import get_average_blur, load_real_samples, save_plot

# from pyfzf.pyfzf import FzfPrompt

# fzf = FzfPrompt()

MODEL_NAME = "aegan_inverse1"
BATCHES_PER_EPOCH = 195


def test_model(epoch):
    plt.clf()
    dataset = load_real_samples()
    print(f"Avg blur dataset = {get_average_blur(dataset)}")
    save_plot(dataset, 0, n=3, filename="dataset_orig.png", show=True)
    mname = f"./{MODEL_NAME}/models/generator_model_{MODEL_NAME}_{epoch}.h5"
    print(f"Loaded model: {mname}")
    generator_model = load_model(mname)
    y = generator_model(dataset)
    print(f"Avg blur output = {get_average_blur(y.numpy())}")
    # TODO: get average blur dataset, and y
    save_plot(y, 0, n=3, filename=f"{MODEL_NAME}_e{epoch}_output.png", show=True)


def plot_losses():
    plt.clf()
    df = pd.read_csv(f"./{MODEL_NAME}/data/alpha_beta_loss_{MODEL_NAME}.csv")
    df2 = df
    plt.title("")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    # print(f"max = {max(df2['ae_loss'])}")
    # print(f"min = {min(df2['ae_loss'])}")
    epoch = df2.index / BATCHES_PER_EPOCH
    # need derivative of the ae_loss
    derivative = df2["ae_loss"].diff(periods=50) / df2[
        "ae_loss"
    ].index.to_series().diff(periods=50)
    # plt.plot(epoch, derivative, label="derivative")

    # beta = abs(-0.00000002 / derivative)
    beta = abs(-0.0000002 / derivative)
    beta = beta.apply(lambda x: min(x, 1)).fillna(0)
    # plt.plot(epoch, beta, label="beta")
    # plt.legend()
    # print(beta)
    # plt.show()
    plt.plot(epoch, df2["ae_loss"], label="Reconstruction Loss")
    plt.plot(epoch, df2["gan_loss"], label="GAN loss orig")
    plt.plot(epoch, df2["gan_loss"] * .0005, label="GAN loss effective")
    plt.plot(epoch, df2["gan_loss"] * beta, label="GAN loss scaled")
    plt.legend()
    plt.show()
    return
    plt.savefig(f"./{MODEL_NAME}/results/ae_loss_generator.png")

    plt.clf()
    df = pd.read_csv(f"./{MODEL_NAME}/data/general_metrics_{MODEL_NAME}.csv")
    plt.title("Discriminator loss")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.plot(df.index / BATCHES_PER_EPOCH, df["d_loss_real"], label="Real images")
    plt.plot(df.index / BATCHES_PER_EPOCH, df["d_loss_fake"], label="Fake images")
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
    # test_model(100)
    plot_losses()
    # plot_discriminator_accuracy()


if __name__ == "__main__":
    main()

# Script for adding fzf searching to project
# Script for adding basic mpl plot
