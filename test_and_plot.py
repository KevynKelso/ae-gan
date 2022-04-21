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
    save_plot(dataset, 0, n=3, filename=f"./{MODEL_NAME}/results/dataset_orig.png")
    mname = f"./{MODEL_NAME}/models/generator_model_{MODEL_NAME}_{epoch}.h5"
    print(f"Loaded model: {mname}")
    generator_model = load_model(mname)
    y = generator_model(dataset)
    print(f"Avg blur output = {get_average_blur(y.numpy())}")
    save_plot(
        y,
        0,
        n=3,
        filename=f"./{MODEL_NAME}/results/{MODEL_NAME}_e{epoch}_output.png",
    )


def plot_losses():
    plt.clf()
    df = pd.read_csv(f"./{MODEL_NAME}/data/alpha_beta_loss_{MODEL_NAME}.csv")
    df2 = df[:1000]
    plt.title("Generator loss")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    print(f"max = {max(df2['ae_loss'])}")
    print(f"min = {min(df2['ae_loss'])}")
    epoch = df2.index / BATCHES_PER_EPOCH
    plt.plot(epoch, df2["ae_loss"], label="Reconstruction Loss")
    plt.plot(epoch, df2["gan_loss"], label="GAN loss orig")
    plt.legend()
    plt.savefig(f"./{MODEL_NAME}/results/ae_loss_generator.png")
    plt.show()

    plt.clf()
    df = pd.read_csv(f"./{MODEL_NAME}/data/general_metrics_{MODEL_NAME}.csv")
    df = df[:1000]
    plt.title("Discriminator loss")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.plot(df.index / BATCHES_PER_EPOCH, df["d_loss_real"], label="Real images")
    plt.plot(df.index / BATCHES_PER_EPOCH, df["d_loss_fake"], label="Fake images")
    plt.legend()
    plt.savefig(f"./{MODEL_NAME}/results/{MODEL_NAME}_discriminator_loss.png")
    plt.show()


def plot_discriminator_accuracy():
    plt.clf()
    df = pd.read_csv(f"./{MODEL_NAME}/data/accuracy_metrics_{MODEL_NAME}.csv")
    df = df[:1000]
    plt.title("Discriminator Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.plot(df["acc_real"] * 100, label="Real images")
    plt.plot(df["acc_fake"] * 100, label="Fake images")
    plt.legend()
    plt.savefig(f"./{MODEL_NAME}/results/{MODEL_NAME}_discriminator_accuracy.png")
    plt.show()


def main():
    test_model(100)
    plot_losses()
    plot_discriminator_accuracy()


if __name__ == "__main__":
    main()

# Script for adding fzf searching to project
# Script for adding basic mpl plot
