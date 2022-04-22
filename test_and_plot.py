import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, load_model

from utils import (fatal_check_is_file, get_average_blur, load_real_samples,
                   save_plot, split_ae_generator_v2)

# from pyfzf.pyfzf import FzfPrompt

# fzf = FzfPrompt()

MODEL_NAME = "aegan_inverse5"
BATCHES_PER_EPOCH = 195
BATCHES_TO_PLOT = 19500


def vector_arithmetic(epoch):
    mname = f"./{MODEL_NAME}/models/generator_model_{MODEL_NAME}_{epoch}.h5"
    fatal_check_is_file(mname)
    generator_model = load_model(mname)
    print(f"Loaded model: {mname}")

    encoder, decoder = split_ae_generator_v2(generator_model)
    dataset = load_real_samples("img_align_celeba2.npz")

    smiling_women = np.asarray([dataset[3], dataset[5], dataset[9]])
    neutral_women = np.asarray([dataset[0], dataset[4], dataset[10]])
    neutral_men = np.asarray([dataset[6], dataset[8], dataset[28]])

    avg_smiling_women_lv = np.mean(encoder.predict(smiling_women), axis=0)
    avg_neutral_women_lv = np.mean(encoder.predict(neutral_women), axis=0)
    avg_neutral_men_lv = np.mean(encoder.predict(neutral_men), axis=0)

    avg_smiling_women = decoder.predict(np.asarray([avg_smiling_women_lv]))[0]
    avg_neutral_women = decoder.predict(np.asarray([avg_neutral_women_lv]))[0]
    avg_neutral_men = decoder.predict(np.asarray([avg_neutral_men_lv]))[0]

    combined_vector = avg_smiling_women_lv - avg_neutral_women_lv + avg_neutral_men_lv
    prediction = decoder.predict(np.asarray([combined_vector]))[0]

    smiling_women = np.vstack((smiling_women, [avg_smiling_women]))
    neutral_women = np.vstack((neutral_women, [avg_neutral_women]))
    neutral_men = np.vstack((neutral_men, [avg_neutral_men]))
    output_row = np.vstack((np.full((3, 80, 80, 3), 255), [prediction]))
    output = np.vstack((smiling_women, neutral_women, neutral_men, output_row))

    save_plot(
        output,
        0,
        n=4,
        filename=f"./{MODEL_NAME}/results/{MODEL_NAME}_vector_arithmetic.png",
        show=True,
    )


def test_model(epoch):
    plt.clf()
    dataset = load_real_samples("validation_img_align_celeba.npz")
    print(f"Avg blur dataset = {get_average_blur(dataset)}")
    save_plot(dataset, 0, n=3, filename=f"./{MODEL_NAME}/results/dataset_orig.png")
    mname = f"./{MODEL_NAME}/models/generator_model_{MODEL_NAME}_{epoch}.h5"
    fatal_check_is_file(mname)
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
    df2 = df[:BATCHES_TO_PLOT]
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
    df = df[:BATCHES_TO_PLOT]
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
    df = df[:BATCHES_TO_PLOT]
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
    vector_arithmetic(100)


if __name__ == "__main__":
    main()

# Script for adding fzf searching to project
# Script for adding basic mpl plot
