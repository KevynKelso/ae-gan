import os
import re
from glob import glob
from os.path import isdir

import imageio
from PIL import Image, ImageDraw
# from tensorflow.keras.models import load_model
from tqdm import tqdm

import utils

# from vae_gan import save_plot

MODEL_NAME = "aegan_inverse1"


def check_dir():
    if not isdir(f"./{MODEL_NAME}/results"):
        os.system(f"mkdir ./{MODEL_NAME}/results")
    if not isdir(f"./{MODEL_NAME}/results/predictions"):
        os.system(f"mkdir ./{MODEL_NAME}/results/predictions")


def main():
    check_dir()
    # dataset = utils.load_real_samples()
    # mfiles = glob(f"./{MODEL_NAME}/models/generators/*generator*.h5")
    # mfiles.sort(key=lambda f: int(re.sub("\D", "", f)))
    # for i, f in enumerate(mfiles):
    # model = load_model(f)
    # y = model(dataset)
    # save_plot(
    # y,
    # 0,
    # n=3,
    # filename=f"./{MODEL_NAME}/results/predictions/gen_predict_e{i+1}.png",
    # )

    files = glob(f"./{MODEL_NAME}/images/*.png")
    files.sort(key=lambda f: int(re.sub("\D", "", f)))

    # for adding the epoch numbers to the images
    # for i, f in enumerate(tqdm(files)):
    # img = Image.open(f)
    # i1 = ImageDraw.Draw(img)
    # i1.text((10, 10), f"EPOCH {i+1}", fill=(0, 0, 0))
    # img.save(f)

    images = []
    for i, f in enumerate(tqdm(files)):
        if i % 5 == 0:
            images.append(imageio.imread(f))
    imageio.mimsave(f"./{MODEL_NAME}/results/training.gif", images, duration=0.5)


if __name__ == "__main__":
    main()
