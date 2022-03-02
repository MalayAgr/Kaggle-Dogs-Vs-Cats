from PIL import Image
import glob
import itertools

from src import config
import os


def resize_images() -> None:
    h, w = config.IMG_HEIGHT, config.IMG_WIDTH

    train_path = os.path.join(config.DATA_DIR, "train", "*.jpg")
    test_path = os.path.join(config.DATA_DIR, "test1", "*.jpg")

    for img_path in itertools.chain(glob.glob(train_path), glob.glob(test_path)):
        img = Image.open(img_path)

        if img.height != h or img.width != w:
            img = img.resize((w, h), resample=Image.BILINEAR)
            img.save(img_path)


resize_images()
