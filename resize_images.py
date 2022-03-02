import glob
import itertools
import os

from PIL import Image

from src import config


def resize_images() -> None:
    h, w = config.IMG_HEIGHT, config.IMG_WIDTH

    train_path = os.path.join(config.DATA_DIR, "train", "*.jpg")
    test_path = os.path.join(config.DATA_DIR, "test1", "*.jpg")

    for img_path in itertools.chain(glob.glob(train_path), glob.glob(test_path)):
        with Image.open(img_path) as img:
            if img.height != h or img.width != w:
                img = img.resize((w, h), resample=Image.BILINEAR)
                img.save(img_path)


resize_images()
