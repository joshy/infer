import os

from pathlib import Path

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img


def save(image_file, data=None):
    image_file.save(os.path.join("infer/static/uploads", image_file.filename))
    image = load_img(os.path.join("infer/static/uploads", image_file.filename))
    image = img_to_array(image)

    if data:
        data["image"] = image_file.filename

    return image, data
