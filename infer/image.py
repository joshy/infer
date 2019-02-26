import logging
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image


def save(image_file, data=None):
    image_file.save(os.path.join("infer/static/uploads", image_file.filename))
    image = load_img(os.path.join("infer/static/uploads", image_file.filename))
    image = img_to_array(image)

    if data:
        data["image"] = image_file.filename

    return image, data


def retrieve(instance):
    url = f"http://localhost:8042/instances/{instance}/preview"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = np.asarray(img, dtype='float32')
    logging.debug(f"Before converting got shape: {img.shape}")
    img = _convert(img)
    logging.debug(f"After converting got shape: {img.shape}")
    img = Image.fromarray(img).resize((224,224))
    img = np.array(img)
    logging.debug(f"After resizing got shape: {img.shape}")
    return img


def _convert(image):
    y, x, _ = image.shape
    startx = x//2-(512//2)
    starty = y//2-(1024//2)
    arr = image[starty:starty+1024,startx:startx+512]

    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(arr,0) / image.max()) * 255.0

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)
    return image_2d_scaled
