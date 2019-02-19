import json
from pathlib import Path

import numpy as np
from keras.models import load_model
from keras import backend as K

global hardplaster_model
global ap_lat_model
global ap_fracture_model
global lat_fracture_model


def _preload(path):
    K.clear_session()
    model = load_model(path)
    model._make_predict_function()
    print(f"Loaded model at path: {path}")
    return model


def preload_models():
    global hardplaster_model
    global ap_lat_model
    global ap_fracture_model
    global lat_fracture_model
    hardplaster_model = _preload("./model/hardplaster_resnet.h5")
    ap_lat_model = _preload("./model/ap_lat_model.h5")
    ap_fracture_model = _preload("./model/ap_center_cropped_resnet.h5")
    lat_fracture_model = _preload("./model/lat_center_cropped_resnet.h5")


def predict_hardplaster(image):
    if len(image.shape) != 3 or (1024, 512, 3) != image.shape:
        print(f"Image not in correct shape, got shape: {image.shape}")
        return None
    image = np.expand_dims(image, axis=0)
    return hardplaster_model.predict(image)


def predict_view(image):
    if len(image.shape) != 3 or (1024, 512, 3) != image.shape:
        print(f"Image not in correct shape, got shape: {image.shape}")
        return None
    image = np.expand_dims(image, axis=0)
    preds = ap_lat_model.predict(image)
    return "lat" if preds[0][0] == 1.0 else "ap"


def predict_ap_fracture(image):
    image = np.expand_dims(image, axis=0)
    pred = ap_fracture_model.predict(image)
    return pred[0][0]


def predict_lat_fracture(image):
    image = np.expand_dims(image, axis=0)
    pred = lat_fracture_model.predict(image)
    return pred[0][0]


def save_result(data):
    filename = Path("results", data["key"]).with_suffix(".json")
    with open(filename, "w") as outfile:
        json.dump(data, outfile)
