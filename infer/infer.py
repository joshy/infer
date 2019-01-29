import json
from pathlib import Path

import numpy as np
from keras.models import load_model

global ap_lat_model
global ap_fracture_model
global lat_fracture_model


def _preload(path):
    model = load_model(path)
    model._make_predict_function()
    print(f"Loaded model at path: {path}")
    return model


def preload_models():
    global ap_lat_model
    global ap_fracture_model
    global lat_fracture_model
    ap_lat_model = _preload("./model/ap_lat_model.h5")
    ap_fracture_model = _preload("./model/ap_center_cropped_resnet.h5")
    lat_fracture_model = _preload("./model/lat_center_cropped_resnet.h5")


def predict_view(image):
    return ap_lat_model.predict(image)


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
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)
