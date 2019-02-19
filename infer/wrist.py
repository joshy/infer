import numpy as np

from infer.infer import (
    predict_hardplaster,
    predict_ap_fracture,
    predict_lat_fracture,
    predict_view,
    save_result,
)


def analyze(image):
    if len(image.shape) != 3 or (1024, 512, 3) != image.shape:
        return None
    image = np.expand_dims(image, axis=0)
    hardplaster = predict_hardplaster(image)
    view = predict_view(image)
    if view == "ap":
        fracture = predict_ap_fracture(image)
    else:
        fracture = predict_lat_fracture(image)

    return {"hardplaster": hardplaster, "view": view, "fracture": fracture}

