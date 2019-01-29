import os
from pathlib import Path

import numpy as np
from flask import (Flask, jsonify, redirect, render_template, request, session,
                   url_for)
from flask_dropzone import Dropzone
from keras.preprocessing.image import img_to_array, load_img

from infer.convert import convert
from infer.image import save
from infer.infer import predict_ap_fracture, predict_lat_fracture, predict_view

# Use cpu, gpu is no nedded for inference
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)


app.secret_key = b'_5#y2L"F4Q8z\n\xec]/897'
app.config.update(
    UPLOADED_PATH="./uploads",
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE="image",
    DROPZONE_MAX_FILE_SIZE=3,
    DROPZONE_MAX_FILES=1,
    DROPZONE_REDIRECT_VIEW="example_view",
)

dropzone = Dropzone(app)



@app.route("/example/view", methods=["POST", "GET"])
def example_view():
    filename = session.get("filename")
    orientation = None
    if filename:
        image = load_img(os.path.join("infer/static/uploads", filename))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        preds = predict_view(image)
        orientation = "lat" if preds[0][0] == 1.0 else "ap"
    return render_template("example.html", file_url=filename, orientation=orientation)


@app.route("/uploads", methods=["GET", "POST"])
def uploads():
    if request.method == "POST":
        f = request.files.get("file")
        f.save(os.path.join("infer/static/uploads", f.filename))
        session["filename"] = f.filename
    return redirect(url_for("example_view"))


@app.route("/inference/wrist/view", methods=["POST"])
def infer_view():
    image_file = request.files.get("image")
    image = save(image_file)
    image = convert(image)
    image = np.expand_dims(image, axis=0)
    preds = predict_view(image)
    view = "lat" if preds[0][0] == 1.0 else "ap"
    return jsonify({"view": view})


@app.route("/inference/wrist/fracture", methods=["POST"])
def infer_wrist_fracture():
    view = request.form["view"]
    print("got view", view)
    if not view:
        return "Request query parameter >view< is missing", 400

    image_file = request.files.get("image")
    image = save(image_file)
    x, y, _ = image.shape
    print(x,y)
    if not all([x==224, y==224]):
        return "Image is not size 224x224", 400

    if view == "ap":
        preds = predict_ap_fracture(image)
        return jsonify({"fracture": str(preds)})
    elif view == "lat":
        preds = predict_lat_fracture(image)
        return jsonify({"fracture": str(preds)})
    else:
        return "Passed view is not >lat< or >ap<", 400
