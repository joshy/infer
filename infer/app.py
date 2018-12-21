import os

import numpy as np
from flask import Flask, redirect, render_template, request, session, url_for
from flask_dropzone import Dropzone
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image

app = Flask(__name__)
model = None

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/897'
app.config.update(
    UPLOADED_PATH="./uploads",
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE="image",
    DROPZONE_MAX_FILE_SIZE=3,
    DROPZONE_MAX_FILES=1,
    DROPZONE_REDIRECT_VIEW="main",
)

dropzone = Dropzone(app)


def preload_model():
    global model
    model = load_model("./model/ap_lat_model.h5")
    # https://github.com/keras-team/keras/issues/6462
    model._make_predict_function()


@app.route("/", methods=["POST", "GET"])
def main():
    filename = session.get("filename")
    orientation = None
    if filename:
        image = load_img(os.path.join("infer/static/uploads", filename))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        preds = model.predict(image)
        orientation = "lat" if preds[0][0] == 1.0 else "ap"
    return render_template("index.html", file_url=filename, orientation=orientation)


@app.route("/uploads", methods=["GET", "POST"])
def uploads():
    if request.method == "POST":
        f = request.files.get("file")
        f.save(os.path.join("infer/static/uploads", f.filename))
        session["filename"] = f.filename
    return redirect(url_for("main"))
