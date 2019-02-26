import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import requests
from flask import (
    Flask,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_dropzone import Dropzone
from orthanc_rest_client import Orthanc

import orthanc.wrist as wrist
from infer.convert import convert
from infer.image import retrieve, save
from infer.results import list
from infer.db import load, store, load_by_series_uid

DATABASE = "wrist.db"

app = Flask(__name__, instance_relative_config=True)
app.config.from_object("infer.default_config")
app.config.from_pyfile("config.cfg")
version = app.config["VERSION"] = "0.0.1"

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


orthanc = Orthanc("http://localhost:8042")


def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db


def init_db():
    with app.app_context():
        db = get_db()
        with app.open_resource("schema/schema.sql", mode="r") as f:
            db.cursor().executescript(f.read())
        db.commit()


init_db()


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


@app.route("/")
def MainDicomTags():
    r = wrist.find_wrist_studies(orthanc)
    return render_template("orthanc.html", results=r)


@app.route("/study")
def s():
    study_id = request.args.get("id", "")
    if not study_id:
        return 400, "no study id given"

    study, series = wrist.find_wrist_study(orthanc, study_id)
    accession_number = study["MainDicomTags"]["AccessionNumber"]
    rows = load(get_db(), accession_number)
    for i in series:
        if i["MainDicomTags"]["Modality"] == "CR":
            result = load_by_series_uid(
                get_db(), i["MainDicomTags"]["SeriesInstanceUID"]
            )
            if not result:
                i["ml_fracture"] = _run_inference(i)
                store(get_db(), study, i, i["ml_fracture"])
            else:
                i["ml_fracture"] = {
                    "hardplaster": result[0]["ml_fracture_hardplaster"],
                    "view": result[0]["ml_fracture_view"],
                    "fracture_probability": result[0]["ml_fracture_probability"],
                }
    return render_template("study.html", study=study, series=series)


def _run_inference(i):
    img_array = retrieve(i.get("Instances")[0])
    data = {"image": img_array.tolist(), "bla": "blub"}
    r = requests.post("http://localhost:9556/inference", json=data)
    return r.json()
