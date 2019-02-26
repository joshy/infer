import sqlite3


def load(conn, accession_number):
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        "SELECT * FROM wrist_fracture WHERE AccessionNumber=?", (accession_number,)
    )
    result = c.fetchall()
    return result


def load_by_series_uid(conn, series_uid):
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM wrist_fracture WHERE SeriesInstanceUID=?", (series_uid,))
    result = c.fetchall()
    return result


def store(conn, study, series, ml_fracture):
    sql = "INSERT INTO wrist_fracture VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    values = _extract(study, series, ml_fracture)
    c = conn.cursor()
    c.execute(sql, values)
    conn.commit()
    return True


def _extract(study, series, ml_fracture):
    patient_tags = study["PatientMainDicomTags"]
    study_tags = study["MainDicomTags"]
    series_tags = series["MainDicomTags"]
    return (
        patient_tags["PatientID"],
        patient_tags["PatientName"],
        patient_tags["PatientSex"],
        study_tags["AccessionNumber"],
        study_tags["StudyDate"],
        study_tags["StudyDescription"],
        series_tags["SeriesNumber"],
        series_tags["SeriesDescription"],
        series_tags["SeriesInstanceUID"],
        ml_fracture["hardplaster"],
        ml_fracture["hardplaster_version"],
        ml_fracture["view"],
        ml_fracture["view_version"],
        ml_fracture["fracture_probability"],
        ml_fracture["fracture_probability_version"],
    )
