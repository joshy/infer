import json


def find_wrist_studies(orthanc):
    search = {
        "Level": "Study",
        "Query": {"PatientID": "*", "StudyDescription": "*Hand*", "PatientName": "*"},
    }
    r = orthanc.find(search)
    rrs = []
    for i in r:
        rrs.append(_find_wrist_study(orthanc, i))
    return rrs


def _find_wrist_study(orthanc, study_id):
    r = orthanc.get_study(study_id)
    return r


def _find_wrist_series(orthanc, series_id):
    r = orthanc.get_one_series(series_id)
    return r


def find_wrist_study(orthanc, study_id):
    r = _find_wrist_study(orthanc, study_id)
    rrs = []
    for i in r["Series"]:
        rrs.append(_find_wrist_series(orthanc, i))
    return r, rrs

