CREATE TABLE IF NOT EXISTS wrist_fracture(
    PatientID TEXT,
    PatientName TEXT,
    PatientSex TEXT,
    AccessionNumber TEXT,
    StudyDate TEXT,
    StudyDescription TEXT,
    SeriesNumber TEXT,
    SeriesDescription TEXT,
    SeriesInstanceUID TEXT,
    ml_fracture_hardplaster TEXT,
    ml_fracture_hardplaster_version TEXT,
    ml_fracture_view TEXT,
    ml_fracture_view_version TEXT,
    ml_fracture_probability REAL,
    ml_fracture_probability_version TEXT
)