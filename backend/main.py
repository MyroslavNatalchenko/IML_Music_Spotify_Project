from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import glob
import tensorflow as tf

app = FastAPI(title="Spotify Predictor API", version="1.0")

METADATA_PATH = "../models/metadata.joblib"

meta = joblib.load(METADATA_PATH)
feature_names = meta["feature_names"]
genre_list = meta["genre_list"]

models = {}
model_files = glob.glob("../models/*_model.joblib") + glob.glob("../models/*_model.keras")

for path in model_files:
    filename = os.path.basename(path)
    model_name = filename.split('.')[0]

    if filename.endswith(".joblib"):
        models[model_name] = {"model": joblib.load(path), "type": "sklearn"}
        print(f"\t -- Loaded Sklearn: {model_name}")
    elif filename.endswith(".keras"):
        models[model_name] = {"model": tf.keras.models.load_model(path), "type": "tensorflow"}
        print(f"\t -- Loaded TF: {model_name}")

def get_prediction(model_entry, df_input):
    model = model_entry["model"]
    m_type = model_entry["type"]

    if m_type == "sklearn":
        return float(model.predict(df_input)[0])
    elif m_type == "tensorflow":
        pred = model.predict(df_input.values.astype(np.float32), verbose=0)[0][0]
        return float(pred * 100.0)

class TrackFeatures(BaseModel):
    track_genre: str
    duration_ms: int
    explicit: bool
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    time_signature: int

def prepare_input(features: TrackFeatures) -> pd.DataFrame:
    data = features.dict()

    df_input = pd.DataFrame(0, index=[0], columns=feature_names)

    numeric_cols = [
        'duration_ms', 'explicit', 'danceability', 'energy', 'key',
        'loudness', 'mode', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'
    ]

    for col in numeric_cols:
        if col in df_input.columns:
            df_input.at[0, col] = data[col]

    df_input.at[0, 'explicit'] = int(data['explicit'])

    target_col = f"genre_{data['track_genre']}"
    if target_col in df_input.columns:
        df_input.at[0, target_col] = 1

    return df_input

@app.get("/meta/info")
def get_metadata():
    return {
        "genres": genre_list
    }

@app.get("/models")
def list_models():
    return {"models": list(models.keys())}

@app.post("/models/all/predict")
def predict_all(features: TrackFeatures):
    df = prepare_input(features)
    res = {}
    for name, entry in models.items():
        try:
            val = get_prediction(entry, df)
            res[name] = max(0, min(100, val))
        except Exception:
            res[name] = None
    return res

@app.post("/models/{model_name}/predict")
def predict_one(features: TrackFeatures, model_name: str = Path(...)):
    if model_name not in models:
        raise HTTPException(404, "Model not found")
    df = prepare_input(features)
    try:
        val = get_prediction(models[model_name], df)
        return {"model": model_name, "popularity_score": max(0, min(100, val))}
    except Exception as e:
        raise HTTPException(500, str(e))