from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import glob
import tensorflow as tf
import xgboost as xgb
from tabnet_keras import TabNetRegressor, feature_transformer

app = FastAPI(title="Spotify Predictor API", version="1.0")

METADATA_PATH = "../models/metadata.joblib"
SCALER_PATH = "../models/tabnet_scaler.joblib"

meta = joblib.load(METADATA_PATH)
feature_names = meta["feature_names"]
genre_list = meta["genre_list"]

tabnet_scaler = joblib.load(SCALER_PATH)
TABNET_PARAMS = {
    "decision_dim": 64,
    "attention_dim": 64,
    "n_steps": 8,
    "n_shared_glus": 2,
    "n_dependent_glus": 2,
    "relaxation_factor": 1.5,
    "epsilon": 1e-15,
    "momentum": 0.98,
    "mask_type": "softmax",
    "lambda_sparse": 1e-4,
}
tabnet_custom_objects = {
    "TabNetRegressor": TabNetRegressor,
    "FeatureTransformer": feature_transformer.FeatureTransformer
}

models = {}
model_files = glob.glob("../models/*_model.joblib") + glob.glob("../models/*_model.keras")

for path in model_files:
    filename = os.path.basename(path)
    model_name = filename.split('.')[0]

    if filename.endswith(".joblib"):
        models[model_name] = {"model": joblib.load(path), "type": "sklearn"}
        print(f"\t -- Loaded Sklearn/XGB: {model_name}")
    elif filename.endswith(".keras"):
        if "tabnet" in filename.lower():
            try:
                print(f"\t .. Attempting to reconstruct TabNet: {model_name}")
                model = TabNetRegressor(n_regressors=1, **TABNET_PARAMS)
                dummy_input = tf.zeros((1, len(feature_names)))
                _ = model(dummy_input)
                model.load_weights(path)
                models[model_name] = {"model": model, "type": "tabnet"}
                print(f"\t -- Loaded TabNet (Weights Only): {model_name}")
            except Exception as e:
                print(f"\t !! Failed to load TabNet {model_name}: {e}")
        else:
            try:
                model = tf.keras.models.load_model(path)
                models[model_name] = {"model": model, "type": "tensorflow"}
                print(f"\t -- Loaded TF: {model_name}")
            except Exception as e:
                print(f"\t !! Failed to load TF {model_name}: {e}")

def get_prediction(model_entry, df_input):
    model = model_entry["model"]
    m_type = model_entry["type"]

    if m_type == "sklearn":
        return float(model.predict(df_input)[0])

    elif m_type == "tensorflow":
        pred = model.predict(df_input.values.astype(np.float32), verbose=0)[0][0]
        return float(pred * 100.0)

    elif m_type == "tabnet":
        X_input = tabnet_scaler.transform(df_input)
        pred = model.predict(X_input, verbose=0)[0][0]
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

    df_input = pd.DataFrame(0.0, index=[0], columns=feature_names)

    numeric_cols = [
        'duration_ms', 'explicit', 'danceability', 'energy', 'key',
        'loudness', 'mode', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'
    ]

    for col in numeric_cols:
        if col in df_input.columns:
            df_input.at[0, col] = float(data[col])

    df_input.at[0, 'explicit'] = float(data['explicit'])

    target_col = f"genre_{data['track_genre']}"
    if target_col in df_input.columns:
        df_input.at[0, target_col] = 1.0

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
        except Exception as e:
            print(f"Error predicting {name}: {e}")
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