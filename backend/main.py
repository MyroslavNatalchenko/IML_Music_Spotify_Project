from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import glob

app = FastAPI(
    title="ML Model Serving API",
    description="API for accessing multiple ML models via separate endpoints",
    version="2.0"
)

METADATA_PATH = "../models/metadata.joblib"

if not os.path.exists(METADATA_PATH):
    raise RuntimeError("No metadata file found.")

meta = joblib.load(METADATA_PATH)
encoders = meta["encoders"]
feature_names = meta["feature_names"]
dropdown_lists = meta["metadata_lists"]

models = {}
model_files = glob.glob("../models/*_model.joblib")

for path in model_files:
    filename = os.path.basename(path).replace(".joblib", "")
    models[filename] = joblib.load(path)
    print(f"\t --- Loaded models: {filename}")

class TrackFeatures(BaseModel):
    artists: str
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

    try:
        artist_code = encoders["artists"].transform([data["artists"]])[0]
    except ValueError:
        artist_code = 0

    try:
        genre_code = encoders["track_genre"].transform([data["track_genre"]])[0]
    except ValueError:
        genre_code = 0

    data["artists"] = artist_code
    data["track_genre"] = genre_code
    data["explicit"] = int(data["explicit"])

    try:
        df_input = pd.DataFrame([data])
        df_input = df_input[feature_names]
        return df_input
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}")

@app.get("/meta/info")
def get_metadata():
    return {
        "artists": dropdown_lists["artists"],
        "genres": dropdown_lists["track_genre"]
    }

@app.get("/models")
def list_models():
    return {"models": list(models.keys())}

@app.post("/models/all/predict")
def predict_all_models(features: TrackFeatures):
    df_input = prepare_input(features)
    results = {}

    for name, model in models.items():
        try:
            pred = model.predict(df_input)[0]
            results[name] = max(0, min(100, float(pred)))
        except Exception:
            results[name] = None

    return results

@app.post("/models/{model_name}/predict")
def predict_specific_model(
        features: TrackFeatures,
        model_name: str = Path(..., description="The name of the model to use")):
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    df_input = prepare_input(features)
    model = models[model_name]

    try:
        pred = model.predict(df_input)[0]
        score = max(0, min(100, float(pred)))
        return {
            "model": model_name,
            "popularity_score": score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))