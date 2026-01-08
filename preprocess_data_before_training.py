import pandas as pd
import joblib
import os

def process_and_save():
    df = pd.read_csv('dataset.csv', index_col=0)

    columns_to_drop = ['track_id', 'track_name', 'album_name', 'artists']
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns]).dropna()

    unique_genres = sorted(df['track_genre'].unique().tolist())
    df = pd.get_dummies(df, columns=['track_genre'], prefix='genre')

    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    feature_names = df.drop(columns=['popularity']).columns.tolist()

    preprocessor_data = {
        "genre_list": unique_genres,
        "feature_names": feature_names
    }

    joblib.dump(preprocessor_data, 'models/metadata.joblib')
    df.to_csv('models/train_data.csv', index=False)

    print(f"\t- Features count: {len(feature_names)}")
    print(f"\t- Genres found: {len(unique_genres)}")

if __name__ == "__main__":
    process_and_save()