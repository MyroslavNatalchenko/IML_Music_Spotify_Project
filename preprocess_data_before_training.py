import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

def process_and_save():
    df = pd.read_csv('dataset.csv', index_col=0)

    columns_to_drop = ['track_id', 'track_name', 'album_name']
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns]).dropna()

    df['explicit'] = df['explicit'].astype(int)

    encoders = {}
    categorical_cols = ['artists', 'track_genre']

    metadata_lists = {}

    for col in categorical_cols:
        df[col] = df[col].astype(str)

        metadata_lists[col] = sorted(df[col].unique().tolist())

        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    feature_names = df.drop(columns=['popularity']).columns.tolist()

    preprocessor_data = {
        "encoders": encoders,
        "metadata_lists": metadata_lists,
        "feature_names": feature_names
    }
    print(f"encoders: {encoders}\nmetadata_lists: {metadata_lists}\nfeature_names: {feature_names}")
    joblib.dump(preprocessor_data, 'models/metadata.joblib')

    df.to_csv('models/train_data.csv', index=False)

if __name__ == "__main__":
    process_and_save()