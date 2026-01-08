import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def header(title):
    print(f"\n{'=' * 60}\n{title.center(60)}\n{'=' * 60}")

def load_data(file_path, target_col='popularity'):
    df = pd.read_csv(file_path)

    X = df.drop(columns=[target_col]).astype('float32')
    y = df[target_col].astype('float32') / 100.0

    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_model(X_train):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(X_train))

    model = tf.keras.Sequential([
        normalizer,

        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),  # Стабилизирует веса
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(32, activation='relu'),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

def train_evaluate(X_train, X_test, y_train, y_test):
    model = create_model(X_train)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    header("TRAINING")

    model.fit(
        np.array(X_train), np.array(y_train),
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    header("TEST RESULTS")
    raw_predictions = model.predict(np.array(X_test))

    y_test_original = y_test * 100.0
    predictions_original = raw_predictions * 100.0

    mse = mean_squared_error(y_test_original, predictions_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, predictions_original)
    r2 = r2_score(y_test_original, predictions_original)

    print(f"- RMSE (Root Mean Sq. Error): {rmse:>10.4f}")
    print(f"- MAE  (Mean Absolute Error): {mae:>10.4f}")
    print(f"- R2   (R-Squared Score):     {r2:>10.4f}")

    return model

def main():
    X_train, X_test, y_train, y_test = load_data('models/train_data.csv')
    model = train_evaluate(X_train, X_test, y_train, y_test)
    model.save('models/tf_model.keras')

if __name__ == "__main__":
    main()