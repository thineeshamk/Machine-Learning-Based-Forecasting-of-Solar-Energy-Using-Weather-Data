import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

# ======================================================
# CONFIGURATION
# ======================================================

TRAIN_FILE_PATH = r"C:\Users\Dell\Solar Power Project\Train Data set.xlsx"
OUTPUT_FOLDER = "xgb_model_artifacts"

INPUT_COLUMNS = ['Irradiation', 'Temp', 'Wind', 'Humidity', 'Barometer']
OUTPUT_COLUMN = 'TOAL ACTIVE POWER [MW]'

LAG_MINUTES = 15          # past 15 minutes
PREDICTION_STEP = 15      # predict 15 minutes ahead

# XGBoost hyperparameters
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 50,
    "learning_rate": 0.1,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

# ======================================================
# UTILITY FUNCTIONS
# ======================================================

def create_output_folder():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created folder: {OUTPUT_FOLDER}")
    else:
        print(f"Using existing folder: {OUTPUT_FOLDER}")


def create_lag_features(data, input_cols, output_col, lag_minutes, prediction_step):
    """
    Converts time-series data into supervised learning format.
    """
    df = data.copy()

    # Create lag features
    for col in input_cols:
        for lag in range(1, lag_minutes + 1):
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # Target shifted to future
    df["target"] = df[output_col].shift(-prediction_step)

    df.dropna(inplace=True)
    return df


def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\nModel Performance")
    print("-----------------")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")

    return mse, rmse, mae, r2


# ======================================================
# MAIN TRAINING PIPELINE
# ======================================================

def main():

    print("Starting XGBoost training pipeline")
    create_output_folder()

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    print("Loading training data...")
    data = pd.read_excel(
        TRAIN_FILE_PATH,
        parse_dates=["Time"],
        index_col="Time"
    )

    data.sort_index(inplace=True)

    # --------------------------------------------------
    # Create supervised dataset
    # --------------------------------------------------
    print("Creating lag features...")
    supervised_df = create_lag_features(
        data,
        INPUT_COLUMNS,
        OUTPUT_COLUMN,
        LAG_MINUTES,
        PREDICTION_STEP
    )

    feature_columns = [
        f"{col}_lag_{i}"
        for col in INPUT_COLUMNS
        for i in range(1, LAG_MINUTES + 1)
    ]

    X = supervised_df[feature_columns].values
    y = supervised_df["target"].values

    print(f"Training samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")

    # --------------------------------------------------
    # Train model
    # --------------------------------------------------
    print("Training XGBoost model...")
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X, y)

    # --------------------------------------------------
    # Evaluate on training data (baseline check)
    # --------------------------------------------------
    y_pred = model.predict(X)
    evaluate_model(y, y_pred)

    # --------------------------------------------------
    # Save model and configuration
    # --------------------------------------------------
    model_path = os.path.join(OUTPUT_FOLDER, "xgb_model_15min.pkl")
    config_path = os.path.join(OUTPUT_FOLDER, "feature_config_15min.pkl")

    joblib.dump(model, model_path)

    feature_config = {
        "input_columns": INPUT_COLUMNS,
        "lag_minutes": LAG_MINUTES,
        "prediction_step": PREDICTION_STEP,
        "feature_columns": feature_columns
    }

    joblib.dump(feature_config, config_path)

    print("\nSaved files:")
    print(f"- Model  : {model_path}")
    print(f"- Config : {config_path}")

    # --------------------------------------------------
    # Plot predictions vs actual
    # --------------------------------------------------
    plt.figure(figsize=(14, 6))
    plt.plot(supervised_df.index, y, label="Actual")
    plt.plot(supervised_df.index, y_pred, label="Predicted", alpha=0.7)
    plt.title("XGBoost – 15 Minute Ahead Prediction")
    plt.xlabel("Time")
    plt.ylabel(OUTPUT_COLUMN)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nTraining pipeline completed successfully")


if __name__ == "__main__":
    main()
