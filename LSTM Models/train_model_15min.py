import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# =================CONFIGURATION=================
# UPDATED PATH
TRAIN_FILE_PATH = r"C:\Users\Dell\Solar Power Project\Train Data set.xlsx"

# Folder where model and scalers will be saved
OUTPUT_FOLDER = 'model_artifacts'

# Model Hyperparameters
SEQUENCE_LENGTH = 15   # Past time steps
PREDICTION_LENGTH = 15 # Future steps to predict
DROPOUT_RATE = 0.2
LSTM_UNITS = 50
EPOCHS = 50
BATCH_SIZE = 32
# ===============================================

def create_output_folder():
    """Creates the folder to save models if it doesn't exist."""
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created folder: {OUTPUT_FOLDER}")
    else:
        print(f"Saving to existing folder: {OUTPUT_FOLDER}")

def create_sequences(input_data, target_data, features_to_scale, output_column, seq_len, pred_len):
    """Prepares data sequences for LSTM."""
    X, y = [], []
    # Convert dataframe parts to numpy arrays to speed up loop
    data_feat = input_data[features_to_scale].values
    data_target = target_data[output_column].values
    
    for i in range(len(input_data) - seq_len - pred_len + 1):
        X.append(data_feat[i:i+seq_len])
        y.append(data_target[i:i+pred_len])
        
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, prediction_length):
    """Defines the LSTM architecture."""
    model = Sequential([
        Bidirectional(LSTM(LSTM_UNITS, activation='tanh', return_sequences=True), input_shape=input_shape),
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS, activation='tanh'),
        Dense(prediction_length)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    print("--- Starting Training Pipeline ---")
    create_output_folder()

    # 1. Load Data
    print(f"Loading data from {TRAIN_FILE_PATH}...")
    if not os.path.exists(TRAIN_FILE_PATH):
        print("Error: File not found. Please check that the file exists and the path is correct.")
        return

    # Load excel - using 'openpyxl' engine is safer for .xlsx
    try:
        data = pd.read_excel(TRAIN_FILE_PATH, parse_dates=['Time'], engine='openpyxl')
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # 2. Scaling
    print("Scaling data...")
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    features_to_scale = ['Irradiation', 'Temp', 'Wind', 'Humidity', 'Barometer']
    output_column = 'TOAL ACTIVE POWER [MW]'

    # Check if columns exist
    if output_column not in data.columns:
        print(f"Error: Column '{output_column}' not found in Excel file.")
        print(f"Available columns: {list(data.columns)}")
        return

    data[features_to_scale] = feature_scaler.fit_transform(data[features_to_scale])
    data[[output_column]] = target_scaler.fit_transform(data[[output_column]])

    # 3. Save Scalers
    feature_scaler_path = os.path.join(OUTPUT_FOLDER, 'feature_scaler_final.pkl')
    target_scaler_path = os.path.join(OUTPUT_FOLDER, 'target_scaler_final.pkl')
    
    joblib.dump(feature_scaler, feature_scaler_path)
    joblib.dump(target_scaler, target_scaler_path)
    print(f"Scalers saved to {OUTPUT_FOLDER}")

    # 4. Prepare Sequences
    print("Creating sequences for LSTM...")
    X, y = create_sequences(data, data, features_to_scale, output_column, SEQUENCE_LENGTH, PREDICTION_LENGTH)
    
    if len(X) == 0:
        print("Error: Not enough data to create sequences. Check dataset size.")
        return

    print(f"Input shape: {X.shape}, Output shape: {y.shape}")

    # 5. Build and Train Model
    print("Building model...")
    model = build_lstm_model((SEQUENCE_LENGTH, len(features_to_scale)), PREDICTION_LENGTH)
    
    early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("Training model...")
    history = model.fit(
        X, y, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_split=0.2, 
        callbacks=[early_stopping_monitor], 
        verbose=1
    )

    # 6. Save Model
    model_path = os.path.join(OUTPUT_FOLDER, 'lstm_model_final.h5')
    model.save(model_path)
    print(f"Model saved successfully to: {model_path}")

    # 7. Plot History
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training Progress')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    print("--- Pipeline Complete ---")

if __name__ == "__main__":
    main()