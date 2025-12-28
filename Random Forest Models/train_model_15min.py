# =========================================
# RANDOM FOREST – 15 MINUTE FORECAST
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump

sns.set(style="whitegrid")

# -----------------------------
# Load Data
# -----------------------------
train_data = pd.read_csv('Train_Data_Set.csv')
test_data = pd.read_csv('Test_Data_Set.csv')

# -----------------------------
# Datetime Processing
# -----------------------------
def combine_date_time(df):
    if 'Date' not in df.columns:
        df[['Date', 'Time']] = df['Time'].str.split(' ', n=1, expand=True)
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%m/%d/%Y %H:%M')
    return df.drop(['Date', 'Time'], axis=1)

train_data = combine_date_time(train_data)
test_data = combine_date_time(test_data)

train_data.sort_values('Datetime', inplace=True)
test_data.sort_values('Datetime', inplace=True)

# -----------------------------
# Target (15 min ahead)
# -----------------------------
train_data['Target'] = train_data['TOTAL_ACTIVE_POWER'].shift(-15)
test_data['Target'] = test_data['TOTAL_ACTIVE_POWER'].shift(-15)

train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# -----------------------------
# Time Features
# -----------------------------
def extract_time_features(df):
    df['Hour'] = df['Datetime'].dt.hour
    df['Minute'] = df['Datetime'].dt.minute
    df['Day'] = df['Datetime'].dt.day
    df['Month'] = df['Datetime'].dt.month
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    return df

train_data = extract_time_features(train_data)
test_data = extract_time_features(test_data)

# -----------------------------
# Features
# -----------------------------
feature_cols = [
    'Irradiation', 'Temp', 'Wind', 'Humidity', 'Barometer',
    'Hour', 'Minute', 'Day', 'Month', 'DayOfWeek'
]

X_train = train_data[feature_cols]
y_train = train_data['Target']
X_test = test_data[feature_cols]
y_test = test_data['Target']

# -----------------------------
# Model
# -----------------------------
rf = RandomForestRegressor(
    n_estimators=700,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("15-Minute Forecast Results")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# -----------------------------
# Plots
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:3000], label="Actual")
plt.plot(y_pred[:3000], label="Predicted")
plt.title("15-Minute Forecast: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Save Model
# -----------------------------
dump(rf, "RandomForest_15min.joblib")
print("Model saved as RandomForest_15min.joblib")
