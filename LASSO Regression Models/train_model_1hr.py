# =========================================
# LASSO REGRESSION – 60 MINUTE (1 HOUR) FORECAST
# =========================================

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import glob, os
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Load Data
# -----------------------------
data_path = '/content/drive/MyDrive/Lasso Regression/'
files = glob.glob(os.path.join(data_path, '*.csv'))

df_list = [pd.read_csv(f, parse_dates=['Time']) for f in files]
data = pd.concat(df_list, ignore_index=True)

# -----------------------------
# Data Cleaning
# -----------------------------
numerical_cols = ['TOAL ACTIVE POWER [MW]', 'Irradiation', 'Temp', 'Wind', 'Humidity', 'Barometer']
for col in numerical_cols:
    data[col].fillna(data[col].median(), inplace=True)

weather_cols = [c for c in data.columns if 'Weather_' in c]
data[weather_cols] = data[weather_cols].fillna(0)

# -----------------------------
# Target Creation (60 min)
# -----------------------------
data['Target_60min'] = data['TOAL ACTIVE POWER [MW]'].shift(-60)

# -----------------------------
# Lag Features
# -----------------------------
for lag in range(1, 6):
    data[f'Lag_{lag}'] = data['TOAL ACTIVE POWER [MW]'].shift(lag)

data.dropna(inplace=True)

# -----------------------------
# Features & Split
# -----------------------------
feature_cols = [c for c in data.columns if c.startswith('Lag_')] + \
               ['Irradiation', 'Temp', 'Wind', 'Humidity', 'Barometer'] + weather_cols

X = data[feature_cols]
y = data['Target_60min']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Lasso + TimeSeries CV
# -----------------------------
lasso = Lasso(max_iter=10000)
param_grid = {'alpha': np.logspace(-5, 1, 100)}
tscv = TimeSeriesSplit(n_splits=5)

grid = GridSearchCV(
    lasso, param_grid, cv=tscv,
    scoring='neg_mean_squared_error', n_jobs=-1
)
grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_

# -----------------------------
# Evaluation
# -----------------------------
y_pred = best_model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("60-Minute Forecast Results")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")
print(f"Best Alpha: {grid.best_params_['alpha']}")

# -----------------------------
# Plots
# -----------------------------
plt.figure(figsize=(12,5))
plt.plot(y_test.values[:3000], label="Actual")
plt.plot(y_pred[:3000], label="Predicted")
plt.title("60-Minute Forecast: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()

# Feature Importance
coef = best_model.coef_
important_idx = np.where(coef != 0)[0]

plt.figure(figsize=(10,6))
plt.barh(
    [feature_cols[i] for i in important_idx],
    coef[important_idx]
)
plt.title("Feature Importance – 60 Minute Forecast")
plt.grid(True)
plt.show()
