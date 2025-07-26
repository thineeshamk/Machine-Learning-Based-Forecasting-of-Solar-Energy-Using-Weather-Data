# Machine Learning-Based Forecasting of Solar Energy Using Weather Data

This is a machine learning project developed as part of a group research project for the BSc (Hons) in Data Science and Business Analytics at General Sir John Kotelawala Defence University.

The project focuses on building an accurate and efficient solar power output forecasting system for the Vydexa Solar Power Plant in Vavuniya, Sri Lanka, using real-world operational and weather data. The system aims to provide both short-term (15-minute ahead) and long-term (60-minute ahead) forecasts to support better energy management and decision-making.

---

## Project Overview

The main goal of this project is to forecast **short-term (15 minutes)** and **long-term (1 hour)** solar power output using machine learning techniques.

By integrating solar plant operational data with local weather data, the model helps:
- Improve energy planning
- Enhance power grid stability
- Reduce operational uncertainty

---

---

## Problem Statement and Aim

Problem Statement- How can the accurate and efficient prediction of short-term and long-term solar power output generation be achieved for a solar power plant, incorporating comprehensive weather data?

Aim- The aim of this project is to implement, evaluate and deploy an accurate and efficient predictive model that integrates comprehensive weather data with operational data of the power plant to forecast the short term (15 minute) and long term (1 hour) power output of the solar power plant.

---

## Objectives

- Acquire and clean solar and weather datasets
- Perform exploratory data analysis (EDA)
- Train and evaluate ML models for short-term (15-minute) and long-term (60-minute) solar power forecasting, including:
  - Lasso Regression
  - Random Forest
  - XGBoost
  - LSTM (Neural Network)
- Deploy the best model using a simple interface (e.g., Streamlit)

---

## Machine Learning Models Used

| Model              | Type           | Purpose                             |
|-------------------|----------------|-------------------------------------|
| Lasso Regression  | Linear         | Baseline model, simple & interpretable |
| Random Forest      | Ensemble Trees | Good for handling non-linear data   |
| XGBoost            | Boosted Trees  | High performance with tuning        |
| LSTM               | Deep Learning  | Excellent for time-series forecasting|

---

## Technologies Used

- **Python**
- **Pandas, NumPy, Scikit-learn**
- **TensorFlow, Keras**
- **XGBoost**
- **Matplotlib, Seaborn**
- **Google Colab & VS Code**
- **Power BI & R Studio** (for visualization and statistics)
- **Streamlit** (for model deployment)

---

## Folder Structure

```text
solar-energy-forecasting/
├── data/               ← (Not included – contains private operational and weather data)
├── notebooks/          ← Jupyter Notebooks for EDA 
├── Forecasting Models/ ← .py files for training and other preprocessing tasks
├── app/                ← Deployment files (e.g., Streamlit app)
├── reports/            ← Graphs, figures, and evaluation plots
├── README.md           ← Project explanation (this file)
├── requirements.txt    ← List of Python libraries used
└── Final_Report.pdf    ← Project documentation/report
````

> Note: Due to privacy and data protection policies, the `data/` folder is not included in this repository.

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/solar-energy-forecasting.git
cd solar-energy-forecasting
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Models (Optional)

To train and evaluate the models, open and run the following python files:

1. `xgboost_short_term.py`  
2. `xgboost_long_term (1).py`  
3. `lstm_15min_model.py`  
4. `lstm_1hr_model.py`  
5. `random_forest_short_term_long_term.py`
6. `lasso_regression_short_term_long_term.py`

**Instructions:**
- Run all cells by clicking **"Run All"** or manually running them one by one.
- Each notebook contains code to load data, some data pre processing parts, train models, testing models and display results.

### 4. Launch the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## Results Summary

All models were evaluated on:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* R² Score (Coefficient of Determination)

Short Term Forecasting(15 minutes ahead)

| Matric              | Lasso Regression   | Random Forest       | XGBoost             | LSTM                |
|-------------------|-------------------|-------------------|-------------------|-------------------|
| MSE                 | 6.723              | 3.031               | 2.650               | 1.169               |
| RMSE                | 2.592              | 1.741               | 1.627               | 1.292               |
| MAE                 | 2.088              | 0.851               | 0.818               | 0.578               |
| R-Square            | 0.398              | 0.657               | 0.700               | 0.811               |

Long Term Forecasting(60 minutes ahead)

| Matric              | Lasso Regression   | Random Forest       | XGBoost             | LSTM                |
|-------------------|-------------------|-------------------|-------------------|-------------------|
| MSE                 | 8.772              | 3.034               | 3.882               | 1.746               |
| RMSE                | 2.961              | 1.741               | 1.970               | 1.321               |
| MAE                 | 2.574              | 0.851               | 1.112               | 0.643               |
| R-Square            | 0.214              | 0.657               | 0.561               | 0.802               |

**Best performing model:** LSTM
**Best for simplicity & interpretability:** Lasso Regression
**Best balance of accuracy & speed:** XGBoost

---

## License

This project is licensed under the MIT License.

---

## Authors

* HMRV Herath 
* RN Silva 
* TM Kahavidhana 
* KT Panditha 

> Supervised by Mrs. SMM Lakmali
> Department of Computational Mathematics
> General Sir John Kotelawala Defence University




