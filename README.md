# Stock Price Forecasting Using Deep Learning

Comparing ANN, LSTM, GRU, and Transformer architectures for next-day stock price forecasting, using 10 years of daily data across 5 stocks from different sectors (AAPL, ABT, JPM, XOM, WMT).

🔗 **Live Dashboard:** [Streamlit App Link Here — add after deployment]

## Project Overview

This project builds and evaluates four deep learning architectures on time-series stock price prediction, including full EDA, feature engineering, hyperparameter tuning, and bonus extensions (10-day forecasting, walk-forward validation, XGBoost/Random Forest baselines, Streamlit dashboard).

## Repository Structure

```
├── app.py                          # Streamlit dashboard
├── requirements.txt                # Python dependencies
├── notebooks/
│   └── Deep_Learning_Stock_Forecasting.ipynb   # Full analysis notebook
├── results/                        # All CSV results (metrics, tuning, forecasts)
├── plots/                          # EDA and evaluation visualizations
├── saved_models/                   # Trained GRU models (.keras)
└── report/                         # Technical report and presentation
```

## Key Results

| Stock | Best Model | R² Score |
|-------|-----------|----------|
| AAPL  | GRU       | 0.877    |
| ABT   | LSTM      | 0.972    |
| JPM   | GRU       | 0.844    |
| XOM   | GRU       | 0.972    |
| WMT   | ANN       | 0.473    |

GRU was the most consistently strong performer across stocks, while the Transformer architecture underperformed on this dataset size, and WMT proved the hardest stock to model for all architectures.

## Models Implemented

- **ANN** — Feed-forward baseline
- **LSTM** — With dropout and early stopping
- **GRU** — Best overall performer
- **Transformer** — Multi-head attention based

## Bonus Features

- 10-day future price forecasting
- Walk-forward validation
- XGBoost / Random Forest baseline comparison
- Interactive Streamlit dashboard

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

Python, TensorFlow/Keras, scikit-learn, Streamlit, Plotly, yFinance, pandas, NumPy

## Author

[Your Name] — Deep Learning Project Assignment, 2026
