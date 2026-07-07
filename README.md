# Stock Forecasting with Deep Learning

This repository contains a deep learning workflow for stock market forecasting using historical time-series data and notebook-based experimentation.  
The project focuses on modeling and evaluating stock behavior across five major U.S. tickers over a 10-year period:

- AAPL
- ABT
- JPM
- XOM
- WMT

The repository is primarily built in Jupyter Notebook, with supporting Python code.

## Live Dashboard

A Streamlit dashboard for interactive exploration is available here:

https://stock-forecasting-deep-learning-u3kc7jwbebtwdcxogp6azf.streamlit.app/

## Project Overview

The project includes:

- Data collection pipeline (via Yahoo Finance using `yfinance`)
- Time-series preprocessing and feature preparation
- Deep learning model training for stock forecasting
- Model evaluation and validation
- Result visualizations and exported outputs

## Repository Contents

Typical contents include:

- Notebook(s) for end-to-end workflow (data fetching, preprocessing, modeling, evaluation)
- `VALIDATED_final_results.csv` containing validated model performance outputs
- Generated chart images and derived artifacts
- Supporting code cells/scripts for reproducibility

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Ahsan-Neural/stock-forecasting-deep-learning.git
cd stock-forecasting-deep-learning
```

### 2. Create and activate environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -U pip
pip install yfinance pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
```

### 4. Run notebook(s)

Open the notebook(s) and execute cells in order to:

- download fresh market data
- preprocess and engineer features
- train/evaluate forecasting models
- regenerate final outputs and visualizations

## Dataset Summary

This project uses 10 years of daily OHLCV (Open, High, Low, Close, Volume) data for:

- AAPL
- ABT
- JPM
- XOM
- WMT

Data is fetched programmatically during runtime rather than stored as raw files in this repository.

## Data Source & Licensing

- All stock market data (10 years of daily OHLCV data for AAPL, ABT, JPM, XOM, WMT) was sourced live via the `yfinance` Python library, which pulls from Yahoo Finance.
- Raw data files are **NOT** included in this repository, in compliance with Yahoo Finance's Terms of Service, which restrict redistribution of their market data.
- To reproduce the dataset, users should install yfinance (`pip install yfinance`) and run the data-fetching code included in the notebook, which downloads fresh data directly from Yahoo Finance for the same five tickers and date range.
- This project uses the data for academic/educational purposes only, not for commercial redistribution.
- Only derived outputs (engineered features, model results, validated performance metrics in `VALIDATED_final_results.csv`, and generated chart images) are included in this repo, not the raw price data itself.

## Reproducibility Notes

Because data is pulled live from Yahoo Finance at runtime, minor differences may occur between runs due to:

- market data corrections by provider
- download timestamp differences
- preprocessing/version changes in dependencies

For best reproducibility, pin package versions and document run date/time in experiment logs.

## Disclaimer

This repository is intended for educational and research purposes only.  
It does not provide financial advice, investment recommendations, or guarantees of market performance.

