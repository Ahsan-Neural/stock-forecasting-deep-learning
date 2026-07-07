# Stock Forecasting with Deep Learning

A deep learning project for forecasting stock prices using historical market time-series data, notebook-based experiments, and a simple interactive Streamlit app.

This repository focuses on five major U.S. stocks over approximately 10 years of daily data:

- AAPL
- ABT
- JPM
- XOM
- WMT

---

## Repository Information

- **Repository:** `Ahsan-Neural/stock-forecasting-deep-learning`
- **Primary language composition:**
  - Jupyter Notebook: **99.8%**
  - Python: **0.2%**
- **License:** **MIT** (see [`LICENSE`](LICENSE))

---

## Project Overview

This project includes:

- Stock data collection using `yfinance`
- Time-series preprocessing and preparation
- Deep learning model training for forecasting
- Validation and performance analysis
- Visual outputs and saved artifacts
- Streamlit app for quick interaction

---

## Repository Structure

Current top-level structure:

```text
stock-forecasting-deep-learning/
├── .gitignore
├── LICENSE
├── README.md
├── app.py
├── requirements.txt
├── Docs/
├── notebooks/
├── plots/
├── results/
└── saved_models/
```

### Directory/File Purpose

- **`notebooks/`** – Main notebook workflows for data loading, preprocessing, model training, and evaluation.
- **`saved_models/`** – Trained model files/checkpoints.
- **`results/`** – Output files such as prediction/evaluation artifacts.
- **`plots/`** – Generated charts and visual analysis figures.
- **`Docs/`** – Project-related documentation assets.
- **`app.py`** – Streamlit application entry point.
- **`requirements.txt`** – Python dependencies.

---

## Live Dashboard

Streamlit app:

https://stock-forecasting-deep-learning-u3kc7jwbebtwdcxogp6azf.streamlit.app/

---

## Getting Started

### 1) Clone the repository

```bash
git clone https://github.com/Ahsan-Neural/stock-forecasting-deep-learning.git
cd stock-forecasting-deep-learning
```

### 2) Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 3) Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

### 4) Run notebooks

Use Jupyter to open notebooks in `notebooks/` and run cells in order.

Typical workflow:

1. Fetch stock data
2. Preprocess and prepare features
3. Train forecasting model(s)
4. Evaluate and export outputs

### 5) Run Streamlit app (optional)

```bash
streamlit run app.py
```

---

## Data Source and Usage Notes

- Market data is fetched programmatically using **`yfinance`** (Yahoo Finance source).
- Raw Yahoo Finance data files are not distributed in this repository.
- Repository mainly contains processed/derived outputs, models, and visualizations.

### Reproducibility

Because data is fetched live, outputs may vary slightly across runs due to:

- Data provider updates/corrections
- Different fetch timestamps
- Package/version differences

For best reproducibility, pin dependency versions and log run date/time.

---

## License

This project is licensed under the **MIT License**.

See the full license text in [`LICENSE`](LICENSE).

---

## Disclaimer

This project is for **educational and research** purposes only.
It does **not** provide financial advice or investment recommendations.
