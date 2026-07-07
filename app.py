
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Stock Forecasting Dashboard", layout="wide", page_icon="📈")

DARK_BG = "#0e1117"
CARD_BG = "#161b22"
ACCENT = "#58a6ff"
TEXT = "#e6edf3"
MUTED = "#8b949e"

st.markdown(f"""
<style>
.stApp {{ background-color: {DARK_BG}; color: {TEXT}; }}
[data-testid="stMetric"] {{
    background-color: {CARD_BG};
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px;
}}
h1, h2, h3 {{ color: {TEXT}; }}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    results = pd.read_csv("results/final_all_models_all_stocks.csv")
    forecasts = pd.read_csv("results/future_10day_forecasts.csv", index_col=0)
    walk_fwd = pd.read_csv("results/walk_forward_validation_AAPL.csv")
    tuning = pd.read_csv("results/tuning_results_AAPL.csv")
    return results, forecasts, walk_fwd, tuning

results_df, forecast_df, walk_df, tuning_df = load_data()

st.title("📈 Stock Price Forecasting Dashboard")
st.caption("Deep Learning Models: ANN · LSTM · GRU · Transformer — 5 stocks, 10 years of daily data")

tickers = sorted(results_df["Ticker"].unique())
models = sorted(results_df["Model"].unique())

with st.sidebar:
    st.header("Controls")
    selected_ticker = st.selectbox("Select Stock", tickers, index=0)
    selected_model = st.selectbox("Select Model", models, index=models.index("GRU") if "GRU" in models else 0)
    st.markdown("---")
    st.markdown("**Project Summary**")
    st.markdown("- 5 stocks x 10 years daily data\n- 4 architectures compared\n- GRU: best & most consistent\n- Bonus: 10-day forecast, walk-forward validation")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Performance", "🔮 10-Day Forecast", "🧪 Walk-Forward Validation", "⚙️ Hyperparameter Tuning"])

with tab1:
    st.subheader(f"{selected_model} Performance — {selected_ticker}")
    row = results_df[(results_df["Ticker"] == selected_ticker) & (results_df["Model"] == selected_model)].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE", f"{row['MAE']:.4f}")
    c2.metric("RMSE", f"{row['RMSE']:.4f}")
    c3.metric("MAPE", f"{row['MAPE']:.2f}%")
    c4.metric("R2 Score", f"{row['R2']:.3f}")
    c5.metric("MSE", f"{row['MSE']:.5f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Model Comparison — {selected_ticker}**")
        sub = results_df[results_df["Ticker"] == selected_ticker]
        fig = px.bar(sub, x="Model", y="R2", color="Model",
                     color_discrete_map={"GRU": ACCENT, "LSTM": "#f78166", "Transformer": "#3fb950", "ANN": "#d29922"})
        fig.update_layout(template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                           showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**R2 Heatmap — All Stocks x Models**")
        pivot = results_df.pivot(index="Ticker", columns="Model", values="R2")
        fig2 = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index,
                                          colorscale="RdYlGn", zmid=0, text=np.round(pivot.values, 2),
                                          texttemplate="%{text}", colorbar=dict(title="R2")))
        fig2.update_layout(template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG, height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Full Results Table**")
    st.dataframe(results_df.style.background_gradient(subset=["R2"], cmap="RdYlGn"), use_container_width=True)

with tab2:
    st.subheader(f"10-Day Future Price Forecast — {selected_ticker}")
    if selected_ticker in forecast_df.columns:
        fc_series = forecast_df[selected_ticker]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=[f"Day {i+1}" for i in range(len(fc_series))], y=fc_series.values,
                                   mode="lines+markers", line=dict(color=ACCENT, width=3), marker=dict(size=9)))
        fig3.update_layout(template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                            xaxis_title="Forecast Day", yaxis_title="Predicted Price (scaled)", height=450)
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(fc_series.reset_index().rename(columns={"index": "Day", selected_ticker: "Predicted Price"}),
                     use_container_width=True)
    else:
        st.info("No forecast data available for this ticker.")

with tab3:
    st.subheader("Walk-Forward Validation — AAPL (GRU)")
    st.caption("Model retrained across 5 sequential folds to simulate live forecasting conditions")
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=walk_df["Fold"], y=walk_df["R2"], marker_color=ACCENT, name="R2"))
    fig4.update_layout(template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                        height=400, xaxis_title="Fold", yaxis_title="R2 Score")
    st.plotly_chart(fig4, use_container_width=True)

    fig4b = go.Figure()
    fig4b.add_trace(go.Scatter(x=walk_df["Fold"], y=walk_df["MAPE"], mode="lines+markers",
                                line=dict(color="#f78166", width=3), name="MAPE"))
    fig4b.update_layout(template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                         height=350, xaxis_title="Fold", yaxis_title="MAPE (%)")
    st.plotly_chart(fig4b, use_container_width=True)

    st.dataframe(walk_df, use_container_width=True)

with tab4:
    st.subheader("GRU Hyperparameter Tuning — AAPL")
    fig5 = px.bar(tuning_df, x="Value", y="R2", color="Hyperparameter")
    fig5.update_layout(template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG, height=450)
    st.plotly_chart(fig5, use_container_width=True)
    st.dataframe(tuning_df, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit x TensorFlow/Keras x Data via yFinance (10 years daily, 5 tickers: AAPL, ABT, JPM, XOM, WMT)")
