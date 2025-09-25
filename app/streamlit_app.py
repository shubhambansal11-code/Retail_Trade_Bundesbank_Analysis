# streamlit_app.py

# -----------------------------
# Imports & Path Setup
# -----------------------------
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import pickle

from src.data_loader import load_all_sectors
from src.features import build_dataset
from src.visualization import plot_revenue_vs_prediction, plot_probability_and_inventory
from src.config import BASE_REVENUE_LATEST

# -----------------------------
# Sidebar: User Inputs
# -----------------------------
st.sidebar.title("Retail Forecast Inputs")

# Load sector data
sectors_data = load_all_sectors()
sectors = list(sectors_data.columns)

# Restrict to single selection
selected_sector = st.sidebar.selectbox("Select sector", sectors)

# Forecast horizon (3–12 months)
forecast_horizon = st.sidebar.slider("Forecast horizon (months)", min_value=5, max_value=12, value=6)

# Fixed confidence/error margin = 95%
error_pct = 0.05  

st.sidebar.markdown("---")
st.sidebar.subheader("Business Calibration")

inventory_threshold = st.sidebar.slider(
    "Inventory threshold", 0.0, 1.0, 0.4,
    help="Minimum coverage level below which inventory is risky."
)

shortage_alpha = st.sidebar.slider(
    "Shortage amplifier", 0.0, 5.0, 1.0,
    help="Controls aggressiveness of restocking when shortages are predicted."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Note on Inventory Simulation:**\n"
    "Inventory values are simulated with a random walk for demo purposes. "
    "They may vary between runs."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Note on Revenues**")
st.sidebar.write(
    """
    Baseline revenues used for scaling forecasts (in €):  
    - Electronics: **€5M**  
    - Groceries: **€12M**  
    - Textiles: **€2M**  
    - Furniture: **€1.5M**  
    - Pharmacy: **€3M**  
    - Motor Vehicles: **€8M**  

    These serve as calibration anchors so predictions are shown in **monetary terms**.
    """
)

# -----------------------------
# Load dataset + models
# -----------------------------
dataset = build_dataset(sectors_data)
with open("trained_models.pkl", "rb") as f:
    trained_models = pickle.load(f)

# -----------------------------
# Simulate Inventory (random walk simulation)
# -----------------------------
inventory_pct = {}
for s in sectors:
    n = len(dataset.index)
    rw = np.cumsum(np.random.randn(n) * 0.02) + 0.75
    inventory_pct[s] = pd.Series(np.clip(rw, 0.2, 1.0), index=dataset.index)

# -----------------------------
# Tabs for Forecast & Insights
# -----------------------------
forecast_tab, insights_tab = st.tabs(["Forecast", "Business Insights"])

# -----------------------------
# Forecast Tab
# -----------------------------
with forecast_tab:
    st.title("Forecasts")

    sector = selected_sector
    st.header(f"{sector} Forecast")

    # Features + model
    X = dataset[[col for col in dataset.columns if col.startswith(sector) and not col.endswith("_target")] + ["Month"]]
    lr_model = trained_models[sector]
    probabilities = lr_model.predict_proba(X)[:, 1]

    # Compute expected forward change
    index_values = sectors_data[sector].reindex(dataset.index).values
    mean_prob = probabilities.mean()
    exp_fwd_pct = probabilities * mean_prob + (1 - probabilities) * mean_prob
    pred_idx = index_values * (1 + exp_fwd_pct / 100)

    # Scale into revenue
    last_index = sectors_data[sector].iloc[-1]
    revenue_now = (index_values / last_index) * BASE_REVENUE_LATEST[sector]
    revenue_pred = (pred_idx / last_index) * BASE_REVENUE_LATEST[sector]

    # Shift predictions
    future_dates = dataset.index + pd.DateOffset(months=forecast_horizon)
    revenue_pred_shifted = pd.Series(revenue_pred, index=future_dates)

    forecast_start = dataset.index[-1] + pd.DateOffset(months=1)
    forecast_end = dataset.index[-1] + pd.DateOffset(months=forecast_horizon)
    revenue_future = revenue_pred_shifted.loc[forecast_start:forecast_end]

    # -----------------------------
    # Fig 1: Future Forecast
    # -----------------------------
    st.subheader(f"{forecast_horizon}-Month Ahead Forecast")
    fig1 = plot_revenue_vs_prediction(
        revenue_future.index,
        None,
        revenue_future.values,
        sector,
        forecast_start_idx=0,
        error_pct=error_pct,
        xaxis_type="month"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown(
        f"""
        **Note:** Forecasts beyond 6 months should be treated with caution due to limited macroeconomic time-series training.  

        Horizon: **{forecast_horizon} months**  
        Confidence band: 95%
        """
    )

    # -----------------------------
    # Fig 2: Historical + Forecast
    # -----------------------------
    st.subheader("Historical vs Forecast")
    fig2 = plot_revenue_vs_prediction(
        dataset.index,
        revenue_now,
        revenue_pred,
        sector,
        forecast_start_idx=len(dataset.index),
        error_pct=error_pct,
        xaxis_type="year"
    )
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Insights Tab
# -----------------------------
with insights_tab:
    st.title("Business Insights")

    sector = selected_sector
    st.header(f"{sector} Planning Insights")

    probabilities = trained_models[sector].predict_proba(
        dataset[[col for col in dataset.columns if col.startswith(sector) and not col.endswith("_target")] + ["Month"]]
    )[:, 1]
    inv = inventory_pct[sector]

    # Fig 3: Probabilities & Inventory
    fig3 = plot_probability_and_inventory(dataset.index, probabilities, inv, inventory_threshold, sector)
    st.plotly_chart(fig3, use_container_width=True)

    # -----------------------------
    # Recommendations
    # -----------------------------
    st.subheader("Inventory Strategy Recommendations")

    st.markdown(
        """
        **Safe to delay/reduce orders**  
        *Demand weak, stock healthy* → Maintain or slow down orders.  

        **Restock aggressively**  
        *Demand high, stock below threshold* → Increase purchasing quickly.  

        **Moderate adjustment**  
        *Demand strong, stock healthy* → Adjust orders upward slightly, no urgency.  

        **Caution zone**  
        *Demand weak, stock below threshold* → Delay large purchases, monitor closely.
        """
    )
