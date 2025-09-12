# The lines below give a global path, otherwise path errors sweep in
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
from src.config import BASE_REVENUE_LATEST, INVENTORY_THRESHOLD

# -----------------------------
# Sidebar: User Inputs
# -----------------------------
st.sidebar.title("Retail Forecast Inputs")

# Load all sectors
sectors_data = load_all_sectors()
sectors = list(sectors_data.columns)
selected_sectors = st.sidebar.multiselect("Select sectors to display", sectors, default=sectors)

prediction_horizon = st.sidebar.slider("Prediction horizon (months)", min_value=1, max_value=12, value=6)

st.sidebar.markdown("---")
st.sidebar.subheader("Business Calibration")

# Business calibration inputs
inventory_threshold = st.sidebar.slider(
    "Inventory threshold",
    0.0, 1.0, 0.4,
    help="The minimum coverage level below which inventory is considered risky."
)

shortage_alpha = st.sidebar.slider(
    "Shortage amplifier",
    0.0, 5.0, 1.0,
    help="Controls aggressiveness of restocking when shortages are predicted. Higher = stronger correction."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Note on Revenues**")
st.sidebar.write(
    """
    Default baseline revenues are assumed for scaling forecasts (in €):  
    1. Electronics: **€5M**  
    2. Groceries: **€12M**  
    3. Textiles: **€2M**  
    4. Furniture: **€1.5M**  
    5. Pharmacy: **€3M**  
    6. Motor Vehicles: **€8M**  

    These serve as calibration anchors so that index predictions are shown in **monetary terms**, making the results more actionable for planning.
    """
)

# -----------------------------
# Load dataset
# -----------------------------
dataset = build_dataset(sectors_data)

# Load trained models
with open("trained_models.pkl", "rb") as f:
    trained_models = pickle.load(f)

# -----------------------------
# Simulate Inventory (random walk for demo purposes)
# -----------------------------
inventory_pct = {}
for s in sectors:
    n = len(dataset.index)
    rw = np.cumsum(np.random.randn(n) * 0.02) + 0.75
    inv_series = pd.Series(np.clip(rw, 0.2, 1.0), index=dataset.index)
    inventory_pct[s] = inv_series

# -----------------------------
# Main Loop: Forecast per sector
# -----------------------------
for sector in selected_sectors:
    st.header(f"{sector} Forecast")

    y = dataset[f"{sector}_target"]
    X = dataset[[col for col in dataset.columns if col.startswith(sector) and not col.endswith("_target")] + ["Month"]]

    # Predict using pre-trained model
    lr_model = trained_models[sector]
    probabilities = lr_model.predict_proba(X)[:, 1]

    # Compute expected forward change
    index_values = sectors_data[sector].reindex(dataset.index).values
    mean_pos, mean_neg = probabilities.mean(), probabilities.mean()  # placeholder, could compute historical stats
    exp_fwd_pct = probabilities * mean_pos + (1 - probabilities) * mean_neg
    pred_idx = index_values * (1 + exp_fwd_pct / 100)

    # Scale into revenue terms
    last_index = sectors_data[sector].iloc[-1]
    revenue_now = (index_values / last_index) * BASE_REVENUE_LATEST[sector]
    revenue_pred = (pred_idx / last_index) * BASE_REVENUE_LATEST[sector]

    inv = inventory_pct[sector]

    # -----------------------------
    # Plots + Explanations
    # -----------------------------
    st.subheader("Revenue Forecast")
    fig1 = plot_revenue_vs_prediction(
        dataset.index,
        revenue_now,
        revenue_pred,
        sector,
        forecast_start_idx=len(dataset.index) - prediction_horizon,
    )
    st.pyplot(fig1)

    st.markdown(
        """
        **Interpretation:**  
        1. Black line = historical revenue (scaled from index).  
        2. Orange dashed line = forecasted revenue.  
        3. Gray shaded area = forecast horizon.  

        If the forecasted orange line trends **up**, demand is expected to increase.  
        If it trends **down**, expect weaker sales.
        """
    )

    st.subheader("📊 Probability & Inventory Planning")
    fig2 = plot_probability_and_inventory(dataset.index, probabilities, inv, inventory_threshold, sector)
    st.pyplot(fig2)

    st.markdown(
        """
        **How to interpret:**  
        1. Orange line = probability demand will rise in the months input from the sidebar.  
        2. Green line = inventory coverage.  
        3. Red dashed line = safety threshold.  
        4. Red dots = inventory below threshold (risky).  

        **Scenarios:**  
        1. *Orange high* & *Green below red*: ** Restock aggressively** (demand high, stock low).  
        2. *Orange low* & *Green above red*: ** Safe to delay/reduce orders** (demand weak, stock healthy).  
        3. *Orange high* & *Green above red*: **Moderate adjustment** (demand strong, but inventory ok).  
        4. *Orange low* & *Green below red*: **Tricky case** — demand weak but stock short. Restock cautiously.
        """
    )