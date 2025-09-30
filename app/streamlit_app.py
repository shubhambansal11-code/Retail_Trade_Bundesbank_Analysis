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
st.sidebar.header("Forecast Inputs")

# Load sector data
sectors_data = load_all_sectors()
sectors = list(sectors_data.columns)

# Restrict to single selection
selected_sector = st.sidebar.selectbox("Select sector", sectors)

# Forecast horizon (5–12 months)
forecast_horizon = st.sidebar.slider("Forecast horizon (months)", min_value=5, max_value=12, value=6)

# Fixed confidence/error margin = 95%
error_pct = 0.05  

st.sidebar.header("Business Calibration Inputs")

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

st.title("Multi-sector retail trade analysis")
st.write(
        """
        This project analyzes and forecasts retail trade sector revenues in Germany using
        official data (until July 2025) from the **Deutsche Bundesbank** and demonstrate business insights for the coming months in 2025.
        
        Several machine learning and statistical models were tested:
        - Random Forest  
        - Decision Trees  
        - Prophet  
        - Logistic Regression  

        After evaluation, **Logistic Regression** was chosen for its consistent predictive
        performance across sectors.
        """
    )

# -----------------------------
# Tabs for Forecast & Insights
# -----------------------------
forecast_tab, insights_tab = st.tabs(["Forecast", "Business Insights"])

# -----------------------------
# Forecast Tab
# -----------------------------
with forecast_tab:
    #st.title("Forecasts")

    sector = selected_sector
    st.header(f"{sector} Revenue Forecast")

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
    st.plotly_chart(fig1, use_container_width=True, key="future_forecast")

    st.markdown(
        f"""
        **Note:** Forecasts beyond 6 months should be treated with caution due to limited macroeconomic time-series training.  
        The model used is **Logistic Regression**, which performed best compared to Random Forest, Decision Trees, and Prophet.  
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
    st.plotly_chart(fig2, use_container_width=True, key="hist_vs_forecast")

# -----------------------------
# Insights Tab
# -----------------------------
with insights_tab:
    #st.title("Business Insights")

    sector = selected_sector
    st.header(f"{sector} Planning Insights")

    probabilities = trained_models[sector].predict_proba(
        dataset[[col for col in dataset.columns if col.startswith(sector) and not col.endswith("_target")] + ["Month"]]
    )[:, 1]
    inv = inventory_pct[sector]
    
    st.subheader("Probability vs Inventory")
    # Fig 3: Probabilities & Inventory
    fig3 = plot_probability_and_inventory(dataset.index, probabilities, inv, inventory_threshold, sector)
    st.plotly_chart(fig3, use_container_width=True, key="prob_inventory")

    # -----------------------------
    # Current Scenario
    # -----------------------------
    st.subheader("Current Scenario")

    latest_prob = probabilities[-1]
    latest_inv = inv.iloc[-1]

    if latest_prob > 0.6 and latest_inv < inventory_threshold:
        scenario = "Restock aggressively"
        explanation = f"Demand probability is **{latest_prob:.2f} (>0.6, strong demand)** and inventory is **{latest_inv:.2f} (<{inventory_threshold}, low stock)**."
        recommendation = "Increase stock levels by **25–40%** immediately to avoid shortages."
    elif latest_prob > 0.6 and latest_inv >= inventory_threshold:
        scenario = "Moderate adjustment"
        explanation = f"Demand probability is **{latest_prob:.2f} (>0.6, strong demand)** and inventory is **{latest_inv:.2f} (≥{inventory_threshold}, healthy)**."
        recommendation = "Increase orders moderately by **5–10%**."
    elif latest_prob <= 0.6 and latest_inv >= inventory_threshold:
        scenario = "Safe to reduce orders"
        explanation = f"Demand probability is **{latest_prob:.2f} (≤0.6, weak demand)** and inventory is **{latest_inv:.2f} (≥{inventory_threshold}, healthy)**."
        recommendation = "Reduce incoming orders by **10–15%**."
    else:
        scenario = "Caution zone"
        explanation = f"Demand probability is **{latest_prob:.2f} (≤0.6, weak demand)** and inventory is **{latest_inv:.2f} (<{inventory_threshold}, low stock)**."
        recommendation = "Restock minimally (**~5%**) and closely monitor demand signals."
    if scenario == "Safe to reduce orders" or scenario == "Moderate adjustment": 
        st.info(f"### **{scenario}**") 
    if scenario == "Caution zone": 
        st.warning(f"### **{scenario}**") 
    if scenario == "Restock aggressively": 
        st.error(f"### **{scenario}**") 
    st.write(explanation)
    st.markdown(f"**Recommended Action:** {recommendation}")

    st.markdown("---")
    st.markdown(
        "### Learn about all Possible Scenarios\n"
        "Each scenario is determined by a combination of **demand probability** (>0.6 = strong, ≤0.6 = weak) "
        "and **inventory levels** (≥ threshold = healthy, < threshold = low). "
        "Click the tabs below to see explanations and recommended actions."
    )

    # -----------------------------
    # Scenario Tabs
    # -----------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Safe to reduce orders", "Restock aggressively", "Moderate adjustment", "Caution zone"]
    )

    with tab1:
        #st.info("### Safe to reduce orders")
        st.write("**Criteria:** Demand probability ≤0.6 (weak) + Inventory ≥ threshold (healthy).")
        st.write("**Explanation:** Demand is expected to be low while stock is sufficient, reducing risk of shortages.")
        st.write("**Recommended Action:** Reduce orders by **10–15%**.")
        fig = plot_probability_and_inventory(dataset.index, probabilities, inv, inventory_threshold, sector)
        #st.plotly_chart(fig, use_container_width=True, key="scenario_safe_reduce")

    with tab2:
        #st.error("### Restock aggressively")
        st.write("**Criteria:** Demand probability >0.6 (strong) + Inventory < threshold (low).")
        st.write("**Explanation:** Demand is strong but inventory is insufficient, creating high risk of stockouts.")
        st.write("**Recommended Action:** Restock by **25–40%** immediately.")
        fig = plot_probability_and_inventory(dataset.index, probabilities, inv, inventory_threshold, sector)
        #st.plotly_chart(fig, use_container_width=True, key="scenario_restock_aggressive")

    with tab3:
        #st.info("### Moderate adjustment")
        st.write("**Criteria:** Demand probability >0.6 (strong) + Inventory ≥ threshold (healthy).")
        st.write("**Explanation:** Strong demand is expected but inventory is at safe levels, reducing urgency.")
        st.write("**Recommended Action:** Increase orders by **5–10%**.")
        fig = plot_probability_and_inventory(dataset.index, probabilities, inv, inventory_threshold, sector)
        #st.plotly_chart(fig, use_container_width=True, key="scenario_moderate_adjustment")

    with tab4:
        #st.warning("### Caution zone")
        st.write("**Criteria:** Demand probability ≤0.6 (weak) + Inventory < threshold (low).")
        st.write("**Explanation:** Demand is weak while inventory is already low, meaning extra orders may not convert to sales.")
        st.write("**Recommended Action:** Restock minimally (**~5%**) and monitor closely.")
        fig = plot_probability_and_inventory(dataset.index, probabilities, inv, inventory_threshold, sector)
        #st.plotly_chart(fig, use_container_width=True, key="scenario_caution_zone")

