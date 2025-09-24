# src/visualization.py
import os
import numpy as np
import plotly.graph_objects as go

def plot_revenue_vs_prediction(idx, revenue_now, revenue_pred, sector, forecast_start_idx=None, save_dir="figures"):
    """
    Interactive plot of current vs predicted revenue using Plotly.

    Args:
        idx: pd.DatetimeIndex or list of timestamps
        revenue_now: np.array or pd.Series of current revenue/index
        revenue_pred: np.array or pd.Series of predicted revenue/index
        sector: str, name of the sector
        forecast_start_idx: int, index to start forecast shading
        save_dir: str, directory to save figure
    """
    os.makedirs(save_dir, exist_ok=True)

    fig = go.Figure()

    # Current revenue
    fig.add_trace(go.Scatter(
        x=idx, y=revenue_now,
        mode="lines", name="Current revenue (proxy)",
        line=dict(color="black", width=2)
    ))

    # Predicted revenue
    fig.add_trace(go.Scatter(
        x=idx, y=revenue_pred,
        mode="lines", name="Predicted revenue (Logistic)",
        line=dict(dash="dash", color="blue")
    ))

    # Highlight forecast horizon
    if forecast_start_idx is not None and forecast_start_idx < len(idx):
        fig.add_vrect(
            x0=idx[forecast_start_idx], x1=idx[-1],
            fillcolor="lightgray", opacity=0.3,
            layer="below", line_width=0,
            annotation_text="Forecast horizon", annotation_position="top left"
        )

    fig.update_layout(
        #title=f"{sector} — Current vs Predicted Revenue",
        title="Current vs Predicted Revenue",
        yaxis=dict(
            title="Revenue (€)",
            tickformat=",.0f"  # forces commas, no scientific notation
        ),
        #xaxis=dict(title="Date"),
        xaxis=dict(title="Year"),
        legend=dict(x=0.01, y=0.99, borderwidth=0),
        template="plotly_white"
    )

    # Save interactive HTML
    file_path = f"{save_dir}/{sector}_revenue_vs_prediction.html"
    fig.write_html(file_path)

    return fig


def plot_probability_and_inventory(idx, probabilities, inventory, threshold, sector, save_dir="figures"):
    """
    Interactive plot of logistic regression probability vs inventory %.

    Args:
        idx: pd.DatetimeIndex or list of timestamps
        probabilities: np.array or pd.Series of P(up in 6m)
        inventory: np.array or pd.Series of inventory coverage (0–1, percentage)
        threshold: float, inventory threshold
        sector: str, sector name
        save_dir: str, directory to save figure
    """
    os.makedirs(save_dir, exist_ok=True)

    fig = go.Figure()

    # Probability line
    fig.add_trace(go.Scatter(
        x=idx, y=probabilities,
        mode="lines", name="P(Up in 6m) - Logistic",
        line=dict(color="orange")
    ))

    # Inventory line
    fig.add_trace(go.Scatter(
        x=idx, y=inventory,
        mode="lines", name="Inventory %",
        line=dict(color="green")
    ))

    # Threshold line
    fig.add_trace(go.Scatter(
        x=idx, y=[threshold] * len(idx),
        mode="lines", name=f"Threshold ({threshold*100:.0f}%)",
        line=dict(color="red", dash="dash")
    ))

    # Highlight shortage points
    shortage_mask = inventory < threshold
    fig.add_trace(go.Scatter(
        x=np.array(idx)[shortage_mask],
        y=np.array(inventory)[shortage_mask],
        mode="markers",
        name="Inventory below threshold",
        marker=dict(color="red", size=8, symbol="triangle-down")
    ))

    fig.update_layout(
        #title=f"{sector} — Probability vs Inventory",
        title="Probability vs Inventory",
        xaxis=dict(title="Year"),
        yaxis=dict(title="Probability / Inventory %", range=[0, 1.05]),
        legend=dict(x=0.01, y=0.99, borderwidth=0),
        template="plotly_white"
    )

    # Save interactive HTML
    file_path = f"{save_dir}/{sector}_probability_inventory.html"
    fig.write_html(file_path)

    return fig