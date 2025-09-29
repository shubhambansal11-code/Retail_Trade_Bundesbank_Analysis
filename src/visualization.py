# src/visualization.py
import os
import numpy as np
import plotly.graph_objects as go

def plot_revenue_vs_prediction(idx, revenue_now, revenue_pred, sector,
                               forecast_start_idx=None, error_pct=0.05,
                               xaxis_type="year", save_dir="figures"):
    """
    Interactive plot of current vs predicted revenue using Plotly.

    Args:
        idx: pd.DatetimeIndex or list of timestamps
        revenue_now: np.array or pd.Series of current revenue/index
        revenue_pred: np.array or pd.Series of predicted revenue/index
        sector: str, name of the sector
        forecast_start_idx: int, index to start forecast shading
        error_pct: float, error band size (default 0.05 = 95%)
        xaxis_type: "month" for monthly view, "year" for yearly ticks
    """
    os.makedirs(save_dir, exist_ok=True)

    fig = go.Figure()

    # Historical line
    if revenue_now is not None:
        fig.add_trace(go.Scatter(
            x=idx, y=revenue_now,
            mode="lines", name="Current revenue (proxy)",
            line=dict(color="black", width=2)
        ))

    # Predicted + error band
    if revenue_pred is not None:
        upper = np.array(revenue_pred) * (1 + error_pct)
        lower = np.array(revenue_pred) * (1 - error_pct)
        fig.add_trace(go.Scatter(
            x=idx, y=upper, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=idx, y=lower, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(0,0,255,0.1)",
            name=f"95% Confidence Band"
        ))
        fig.add_trace(go.Scatter(
            x=idx, y=revenue_pred,
            mode="lines", name="Predicted revenue (Logistic)",
            line=dict(dash="dash", color="blue")
        ))

    # Highlight forecast horizon
    if forecast_start_idx is not None and forecast_start_idx < len(idx):
        fig.add_vrect(
            x0=idx[forecast_start_idx], x1=idx[-1],
            fillcolor="white", opacity=0.3,
            layer="below", line_width=0
        )

    # X-axis formatting
    if xaxis_type == "month":
        xaxis_cfg = dict(title="Month", tickformat="%b %Y")
    else:  # yearly ticks
        xaxis_cfg = dict(title="Year", dtick="M12", tickformat="%Y")

    fig.update_layout(
        yaxis=dict(
            title="Revenue (€)",
            tickformat=",.0f"
        ),
        xaxis=xaxis_cfg,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        template="plotly_white",
        margin=dict(t=50, b=80, l=40, r=40)
    )

    # Save interactive HTML
    file_path = f"{save_dir}/{sector}_revenue_vs_prediction.html"
    fig.write_html(file_path)

    return fig


def plot_probability_and_inventory(idx, probabilities, inventory, threshold, sector, save_dir="figures"):
    """
    Interactive plot of logistic regression probability vs inventory %.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=idx, y=probabilities,
        #mode="lines", name="P(Up in 6m) - Logistic",
        mode="lines", name="Demand Probability",
        line=dict(color="orange")
    ))

    fig.add_trace(go.Scatter(
        x=idx, y=inventory,
        mode="lines", name="Inventory %",
        line=dict(color="green")
    ))

    fig.add_trace(go.Scatter(
        x=idx, y=[threshold] * len(idx),
        mode="lines", name=f"Threshold ({threshold*100:.0f}%)",
        line=dict(color="red", dash="dash")
    ))

    shortage_mask = inventory < threshold
    fig.add_trace(go.Scatter(
        x=np.array(idx)[shortage_mask],
        y=np.array(inventory)[shortage_mask],
        mode="markers",
        name="Inventory below threshold",
        marker=dict(color="red", size=8, symbol="triangle-down")
    ))

    fig.update_layout(
        title="Probability vs Inventory",
        xaxis=dict(title="Year", dtick="M12", tickformat="%Y"),
        yaxis=dict(title="Probability / Inventory %", range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        template="plotly_white",
        margin=dict(t=50, b=80, l=40, r=40)
    )

    file_path = f"{save_dir}/{sector}_probability_inventory.html"
    fig.write_html(file_path)

    return fig
