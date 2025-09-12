# src/visualization.py
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_revenue_vs_prediction(idx, revenue_now, revenue_pred, sector, forecast_start_idx=None, save_dir="figures"):
    """
    Plot current vs predicted revenue (or index) and optionally highlight forecast horizon.

    Args:
        idx: pd.DatetimeIndex or list of timestamps
        revenue_now: np.array or pd.Series of current revenue/index
        revenue_pred: np.array or pd.Series of predicted revenue/index
        sector: str, name of the sector
        forecast_start_idx: int, index to start 6-month forecast shading
        save_dir: str, directory to save figure
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(idx, revenue_now, label="Current revenue (proxy)", color="black", lw=1.5)
    ax.plot(idx, revenue_pred, label="Predicted revenue (Logistic)", linestyle="--", alpha=0.9)

    # Highlight forecast horizon if provided
    if forecast_start_idx is not None and forecast_start_idx < len(idx):
        ax.axvspan(idx[forecast_start_idx], idx[-1], color="gray", alpha=0.2, label="6-month forecast horizon")

    ax.set_title(f"{sector} — Current vs Predicted Revenue")
    ax.set_ylabel("Revenue (or Index units)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{save_dir}/{sector}_revenue_vs_prediction.png")

    return fig


def plot_probability_and_inventory(idx, probabilities, inventory, threshold, sector, save_dir="figures"):
    """
    Plot logistic regression probability of upward movement and inventory percentage.
    Highlights points where inventory falls below threshold.

    Args:
        idx: pd.DatetimeIndex or list of timestamps
        probabilities: np.array or pd.Series of P(up in 6m)
        inventory: np.array or pd.Series of inventory coverage (0 to 1, percentage)
        threshold: float, inventory threshold
        sector: str, sector name
        save_dir: str, directory to save figure
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(idx, probabilities, label="P(Up in 6m) - Logistic", color="orange")
    ax1.set_ylabel("Probability")
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()
    ax2.plot(idx, inventory, label="Inventory %", color="tab:green", alpha=0.6)
    ax2.axhline(threshold, color="red", linestyle="--", label=f"Threshold ({threshold*100:.0f}%)")

    # highlight shortage points
    shortage_mask = inventory < threshold
    ax2.scatter(np.array(idx)[shortage_mask], np.array(inventory)[shortage_mask], color="red", marker="v", zorder=10, label="Inventory below threshold")
    ax2.set_ylabel("Inventory %")

    # combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(f"{save_dir}/{sector}_probability_inventory.png")

    return fig