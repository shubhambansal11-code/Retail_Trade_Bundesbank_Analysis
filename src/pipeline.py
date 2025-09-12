# src/pipeline.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_all_sectors
from src.features import build_dataset
from src.model import train_logistic_regression, compute_forward6_stats
from src.inventory import simulate_inventory, compute_restock_multiplier
from src.visualization import (
    plot_revenue_vs_prediction,
    plot_probability_and_inventory,
)
from src.config import BASE_REVENUE_LATEST, INVENTORY_THRESHOLD


def run_pipeline():
    data = load_all_sectors()
    dataset = build_dataset(data)
    sectors = data.columns

    for sector in sectors:
        print(f"\n=== {sector} ===")
        idx = dataset.index
        y = dataset[f"{sector}_target"]
        X = dataset[
            [col for col in dataset.columns if col.startswith(sector) and not col.endswith("_target")]
            + ["Month"]
        ]

        # 1. Logistic regression → probabilities
        prob_all, _ = train_logistic_regression(X, y)

        # 2. Convert probability into expected forward 6% change
        mean_pos, mean_neg, _ = compute_forward6_stats(data[sector])
        exp_fwd6_pct = prob_all * mean_pos + (1 - prob_all) * mean_neg

        # 3. Current & predicted index
        index_values = data[sector].reindex(idx).values
        pred_idx = index_values * (1 + exp_fwd6_pct / 100)

        # 4. Scale into revenue units if configured, well yes now!
        last_index = data[sector].iloc[-1]
        revenue_now = (index_values / last_index) * BASE_REVENUE_LATEST[sector]
        revenue_pred = (pred_idx / last_index) * BASE_REVENUE_LATEST[sector]

        # 5. Inventory simulation & restock multiplier
        inv = simulate_inventory(idx)
        restock_mult = compute_restock_multiplier(inv)

        # 6. Plots (saved to figures/ by visualization.py)
        forecast_start_idx = len(idx) - 6
        plot_revenue_vs_prediction(idx, revenue_now, revenue_pred, sector, forecast_start_idx=forecast_start_idx)

        plot_probability_and_inventory(idx, prob_all, inv, INVENTORY_THRESHOLD, sector)
