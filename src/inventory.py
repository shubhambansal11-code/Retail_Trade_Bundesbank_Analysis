# src/inventory.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.config import INVENTORY_THRESHOLD, SHORTAGE_ALPHA

# Here is a function to help define simulating an inventory
def simulate_inventory(dataset_index, seed=42):
    np.random.seed(seed)
    rw = np.cumsum(np.random.randn(len(dataset_index)) * 0.02) + 0.75
    inv_series = pd.Series(np.clip(rw, 0.2, 1.0), index=dataset_index)
    return inv_series

# Here is a function to help computing restock multiplier
def compute_restock_multiplier(inv_series):
    shortage = np.maximum(0.0, (INVENTORY_THRESHOLD - inv_series) / INVENTORY_THRESHOLD)
    restock_mult = 1.0 + SHORTAGE_ALPHA * shortage
    return restock_mult