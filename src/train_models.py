import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_all_sectors
from src.features import build_dataset
# from src.model import train_logistic_regression
from src.model import train_logistic_regression

# Load and prepare data
data = load_all_sectors()
dataset = build_dataset(data)

trained_models = {}

for sector in data.columns:
    y = dataset[f"{sector}_target"]
    X = dataset[[col for col in dataset.columns if col.startswith(sector) and not col.endswith("_target")] + ["Month"]]
    lr_model = train_logistic_regression(X, y) 
    # can also add probability here, but I guess at the moment not needed!
    trained_models[sector] = lr_model

# Save trained models
with open("trained_models.pkl", "wb") as f:
    pickle.dump(trained_models, f)

print("Model trained and saved successfully")