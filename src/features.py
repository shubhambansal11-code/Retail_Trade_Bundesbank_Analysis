# src/features.py
import pandas as pd

def build_target(df: pd.DataFrame, sector: str) -> pd.Series:
    idx = df[sector]
    yoy_forward = (idx.shift(-6) - idx.shift(6)) / idx.shift(6) * 100
    target = (yoy_forward > 0).astype(int).rename(f"{sector}_target")
    return target

def build_features(data: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=data.index)

    for sector in data.columns:
        features[f"{sector}_t-12"] = data[sector].shift(12)
        features[f"{sector}_YoY"] = data[sector].pct_change(12) * 100
        features[f"{sector}_MoM"] = data[sector].pct_change(1) * 100
        features[f"{sector}_vol_12m"] = data[sector].pct_change(1).rolling(12).std()
        features[f"{sector}_CumGrowth_6m"] = (
            (data[sector].pct_change(1) + 1).rolling(6).apply(lambda x: x.prod() - 1)
        ) * 100
        features[f"{sector}_RollMax_6m"] = data[sector].rolling(6).max()
        features[f"{sector}_RollMin_6m"] = data[sector].rolling(6).min()

    features["Month"] = data.index.month
    return features

def build_dataset(data: pd.DataFrame) -> pd.DataFrame:
    targets = [build_target(data, sector) for sector in data.columns]
    targets = pd.concat(targets, axis=1)
    features = build_features(data)
    dataset = pd.concat([features, targets], axis=1).dropna()
    return dataset