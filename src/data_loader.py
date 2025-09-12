# src/data_loader.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.config import BUNDESBANK_URLS

def load_csv(url: str, sector: str) -> pd.DataFrame:
    df = pd.read_csv(url, skiprows=9)
    df = df.iloc[:, :2]
    df = df.rename(columns={df.columns[0]: "Date", df.columns[1]: sector})
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m", errors="coerce")
    df = df.set_index("Date")
    return df

def load_all_sectors() -> pd.DataFrame:
    dfs = [load_csv(url, sector) for sector, url in BUNDESBANK_URLS.items()]
    data = pd.concat(dfs, axis=1).dropna()
    return data