import pandas as pd
import numpy as np

from config import VOL_WINDOW


def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema8"] = df["close"].ewm(span=8, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    # annualized historical volatility from 15-min log returns
    log_returns = np.log(df["close"] / df["close"].shift(1))
    bars_per_year = 252 * 26  # 26 fifteen-minute bars per trading day
    df["hist_vol"] = log_returns.rolling(VOL_WINDOW).std() * np.sqrt(bars_per_year)
    return df
