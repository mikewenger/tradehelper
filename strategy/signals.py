import pandas as pd
import pytz

from config import MARKET_OPEN, MARKET_CLOSE

EST = pytz.timezone("US/Eastern")


def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Market hours filter: 9:30–16:00 EST, weekdays only
    time = df.index.time
    open_t = pd.Timestamp(f"2000-01-01 {MARKET_OPEN}").time()
    close_t = pd.Timestamp(f"2000-01-01 {MARKET_CLOSE}").time()
    is_weekday = df.index.dayofweek < 5
    in_hours = (time >= open_t) & (time < close_t)
    df["in_market_hours"] = is_weekday & in_hours

    # Crossover signals (only meaningful during market hours)
    ema_diff = df["ema8"] - df["ema21"]
    prev_diff = ema_diff.shift(1)
    df["cross_up"] = (prev_diff <= 0) & (ema_diff > 0) & df["in_market_hours"]
    df["cross_down"] = (prev_diff >= 0) & (ema_diff < 0) & df["in_market_hours"]

    return df
