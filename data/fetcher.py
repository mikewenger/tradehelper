import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import pytz

from config import MASSIVE_API_KEY, MASSIVE_BASE_URL, TICKER

CACHE_FILE = Path(__file__).parent / "QQQ_15min.parquet"
CACHE_MAX_AGE_HOURS = 24
EST = pytz.timezone("US/Eastern")


def _bars_url(ticker: str, multiplier: int, timespan: str, from_date: str, to_date: str) -> str:
    return f"{MASSIVE_BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"


def _fetch_all_pages(url: str) -> list[dict]:
    headers = {"Authorization": f"Bearer {MASSIVE_API_KEY}"}
    params = {"adjusted": "true", "sort": "asc", "limit": 50000}
    results = []

    while url:
        resp = requests.get(url, headers=headers, params=params if "?" not in url else None, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results.extend(data.get("results", []))
        url = data.get("next_url")
        params = None  # next_url already has params baked in

    return results


def fetch_qqq_bars() -> pd.DataFrame:
    if CACHE_FILE.exists():
        age = datetime.now() - datetime.fromtimestamp(CACHE_FILE.stat().st_mtime)
        if age < timedelta(hours=CACHE_MAX_AGE_HOURS):
            print(f"Loading cached data from {CACHE_FILE}")
            return pd.read_parquet(CACHE_FILE)

    to_date = datetime.now(EST).strftime("%Y-%m-%d")
    from_date = (datetime.now(EST) - timedelta(days=183)).strftime("%Y-%m-%d")

    print(f"Fetching {TICKER} 15-min bars from Massive.com ({from_date} to {to_date}, all hours)...")
    url = _bars_url(TICKER, 15, "minute", from_date, to_date)
    raw = _fetch_all_pages(url)

    if not raw:
        raise RuntimeError("No data returned from Massive API. Check your API key and tier.")

    df = pd.DataFrame(raw)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
                              "vw": "vwap", "t": "timestamp", "n": "trades"})
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(EST)
    df = df.set_index("datetime").sort_index()
    df = df[["open", "high", "low", "close", "volume", "vwap"]]

    # Keep ALL hours (including pre/post market) so EMAs are computed continuously
    # — exactly like ThinkorSwim. Market-hours filtering happens downstream after
    # add_emas(), so signals only fire 9:30–16:00 EST.
    is_weekday = df.index.dayofweek < 5
    df = df[is_weekday]

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_FILE)
    print(f"Fetched {len(df)} bars (all hours, weekdays). Cached to {CACHE_FILE}")
    return df
