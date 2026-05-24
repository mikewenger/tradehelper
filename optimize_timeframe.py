"""
Timeframe Optimizer — QQQ EMA 8/21 Crossover
Tests 5, 10, 15, 30, and 60-minute bars and ranks by Calmar ratio.
Uses real Massive.com option chain prices (same as the main backtest).
Cached option data is reused across timeframes — only new contracts are fetched.
"""
import time
import requests
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from pathlib import Path

from config import (MASSIVE_API_KEY, MASSIVE_BASE_URL, TICKER,
                    TAKE_PROFIT, STOP_LOSS, CONTRACTS, RISK_FREE_RATE)
from strategy.signals import add_signals
from strategy.options import otm_strike
from backtest.engine import run_backtest
from data.options_fetcher import prefetch_options

EST = pytz.timezone("US/Eastern")
TIMEFRAMES = [5, 10, 15, 30, 60]   # minutes to test


# ── Data fetch (per timeframe) ─────────────────────────────────────────────────

def _get_with_retry(url: str, headers: dict, params: dict | None,
                    max_retries: int = 6) -> requests.Response:
    """GET with exponential backoff on 429 / 5xx."""
    delay = 2
    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 429:
            wait = delay * (2 ** attempt)
            print(f"    429 rate-limit — waiting {wait}s (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait)
            continue
        if resp.status_code >= 500:
            wait = delay * (2 ** attempt)
            print(f"    {resp.status_code} server error — waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError(f"Failed after {max_retries} retries: {url}")


def fetch_bars(multiplier: int) -> pd.DataFrame:
    # Try to reuse the main-app 15-min cache (same data, different filename)
    main_cache = Path("data/QQQ_15min.parquet")
    cache = Path(f"data/QQQ_{multiplier}min_tf.parquet")

    # For 15-min specifically, prefer the main app cache if it's fresh
    if multiplier == 15 and main_cache.exists():
        age = datetime.now() - datetime.fromtimestamp(main_cache.stat().st_mtime)
        if age < timedelta(hours=24):
            print(f"  [15-min] Using cached main-app data.")
            return pd.read_parquet(main_cache)

    if cache.exists():
        age = datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime)
        if age < timedelta(hours=24):
            print(f"  [{multiplier}-min] Using cached data.")
            return pd.read_parquet(cache)

    to_date   = datetime.now(EST).strftime("%Y-%m-%d")
    from_date = (datetime.now(EST) - timedelta(days=183)).strftime("%Y-%m-%d")

    print(f"  Fetching {multiplier}-min bars ({from_date} to {to_date})...")
    url    = (f"{MASSIVE_BASE_URL}/v2/aggs/ticker/{TICKER}"
              f"/range/{multiplier}/minute/{from_date}/{to_date}")
    headers = {"Authorization": f"Bearer {MASSIVE_API_KEY}"}
    params  = {"adjusted": "true", "sort": "asc", "limit": 50000}
    results = []

    while url:
        resp = _get_with_retry(url, headers,
                               params if "?" not in url else None)
        data = resp.json()
        results.extend(data.get("results", []))
        url    = data.get("next_url")
        params = None

    if not results:
        raise RuntimeError(f"No data returned for {multiplier}-min bars.")

    df = pd.DataFrame(results)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low",
                             "c": "close", "v": "volume", "t": "timestamp"})
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(EST)
    df = df.set_index("datetime").sort_index()
    df = df[["open", "high", "low", "close", "volume"]]
    df = df[df.index.dayofweek < 5]     # weekdays only (all hours for EMA)

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


# ── EMA + volatility (timeframe-aware) ────────────────────────────────────────

def add_emas(df: pd.DataFrame, multiplier: int) -> pd.DataFrame:
    df = df.copy()
    df["ema8"]  = df["close"].ewm(span=8,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    bars_per_day  = 390 // multiplier          # 390 trading minutes per day
    bars_per_year = 252 * bars_per_day
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["hist_vol"] = log_ret.rolling(20).std() * np.sqrt(bars_per_year)
    return df


# ── Signal collector (for option pre-fetch) ───────────────────────────────────

def collect_signals(df: pd.DataFrame) -> list[dict]:
    signals = []
    for ts, row in df[df["in_market_hours"]].iterrows():
        if row["cross_up"]:
            signals.append({"date": ts.date(), "option_type": "call",
                             "strike": otm_strike(row["close"], "call")})
        elif row["cross_down"]:
            signals.append({"date": ts.date(), "option_type": "put",
                             "strike": otm_strike(row["close"], "put")})
    return signals


# ── Stats helper ──────────────────────────────────────────────────────────────

def stats(log: pd.DataFrame, multiplier: int) -> dict:
    if log.empty:
        return {"tf_min": multiplier, "trades": 0}

    pnl    = log["pnl"]
    cum    = pnl.cumsum()
    dd     = (cum - cum.cummax()).min()
    calmar = round(pnl.sum() / abs(dd), 3) if dd != 0 else 0
    reasons = log.get("exit_reason", pd.Series(dtype=str))

    return {
        "tf_min":      multiplier,
        "trades":      len(log),
        "win_rate":    round((pnl > 0).mean() * 100, 1),
        "total_pnl":   round(pnl.sum(), 2),
        "avg_pnl":     round(pnl.mean(), 2),
        "max_dd":      round(dd, 2),
        "calmar":      calmar,
        "tp_hits":     int((reasons == "TP").sum()),
        "sl_hits":     int((reasons == "SL").sum()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nTimeframe Optimizer — QQQ EMA 8/21 Crossover")
    print(f"TP: +${TAKE_PROFIT:,.0f}  |  SL: -${abs(STOP_LOSS):,.0f}  |  {CONTRACTS} contracts")
    print("=" * 72)

    rows = []
    for i, tf in enumerate(TIMEFRAMES):
        print(f"\n[{tf}-min]")
        try:
            df = fetch_bars(tf)
            df = add_emas(df, tf)
            df = add_signals(df)
            df = df[df["in_market_hours"]]

            log = run_backtest(df, use_real_prices=False)   # Black-Scholes — fast
            r = stats(log, tf)
            rows.append(r)
            if r["trades"]:
                print(f"  Trades: {r['trades']}  |  Win: {r['win_rate']}%  |  "
                      f"P&L: ${r['total_pnl']:,.2f}  |  "
                      f"MaxDD: ${r['max_dd']:,.2f}  |  Calmar: {r['calmar']}")
            else:
                print("  No trades generated.")
        except Exception as e:
            print(f"  ERROR: {e}")

        pass  # no inter-timeframe delay needed on upgraded plan

    if not rows:
        print("No results.")
    else:
        results = pd.DataFrame([r for r in rows if r.get("trades", 0) > 0])
        results = results.sort_values("calmar", ascending=False)

        print("\n" + "=" * 72)
        print("RANKING BY CALMAR RATIO  (total P&L / max drawdown)")
        print("=" * 72)
        print(f"{'Rank':<5} {'TF':>6} {'Trades':>7} {'Win%':>7} "
              f"{'Total P&L':>12} {'Avg P&L':>9} {'Max DD':>11} "
              f"{'Calmar':>8} {'TP':>5} {'SL':>5}")
        print("-" * 72)
        for rank, (_, r) in enumerate(results.iterrows(), 1):
            print(f"{rank:<5} {r['tf_min']:>4}min "
                  f"{r['trades']:>7} {r['win_rate']:>6}% "
                  f"${r['total_pnl']:>10,.2f} "
                  f"${r['avg_pnl']:>7,.2f} "
                  f"${r['max_dd']:>9,.2f} "
                  f"{r['calmar']:>8.3f} "
                  f"{r['tp_hits']:>5} {r['sl_hits']:>5}")

        best = results.iloc[0]
        print(f"\n  WINNER: {best['tf_min']}-minute bars  "
              f"(Calmar {best['calmar']}, "
              f"${best['total_pnl']:,.2f} P&L, "
              f"{best['win_rate']}% win rate)")
