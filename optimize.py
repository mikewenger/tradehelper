"""
Grid search over Take Profit / Stop Loss combinations.
Ranks by Profit Factor (gross wins / gross losses) weighted with total P&L.
Prints full results table, then writes the winner to config.py.
"""
import re
import itertools
import pandas as pd
import numpy as np
from pathlib import Path

from data.fetcher import fetch_qqq_bars
from strategy.indicators import add_emas
from strategy.signals import add_signals
from backtest.engine import run_backtest
import config
from config import BACKTEST_START

# ── grid ──────────────────────────────────────────────────────────────────────
TP_VALUES  = [200, 300, 500, 750, 1000, 1500, 2000, 3000]   # take-profit $
SL_VALUES  = [50, 75, 100, 150, 200, 300, 500]               # stop-loss   $


def metrics(trade_log: pd.DataFrame) -> dict:
    if trade_log.empty or len(trade_log) < 5:
        return None

    pnl = trade_log["pnl"]
    total   = pnl.sum()
    wins    = pnl[pnl > 0].sum()
    losses  = abs(pnl[pnl < 0].sum())
    pf      = wins / losses if losses > 0 else float("inf")
    win_pct = (pnl > 0).mean() * 100

    # max drawdown on cumulative P&L curve
    cumulative = pnl.cumsum()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max)
    max_dd  = drawdown.min()

    # Calmar-style ratio: total P&L / max drawdown (higher = better)
    calmar = total / abs(max_dd) if max_dd != 0 else 0.0

    # daily Sharpe
    trade_log = trade_log.copy()
    trade_log["date"] = pd.to_datetime(trade_log["date"])
    daily = trade_log.groupby("date")["pnl"].sum()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0

    return {
        "trades":   len(trade_log),
        "win_pct":  round(win_pct, 1),
        "total_pnl": round(total, 2),
        "profit_factor": round(pf, 2),
        "max_dd":   round(max_dd, 2),
        "calmar":   round(calmar, 3),
        "sharpe":   round(sharpe, 3),
    }


def main():
    print("Loading data...")
    df = fetch_qqq_bars()
    df = add_emas(df)
    df = add_signals(df)
    df = df[df["in_market_hours"]]
    if BACKTEST_START:
        df = df[df.index >= pd.Timestamp(BACKTEST_START, tz=df.index.tz)]
        print(f"Backtest range: {df.index[0].date()} to {df.index[-1].date()}")

    results = []
    combos = list(itertools.product(TP_VALUES, SL_VALUES))
    print(f"Testing {len(combos)} TP/SL combinations (real prices, Real-only trades)...\n")

    for tp, sl in combos:
        # temporarily override config values in-process
        config.TAKE_PROFIT =  float(tp)
        config.STOP_LOSS   = -float(sl)

        # re-import engine so it picks up the new config values
        import importlib
        import backtest.engine as eng
        importlib.reload(eng)

        trade_log = eng.run_backtest(df, use_real_prices=True)
        trade_log = trade_log[trade_log["pricing"] == "Real"].copy()
        m = metrics(trade_log)
        if m:
            results.append({"tp": tp, "sl": sl, **m})

    results_df = pd.DataFrame(results)

    # ── rank: primary = Calmar ratio (P&L / drawdown), secondary = total P&L
    results_df = results_df.sort_values(
        ["calmar", "total_pnl"], ascending=[False, False]
    ).reset_index(drop=True)

    pd.set_option("display.float_format", "{:,.2f}".format)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 120)

    print("=" * 100)
    print(f"{'Rank':>4}  {'TP':>6}  {'SL':>6}  {'Trades':>6}  {'Win%':>6}  "
          f"{'Total P&L':>12}  {'Prof.Factor':>11}  {'Max DD':>10}  {'Calmar':>8}  {'Sharpe':>8}")
    print("-" * 100)
    for i, row in results_df.iterrows():
        print(f"{i+1:>4}  ${row['tp']:>5}  ${row['sl']:>5}  {row['trades']:>6}  "
              f"{row['win_pct']:>5.1f}%  ${row['total_pnl']:>11,.2f}  "
              f"{row['profit_factor']:>11.2f}  ${row['max_dd']:>9,.2f}  "
              f"{row['calmar']:>8.3f}  {row['sharpe']:>8.3f}")
    print("=" * 100)

    best = results_df.iloc[0]
    print(f"\nBEST:  TP=${best['tp']:,.0f}  SL=${best['sl']:,.0f}  ->  "
          f"{best['trades']} trades | {best['win_pct']}% wins | "
          f"${best['total_pnl']:,.2f} P&L | Calmar {best['calmar']:.3f}\n")

    # ── write winner to config.py ─────────────────────────────────────────────
    cfg_path = Path(__file__).parent / "config.py"
    text = cfg_path.read_text()
    text = re.sub(r"TAKE_PROFIT\s*=\s*[\d.]+", f"TAKE_PROFIT = {float(best['tp'])}", text)
    text = re.sub(r"STOP_LOSS\s*=\s*-?[\d.]+", f"STOP_LOSS   = -{float(best['sl'])}", text)
    cfg_path.write_text(text)
    print(f"config.py updated → TAKE_PROFIT={best['tp']}  STOP_LOSS=-{best['sl']}")


if __name__ == "__main__":
    main()
