"""
Compare all 4 exit strategies against real Massive.com prices.
Ranks by Calmar ratio (total P&L / max drawdown).

Exit strategies tested:
  1. Trailing stop  — track highest bar high (CALL) / lowest bar low (PUT) since entry;
                      exit when close < prev bar's low (CALL) or close > prev bar's high (PUT)
  2. Lower/Higher   — exit when bar high < prev bar high (CALL) or bar low > prev bar low (PUT)
  3. Day high/low   — exit at the bar that makes the day's absolute high (CALL) or low (PUT);
                      theoretical best-case (uses intraday lookahead within same bar sequence)
  4. TP/SL/Force only — remove all trend-based exits; hold until TP, SL, or 3:55 PM force-close
"""
import pandas as pd
import numpy as np
import pytz

from data.fetcher import fetch_qqq_bars
from data.options_fetcher import get_real_option_price
from strategy.indicators import add_emas
from strategy.signals import add_signals
from strategy.options import otm_strike, option_price_for_position
from config import (CONTRACTS, RISK_FREE_RATE, FORCE_CLOSE,
                    TAKE_PROFIT, STOP_LOSS, BACKTEST_START)

EST = pytz.timezone("US/Eastern")
FORCE_CLOSE_TIME = pd.Timestamp(f"2000-01-01 {FORCE_CLOSE}").time()
UNDERLYING = "QQQ"


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _price(row, strike, option_type, use_real=True):
    if use_real:
        real = get_real_option_price(UNDERLYING, row.name, option_type, strike)
        if real is not None:
            return real, True
    from strategy.options import option_price_for_position
    return option_price_for_position(row, strike, option_type, RISK_FREE_RATE), False


def _close(pos, ts, row, price, reason, real):
    pnl = (price - pos["entry_price"]) * 100 * CONTRACTS
    entry_real = pos.get("entry_real", False)
    pricing = "Real" if (entry_real and real) else ("BS" if (not entry_real and not real) else "Mixed")
    return {"date": pos["entry_time"].date(), "direction": pos["type"].upper(),
            "entry_price": pos["entry_price"], "exit_price": round(price, 4),
            "exit_reason": reason, "pnl": round(pnl, 2), "pricing": pricing}


def metrics(log):
    if log.empty or len(log) < 3:
        return None
    pnl = log["pnl"]
    total   = pnl.sum()
    wins    = pnl[pnl > 0].sum()
    losses  = abs(pnl[pnl < 0].sum())
    pf      = wins / losses if losses > 0 else float("inf")
    cum     = pnl.cumsum()
    max_dd  = (cum - cum.cummax()).min()
    calmar  = total / abs(max_dd) if max_dd != 0 else 0.0
    daily   = log.groupby("date")["pnl"].sum()
    sharpe  = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    return {
        "trades":   len(log),
        "win_pct":  round((pnl > 0).mean() * 100, 1),
        "total_pnl": round(total, 2),
        "profit_factor": round(pf, 2),
        "max_dd":   round(max_dd, 2),
        "calmar":   round(calmar, 3),
        "sharpe":   round(sharpe, 3),
    }


# ── Strategy 1: Trailing stop on bar high / low ────────────────────────────────

def run_trailing_stop(market_df):
    trades, position = [], None
    prev_low = prev_high = None

    for ts, row in market_df.iterrows():
        bar_time = ts.time()
        if position:
            cp, cr = _price(row, position["strike"], position["type"])
            pnl = (cp - position["entry_price"]) * 100 * CONTRACTS
            exited = False
            if pnl >= TAKE_PROFIT:
                trades.append(_close(position, ts, row, cp, "TP", cr)); exited = True
            elif pnl <= STOP_LOSS:
                trades.append(_close(position, ts, row, cp, "SL", cr)); exited = True
            elif bar_time >= FORCE_CLOSE_TIME:
                trades.append(_close(position, ts, row, cp, "Force", cr)); exited = True
            elif position["type"] == "call" and prev_low is not None and row["close"] < prev_low:
                trades.append(_close(position, ts, row, cp, "Trail", cr)); exited = True
            elif position["type"] == "put"  and prev_high is not None and row["close"] > prev_high:
                trades.append(_close(position, ts, row, cp, "Trail", cr)); exited = True
            if exited:
                position = None

        if position is None:
            if row["cross_up"]:
                strike = otm_strike(row["close"], "call")
                ep, er = _price(row, strike, "call")
                position = {"type": "call", "entry_time": ts, "strike": strike,
                            "entry_price": ep, "entry_underlying": row["close"], "entry_real": er}
            elif row["cross_down"]:
                strike = otm_strike(row["close"], "put")
                ep, er = _price(row, strike, "put")
                position = {"type": "put", "entry_time": ts, "strike": strike,
                            "entry_price": ep, "entry_underlying": row["close"], "entry_real": er}

        prev_high = row["high"]
        prev_low  = row["low"]

    if position:
        last_ts, last_row = list(market_df.iterrows())[-1]
        fp, fr = _price(last_row, position["strike"], position["type"])
        trades.append(_close(position, last_ts, last_row, fp, "EOD", fr))

    log = pd.DataFrame(trades) if trades else pd.DataFrame()
    return log[log["pricing"] == "Real"].copy() if not log.empty else log


# ── Strategy 2: Lower high / higher low ───────────────────────────────────────

def run_lower_high(market_df):
    trades, position = [], None
    prev_high = prev_low = None

    for ts, row in market_df.iterrows():
        bar_time = ts.time()
        if position:
            cp, cr = _price(row, position["strike"], position["type"])
            pnl = (cp - position["entry_price"]) * 100 * CONTRACTS
            exited = False
            if pnl >= TAKE_PROFIT:
                trades.append(_close(position, ts, row, cp, "TP", cr)); exited = True
            elif pnl <= STOP_LOSS:
                trades.append(_close(position, ts, row, cp, "SL", cr)); exited = True
            elif bar_time >= FORCE_CLOSE_TIME:
                trades.append(_close(position, ts, row, cp, "Force", cr)); exited = True
            elif position["type"] == "call" and prev_high is not None and row["high"] < prev_high:
                trades.append(_close(position, ts, row, cp, "LowerHigh", cr)); exited = True
            elif position["type"] == "put"  and prev_low  is not None and row["low"]  > prev_low:
                trades.append(_close(position, ts, row, cp, "HigherLow", cr)); exited = True
            if exited:
                position = None

        if position is None:
            if row["cross_up"]:
                strike = otm_strike(row["close"], "call")
                ep, er = _price(row, strike, "call")
                position = {"type": "call", "entry_time": ts, "strike": strike,
                            "entry_price": ep, "entry_underlying": row["close"], "entry_real": er}
            elif row["cross_down"]:
                strike = otm_strike(row["close"], "put")
                ep, er = _price(row, strike, "put")
                position = {"type": "put", "entry_time": ts, "strike": strike,
                            "entry_price": ep, "entry_underlying": row["close"], "entry_real": er}

        prev_high = row["high"]
        prev_low  = row["low"]

    if position:
        last_ts, last_row = list(market_df.iterrows())[-1]
        fp, fr = _price(last_row, position["strike"], position["type"])
        trades.append(_close(position, last_ts, last_row, fp, "EOD", fr))

    log = pd.DataFrame(trades) if trades else pd.DataFrame()
    return log[log["pricing"] == "Real"].copy() if not log.empty else log


# ── Strategy 3: Day absolute high/low (theoretical best-case) ─────────────────

def run_day_extreme(market_df):
    """
    For each day, find the bar after entry with the highest high (CALL) or
    lowest low (PUT) and exit there. Lookahead within the trading day only.
    This is the theoretical best-case ceiling for trend-following exits.
    """
    trades, position = [], None

    for ts, row in market_df.iterrows():
        bar_time = ts.time()
        if position:
            cp, cr = _price(row, position["strike"], position["type"])
            pnl = (cp - position["entry_price"]) * 100 * CONTRACTS
            exited = False
            if pnl >= TAKE_PROFIT:
                trades.append(_close(position, ts, row, cp, "TP", cr)); exited = True
            elif pnl <= STOP_LOSS:
                trades.append(_close(position, ts, row, cp, "SL", cr)); exited = True
            elif bar_time >= FORCE_CLOSE_TIME:
                # At force-close time, look back and find the bar with the best extreme
                day_str = ts.strftime("%Y-%m-%d")
                day_bars = market_df[market_df.index.strftime("%Y-%m-%d") == day_str]
                entry_bars = day_bars[day_bars.index >= position["entry_time"]]
                if not entry_bars.empty:
                    if position["type"] == "call":
                        best_ts = entry_bars["high"].idxmax()
                    else:
                        best_ts = entry_bars["low"].idxmin()
                    best_row = entry_bars.loc[best_ts]
                    best_p, best_r = _price(best_row, position["strike"], position["type"])
                    trades.append(_close(position, best_ts, best_row, best_p, "DayExtreme", best_r))
                else:
                    trades.append(_close(position, ts, row, cp, "Force", cr))
                exited = True
            if exited:
                position = None

        if position is None:
            if row["cross_up"]:
                strike = otm_strike(row["close"], "call")
                ep, er = _price(row, strike, "call")
                position = {"type": "call", "entry_time": ts, "strike": strike,
                            "entry_price": ep, "entry_underlying": row["close"], "entry_real": er}
            elif row["cross_down"]:
                strike = otm_strike(row["close"], "put")
                ep, er = _price(row, strike, "put")
                position = {"type": "put", "entry_time": ts, "strike": strike,
                            "entry_price": ep, "entry_underlying": row["close"], "entry_real": er}

    if position:
        last_ts, last_row = list(market_df.iterrows())[-1]
        fp, fr = _price(last_row, position["strike"], position["type"])
        trades.append(_close(position, last_ts, last_row, fp, "EOD", fr))

    log = pd.DataFrame(trades) if trades else pd.DataFrame()
    return log[log["pricing"] == "Real"].copy() if not log.empty else log


# ── Strategy 4: TP / SL / Force-close only ────────────────────────────────────

def run_tp_sl_only(market_df):
    trades, position = [], None

    for ts, row in market_df.iterrows():
        bar_time = ts.time()
        if position:
            cp, cr = _price(row, position["strike"], position["type"])
            pnl = (cp - position["entry_price"]) * 100 * CONTRACTS
            exited = False
            if pnl >= TAKE_PROFIT:
                trades.append(_close(position, ts, row, cp, "TP", cr)); exited = True
            elif pnl <= STOP_LOSS:
                trades.append(_close(position, ts, row, cp, "SL", cr)); exited = True
            elif bar_time >= FORCE_CLOSE_TIME:
                trades.append(_close(position, ts, row, cp, "Force", cr)); exited = True
            if exited:
                position = None

        if position is None:
            if row["cross_up"]:
                strike = otm_strike(row["close"], "call")
                ep, er = _price(row, strike, "call")
                position = {"type": "call", "entry_time": ts, "strike": strike,
                            "entry_price": ep, "entry_underlying": row["close"], "entry_real": er}
            elif row["cross_down"]:
                strike = otm_strike(row["close"], "put")
                ep, er = _price(row, strike, "put")
                position = {"type": "put", "entry_time": ts, "strike": strike,
                            "entry_price": ep, "entry_underlying": row["close"], "entry_real": er}

    if position:
        last_ts, last_row = list(market_df.iterrows())[-1]
        fp, fr = _price(last_row, position["strike"], position["type"])
        trades.append(_close(position, last_ts, last_row, fp, "EOD", fr))

    log = pd.DataFrame(trades) if trades else pd.DataFrame()
    return log[log["pricing"] == "Real"].copy() if not log.empty else log


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = fetch_qqq_bars()
    df = add_emas(df)
    df = add_signals(df)
    df = df[df["in_market_hours"]]
    if BACKTEST_START:
        df = df[df.index >= pd.Timestamp(BACKTEST_START, tz=df.index.tz)]
        print(f"Backtest range: {df.index[0].date()} to {df.index[-1].date()}")

    strategies = [
        ("1. Trailing Stop (close < prev low / close > prev high)", run_trailing_stop),
        ("2. Lower High / Higher Low (bar high < prev high)",        run_lower_high),
        ("3. Day Absolute High/Low (theoretical best-case)",         run_day_extreme),
        ("4. TP / SL / Force-close only (no trend exit)",           run_tp_sl_only),
    ]

    results = []
    for name, fn in strategies:
        print(f"  Running: {name[:50]}...")
        log = fn(df)
        m = metrics(log)
        if m:
            results.append({"strategy": name, **m})
        else:
            print(f"    -> No qualifying trades.")

    if not results:
        print("No results.")
        return

    results_df = pd.DataFrame(results).sort_values("calmar", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 110)
    print(f"{'Rank':<5} {'Strategy':<55} {'Trades':>6} {'Win%':>6} {'Total P&L':>12} "
          f"{'PF':>6} {'Max DD':>10} {'Calmar':>8} {'Sharpe':>8}")
    print("-" * 110)
    for i, row in results_df.iterrows():
        print(f"{i+1:<5} {row['strategy']:<55} {row['trades']:>6} "
              f"{row['win_pct']:>5.1f}%  ${row['total_pnl']:>10,.2f}  "
              f"{row['profit_factor']:>6.2f}  ${row['max_dd']:>9,.2f}  "
              f"{row['calmar']:>8.3f}  {row['sharpe']:>8.3f}")
    print("=" * 110)

    best = results_df.iloc[0]
    print(f"\nBEST: {best['strategy']}")
    print(f"      {best['trades']} trades | {best['win_pct']}% win rate | "
          f"${best['total_pnl']:,.2f} P&L | Calmar {best['calmar']:.3f}\n")


if __name__ == "__main__":
    main()
