"""
SPY EMA 8/21 Crossover — 0DTE Options Backtest Dashboard
Identical strategy to QQQ but runs on port 8051.
"""
import pandas as pd
from data.fetcher import fetch_bars
from data.options_fetcher import prefetch_options
from strategy.indicators import add_emas
from strategy.signals import add_signals
from strategy.options import otm_strike
from backtest.engine import run_backtest
from dashboard.app import create_app
from config import BACKTEST_START

TICKER          = "SPY"
PORT            = 8051
CONTRACT_SIZES  = [10, 20, 30, 40, 50]
TF_COMPARE_MINS = [5, 15, 30]


def _collect_signals(df: pd.DataFrame) -> list[dict]:
    signals = []
    for ts, row in df[df["in_market_hours"]].iterrows():
        if row["cross_up"]:
            signals.append({"date": ts.date(), "option_type": "call",
                             "strike": otm_strike(row["close"], "call")})
        elif row["cross_down"]:
            signals.append({"date": ts.date(), "option_type": "put",
                             "strike": otm_strike(row["close"], "put")})
    return signals


def build_contract_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n in CONTRACT_SIZES:
        log = run_backtest(df, contracts=n, use_real_prices=True, underlying=TICKER)
        log = log[log["pricing"] == "Real"].copy()
        for direction in ["CALL", "PUT", "COMBINED"]:
            subset = log if direction == "COMBINED" else log[log["direction"] == direction]
            if subset.empty:
                rows.append({"contracts": n, "direction": direction,
                             "trades": 0, "win_rate": 0.0,
                             "tp_hits": 0, "sl_hits": 0, "trend_exits": 0, "force_exits": 0,
                             "total_pnl": 0.0, "avg_pnl": 0.0,
                             "best_trade": 0.0, "worst_trade": 0.0})
            else:
                pnl = subset["pnl"]
                reasons = subset["exit_reason"] if "exit_reason" in subset.columns else pd.Series(dtype=str)
                rows.append({
                    "contracts":   n,   "direction":   direction,
                    "trades":      len(subset),
                    "win_rate":    round((pnl > 0).mean() * 100, 1),
                    "tp_hits":     int((reasons == "TP").sum()),
                    "sl_hits":     int((reasons == "SL").sum()),
                    "trend_exits": int(reasons.isin(["LowerHigh", "HigherLow"]).sum()),
                    "force_exits": int(reasons.isin(["Force", "EOD"]).sum()),
                    "total_pnl":   round(pnl.sum(), 2),
                    "avg_pnl":     round(pnl.mean(), 2),
                    "best_trade":  round(pnl.max(), 2),
                    "worst_trade": round(pnl.min(), 2),
                })
    return pd.DataFrame(rows)


def build_timeframe_comparison(df_15: pd.DataFrame) -> dict:
    from optimize_timeframe import fetch_bars as _fetch_bars
    from optimize_timeframe import add_emas as _add_emas_tf

    results = {}
    for tf in TF_COMPARE_MINS:
        print(f"  Building {tf}-min comparison backtest...")
        try:
            if tf == 15:
                df = df_15.copy()
            else:
                df = _fetch_bars(tf, ticker=TICKER)
                df = _add_emas_tf(df, tf)
                df = add_signals(df)
                df = df[df["in_market_hours"]]
                if BACKTEST_START:
                    df = df[df.index >= pd.Timestamp(BACKTEST_START, tz=df.index.tz)]

            tf_signals = _collect_signals(df)
            print(f"  [{tf}-min] Prefetching real prices for {len(tf_signals)} signals...")
            prefetch_options(tf_signals, underlying=TICKER)

            log = run_backtest(df, use_real_prices=True, underlying=TICKER)
            log = log[log["pricing"] == "Real"].copy().reset_index(drop=True)
            if log.empty:
                continue
            log["cumulative_pnl"] = log["pnl"].cumsum()

            pnl     = log["pnl"]
            cum     = pnl.cumsum()
            dd      = (cum - cum.cummax()).min()
            calmar  = round(pnl.sum() / abs(dd), 3) if dd != 0 else 0
            reasons = log["exit_reason"] if "exit_reason" in log.columns else pd.Series(dtype=str)
            results[tf] = {
                "log":          log,
                "trades":       len(log),
                "win_rate":     round((pnl > 0).mean() * 100, 1),
                "total_pnl":    round(pnl.sum(), 2),
                "avg_pnl":      round(pnl.mean(), 2),
                "max_dd":       round(dd, 2),
                "calmar":       calmar,
                "tp_hits":      int((reasons == "TP").sum()),
                "sl_hits":      int((reasons == "SL").sum()),
                "trend_exits":  int(reasons.isin(["LowerHigh", "HigherLow"]).sum()),
                "force_exits":  int(reasons.isin(["Force", "EOD"]).sum()),
            }
        except Exception as e:
            print(f"  [{tf}-min] ERROR: {e}")
    return results


def main():
    df = fetch_bars(TICKER)
    df = add_emas(df)
    df = add_signals(df)
    df = df[df["in_market_hours"]]

    if BACKTEST_START:
        df = df[df.index >= pd.Timestamp(BACKTEST_START, tz=df.index.tz)]
        print(f"Backtest range: {df.index[0].date()} to {df.index[-1].date()}")

    print(f"Pre-fetching real {TICKER} option chain data from Massive.com...")
    signals = _collect_signals(df)
    prefetch_options(signals, underlying=TICKER)

    print("Running primary backtest...")
    trade_log = run_backtest(df, use_real_prices=True, underlying=TICKER)
    trade_log = trade_log[trade_log["pricing"] == "Real"].copy().reset_index(drop=True)
    trade_log["cumulative_pnl"] = trade_log["pnl"].cumsum()
    total = len(trade_log)
    if total:
        wins = len(trade_log[trade_log["pnl"] > 0])
        print(f"Backtest complete: {total} real-priced trades | "
              f"Win rate: {wins/total*100:.1f}% | "
              f"Total P&L: ${trade_log['pnl'].sum():,.2f}")
    else:
        print("No trades generated.")

    print("Building contract comparison table...")
    comparison = build_contract_comparison(df)

    print("Building timeframe comparison (5 / 15 / 30-min)...")
    tf_data = build_timeframe_comparison(df)

    app = create_app(df, trade_log, comparison, tf_data, ticker=TICKER)

    print(f"\n{TICKER} Dashboard running at http://localhost:{PORT}")
    app.run(debug=False, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
