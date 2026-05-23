import pandas as pd
from data.fetcher import fetch_qqq_bars
from strategy.indicators import add_emas
from strategy.signals import add_signals
from backtest.engine import run_backtest
from dashboard.app import create_app

CONTRACT_SIZES = [10, 20, 30, 40, 50]


def build_contract_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n in CONTRACT_SIZES:
        log = run_backtest(df, contracts=n)
        for direction in ["CALL", "PUT", "COMBINED"]:
            if direction == "COMBINED":
                subset = log
            else:
                subset = log[log["direction"] == direction]

            if subset.empty:
                rows.append({
                    "contracts": n, "direction": direction,
                    "trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
                    "avg_pnl": 0.0, "best_trade": 0.0, "worst_trade": 0.0,
                })
            else:
                pnl = subset["pnl"]
                rows.append({
                    "contracts": n,
                    "direction": direction,
                    "trades": len(subset),
                    "win_rate": round((pnl > 0).mean() * 100, 1),
                    "total_pnl": round(pnl.sum(), 2),
                    "avg_pnl": round(pnl.mean(), 2),
                    "best_trade": round(pnl.max(), 2),
                    "worst_trade": round(pnl.min(), 2),
                })
    return pd.DataFrame(rows)


def main():
    df = fetch_qqq_bars()
    df = add_emas(df)
    df = add_signals(df)

    # Main backtest (current contract size from config)
    trade_log = run_backtest(df)
    total = len(trade_log)
    if total:
        wins = len(trade_log[trade_log["pnl"] > 0])
        print(f"\nBacktest complete: {total} trades | "
              f"Win rate: {wins/total*100:.1f}% | "
              f"Total P&L: ${trade_log['pnl'].sum():,.2f}")
    else:
        print("\nNo trades generated.")

    # Contract comparison table
    print("Building contract comparison table...")
    comparison = build_contract_comparison(df)

    app = create_app(df, trade_log, comparison)
    print("\nDashboard running at http://localhost:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
