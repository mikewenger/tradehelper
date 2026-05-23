import pandas as pd
import pytz

from config import CONTRACTS as DEFAULT_CONTRACTS, RISK_FREE_RATE, FORCE_CLOSE, TAKE_PROFIT, STOP_LOSS
from strategy.options import otm_strike, option_price_at_bar, option_price_for_position

EST = pytz.timezone("US/Eastern")
FORCE_CLOSE_TIME = pd.Timestamp(f"2000-01-01 {FORCE_CLOSE}").time()


def run_backtest(df: pd.DataFrame, contracts: int = None) -> pd.DataFrame:
    n_contracts = contracts if contracts is not None else DEFAULT_CONTRACTS
    trades = []
    position = None

    market_df = df[df["in_market_hours"]].copy()

    for ts, row in market_df.iterrows():
        bar_time = ts.time()

        if position:
            current_price = option_price_for_position(
                row, position["strike"], position["type"], RISK_FREE_RATE
            )
            current_pnl = (current_price - position["entry_price"]) * 100 * n_contracts

            if current_pnl >= TAKE_PROFIT:
                trades.append(_close_trade(position, ts, row, current_price, "TP", n_contracts))
                position = None
                continue

            if current_pnl <= STOP_LOSS:
                trades.append(_close_trade(position, ts, row, current_price, "SL", n_contracts))
                position = None
                continue

            if bar_time >= FORCE_CLOSE_TIME:
                trades.append(_close_trade(position, ts, row, current_price, "Force", n_contracts))
                position = None
                continue

        cross_up = row["cross_up"]
        cross_down = row["cross_down"]

        if cross_up:
            if position and position["type"] == "put":
                current_price = option_price_for_position(
                    row, position["strike"], position["type"], RISK_FREE_RATE
                )
                trades.append(_close_trade(position, ts, row, current_price, "Cross", n_contracts))
                position = None
            if position is None:
                price = option_price_at_bar(row, "call", RISK_FREE_RATE)
                position = {
                    "type": "call",
                    "entry_time": ts,
                    "strike": otm_strike(row["close"], "call"),
                    "entry_price": price,
                    "entry_underlying": row["close"],
                }

        elif cross_down:
            if position and position["type"] == "call":
                current_price = option_price_for_position(
                    row, position["strike"], position["type"], RISK_FREE_RATE
                )
                trades.append(_close_trade(position, ts, row, current_price, "Cross", n_contracts))
                position = None
            if position is None:
                price = option_price_at_bar(row, "put", RISK_FREE_RATE)
                position = {
                    "type": "put",
                    "entry_time": ts,
                    "strike": otm_strike(row["close"], "put"),
                    "entry_price": price,
                    "entry_underlying": row["close"],
                }

    if position:
        last_ts, last_row = list(market_df.iterrows())[-1]
        final_price = option_price_for_position(
            last_row, position["strike"], position["type"], RISK_FREE_RATE
        )
        trades.append(_close_trade(position, last_ts, last_row, final_price, "EOD", n_contracts))

    if not trades:
        return pd.DataFrame(columns=[
            "date", "direction", "strike", "entry_time", "entry_underlying",
            "entry_price", "exit_time", "exit_underlying", "exit_price",
            "exit_reason", "contracts", "pnl", "cumulative_pnl"
        ])

    trade_df = pd.DataFrame(trades)
    trade_df["cumulative_pnl"] = trade_df["pnl"].cumsum()
    return trade_df


def _close_trade(position: dict, exit_ts, exit_row, exit_price: float,
                 reason: str, n_contracts: int) -> dict:
    pnl = (exit_price - position["entry_price"]) * 100 * n_contracts
    return {
        "date": position["entry_time"].date(),
        "direction": position["type"].upper(),
        "strike": position["strike"],
        "entry_time": position["entry_time"],
        "entry_underlying": position["entry_underlying"],
        "entry_price": round(position["entry_price"], 4),
        "exit_time": exit_ts,
        "exit_underlying": exit_row["close"],
        "exit_price": round(exit_price, 4),
        "exit_reason": reason,
        "contracts": n_contracts,
        "pnl": round(pnl, 2),
    }
