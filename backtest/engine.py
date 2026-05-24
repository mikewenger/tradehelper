import pandas as pd
import pytz

from config import CONTRACTS as DEFAULT_CONTRACTS, RISK_FREE_RATE, FORCE_CLOSE, TAKE_PROFIT, STOP_LOSS
from strategy.options import otm_strike, option_price_for_position
from data.options_fetcher import get_real_option_price

EST = pytz.timezone("US/Eastern")
FORCE_CLOSE_TIME = pd.Timestamp(f"2000-01-01 {FORCE_CLOSE}").time()
UNDERLYING = "QQQ"


def _price(row, strike: float, option_type: str, r: float,
           use_real: bool = True) -> tuple[float, bool]:
    """
    Price an open option position at the current bar.
    Returns (price, is_real) — is_real=True means real Massive.com data was used,
    False means Black-Scholes fallback.
    """
    if use_real:
        real = get_real_option_price(UNDERLYING, row.name, option_type, strike)
        if real is not None:
            return real, True
    return option_price_for_position(row, strike, option_type, r), False


def run_backtest(df: pd.DataFrame, contracts: int = None,
                 use_real_prices: bool = True) -> pd.DataFrame:
    n_contracts = contracts if contracts is not None else DEFAULT_CONTRACTS
    trades   = []
    position = None

    market_df = df[df["in_market_hours"]].copy()

    for ts, row in market_df.iterrows():
        bar_time = ts.time()

        # ── Manage open position ──────────────────────────────────────────────
        if position:
            current_price, exit_real = _price(row, position["strike"], position["type"], RISK_FREE_RATE, use_real_prices)
            current_pnl = (current_price - position["entry_price"]) * 100 * n_contracts

            if current_pnl >= TAKE_PROFIT:
                trades.append(_close_trade(position, ts, row, current_price, "TP", n_contracts, exit_real))
                position = None
                continue

            if current_pnl <= STOP_LOSS:
                trades.append(_close_trade(position, ts, row, current_price, "SL", n_contracts, exit_real))
                position = None
                continue

            if bar_time >= FORCE_CLOSE_TIME:
                trades.append(_close_trade(position, ts, row, current_price, "Force", n_contracts, exit_real))
                position = None
                continue

        # ── Crossover signals ─────────────────────────────────────────────────
        cross_up   = row["cross_up"]
        cross_down = row["cross_down"]

        if cross_up:
            if position and position["type"] == "put":
                cp, exit_real = _price(row, position["strike"], position["type"], RISK_FREE_RATE, use_real_prices)
                trades.append(_close_trade(position, ts, row, cp, "Cross", n_contracts, exit_real))
                position = None

            if position is None:
                strike = otm_strike(row["close"], "call")
                entry_price, entry_real = _price(row, strike, "call", RISK_FREE_RATE, use_real_prices)
                position = {
                    "type": "call",
                    "entry_time": ts,
                    "strike": strike,
                    "entry_price": entry_price,
                    "entry_underlying": row["close"],
                    "entry_real": entry_real,
                }

        elif cross_down:
            if position and position["type"] == "call":
                cp, exit_real = _price(row, position["strike"], position["type"], RISK_FREE_RATE, use_real_prices)
                trades.append(_close_trade(position, ts, row, cp, "Cross", n_contracts, exit_real))
                position = None

            if position is None:
                strike = otm_strike(row["close"], "put")
                entry_price, entry_real = _price(row, strike, "put", RISK_FREE_RATE, use_real_prices)
                position = {
                    "type": "put",
                    "entry_time": ts,
                    "strike": strike,
                    "entry_price": entry_price,
                    "entry_underlying": row["close"],
                    "entry_real": entry_real,
                }

    # ── Close any position still open at end of data ──────────────────────────
    if position:
        last_ts, last_row = list(market_df.iterrows())[-1]
        final_price, exit_real = _price(last_row, position["strike"], position["type"], RISK_FREE_RATE, use_real_prices)
        trades.append(_close_trade(position, last_ts, last_row, final_price, "EOD", n_contracts, exit_real))

    if not trades:
        return pd.DataFrame(columns=[
            "date", "direction", "strike", "entry_time", "entry_underlying",
            "entry_price", "exit_time", "exit_underlying", "exit_price",
            "exit_reason", "contracts", "pnl", "pricing", "cumulative_pnl"
        ])

    trade_df = pd.DataFrame(trades)
    trade_df["cumulative_pnl"] = trade_df["pnl"].cumsum()
    return trade_df


def _close_trade(position: dict, exit_ts, exit_row, exit_price: float,
                 reason: str, n_contracts: int, exit_real: bool = False) -> dict:
    pnl = (exit_price - position["entry_price"]) * 100 * n_contracts
    entry_real = position.get("entry_real", False)
    if entry_real and exit_real:
        pricing = "Real"
    elif not entry_real and not exit_real:
        pricing = "BS"
    else:
        pricing = "Mixed"
    return {
        "date":              position["entry_time"].date(),
        "direction":         position["type"].upper(),
        "strike":            position["strike"],
        "entry_time":        position["entry_time"],
        "entry_underlying":  position["entry_underlying"],
        "entry_price":       round(position["entry_price"], 4),
        "exit_time":         exit_ts,
        "exit_underlying":   exit_row["close"],
        "exit_price":        round(exit_price, 4),
        "exit_reason":       reason,
        "contracts":         n_contracts,
        "pnl":               round(pnl, 2),
        "pricing":           pricing,
    }
