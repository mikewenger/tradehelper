import pandas as pd
import pytz

from config import CONTRACTS as DEFAULT_CONTRACTS, RISK_FREE_RATE, FORCE_CLOSE, TAKE_PROFIT, STOP_LOSS
from strategy.options import otm_strike, option_price_for_position
from data.options_fetcher import get_real_option_price

EST = pytz.timezone("US/Eastern")
FORCE_CLOSE_TIME = pd.Timestamp(f"2000-01-01 {FORCE_CLOSE}").time()
UNDERLYING = "QQQ"  # default; overridden per-call via underlying= param


def _price(row, strike: float, option_type: str, r: float,
           use_real: bool = True, underlying: str = UNDERLYING) -> tuple[float, bool]:
    """
    Price an open option position at the current bar.
    Returns (price, is_real) — is_real=True means real Massive.com data was used,
    False means Black-Scholes fallback.
    """
    if use_real:
        real = get_real_option_price(underlying, row.name, option_type, strike)
        if real is not None:
            return real, True
    return option_price_for_position(row, strike, option_type, r), False


def run_backtest(df: pd.DataFrame, contracts: int = None,
                 use_real_prices: bool = True,
                 underlying: str = UNDERLYING) -> pd.DataFrame:
    n_contracts = contracts if contracts is not None else DEFAULT_CONTRACTS
    trades   = []
    position = None
    prev_high = None   # previous bar's high — used for lower-high exit on CALLs
    prev_low  = None   # previous bar's low  — used for higher-low exit on PUTs

    market_df = df[df["in_market_hours"]].copy()

    for ts, row in market_df.iterrows():
        bar_time = ts.time()

        # ── Manage open position ──────────────────────────────────────────────
        if position:
            current_price, exit_real = _price(row, position["strike"], position["type"], RISK_FREE_RATE, use_real_prices, underlying)
            current_pnl = (current_price - position["entry_price"]) * 100 * n_contracts

            if current_pnl >= TAKE_PROFIT:
                trades.append(_close_trade(position, ts, row, current_price, "TP", n_contracts, exit_real))
                position = None

            elif current_pnl <= STOP_LOSS:
                trades.append(_close_trade(position, ts, row, current_price, "SL", n_contracts, exit_real))
                position = None

            elif bar_time >= FORCE_CLOSE_TIME:
                trades.append(_close_trade(position, ts, row, current_price, "Force", n_contracts, exit_real))
                position = None

            # Lower high → CALL uptrend stalling; exit
            elif position["type"] == "call" and prev_high is not None and row["high"] < prev_high:
                trades.append(_close_trade(position, ts, row, current_price, "LowerHigh", n_contracts, exit_real))
                position = None

            # Higher low → PUT downtrend stalling; exit
            elif position["type"] == "put" and prev_low is not None and row["low"] > prev_low:
                trades.append(_close_trade(position, ts, row, current_price, "HigherLow", n_contracts, exit_real))
                position = None

        # ── Crossover signals — entry only (no cross-based exits) ─────────────
        if position is None:
            cross_up   = row["cross_up"]
            cross_down = row["cross_down"]

            if cross_up:
                strike = otm_strike(row["close"], "call")
                entry_price, entry_real = _price(row, strike, "call", RISK_FREE_RATE, use_real_prices, underlying)
                position = {
                    "type":              "call",
                    "entry_time":        ts,
                    "strike":            strike,
                    "entry_price":       entry_price,
                    "entry_underlying":  row["close"],
                    "entry_real":        entry_real,
                }

            elif cross_down:
                strike = otm_strike(row["close"], "put")
                entry_price, entry_real = _price(row, strike, "put", RISK_FREE_RATE, use_real_prices, underlying)
                position = {
                    "type":              "put",
                    "entry_time":        ts,
                    "strike":            strike,
                    "entry_price":       entry_price,
                    "entry_underlying":  row["close"],
                    "entry_real":        entry_real,
                }

        # ── Track bar high/low for next bar's trend-reversal check ───────────
        prev_high = row["high"]
        prev_low  = row["low"]

    # ── Close any position still open at end of data ──────────────────────────
    if position:
        last_ts, last_row = list(market_df.iterrows())[-1]
        final_price, exit_real = _price(last_row, position["strike"], position["type"], RISK_FREE_RATE, use_real_prices, underlying)
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
