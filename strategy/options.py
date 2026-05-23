import math
import pytz

from scipy.stats import norm

EST = pytz.timezone("US/Eastern")
MARKET_CLOSE_HOUR = 16  # 4:00 PM EST


def _intraday_T(bar_datetime) -> float:
    """
    0DTE: time to expiry = remaining seconds until 4:00 PM EST today,
    expressed as a fraction of a calendar year.
    Minimum of 1 minute so Black-Scholes doesn't collapse to pure intrinsic.
    """
    close_time = bar_datetime.replace(hour=MARKET_CLOSE_HOUR, minute=0,
                                      second=0, microsecond=0)
    remaining_seconds = max((close_time - bar_datetime).total_seconds(), 60)
    return remaining_seconds / (365 * 24 * 3600)


def otm_strike(price: float, option_type: str) -> float:
    if option_type == "call":
        return float(math.ceil(price) + 1)
    else:
        return float(math.floor(price) - 1)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    if T <= 0 or sigma <= 0:
        # intrinsic value only
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def option_price_at_bar(row, option_type: str, r: float) -> float:
    """Price a new 1-OTM 0DTE option at entry (T = remaining time today)."""
    S = row["close"]
    K = otm_strike(S, option_type)
    sigma = row["hist_vol"] if row["hist_vol"] > 0 else 0.20
    T = _intraday_T(row.name)
    return bs_price(S, K, T, r, sigma, option_type)


def option_price_for_position(row, strike: float, option_type: str, r: float) -> float:
    """Price an open position's fixed strike 0DTE option (T = remaining time today)."""
    S = row["close"]
    sigma = row["hist_vol"] if row["hist_vol"] > 0 else 0.20
    T = _intraday_T(row.name)
    return bs_price(S, strike, T, r, sigma, option_type)
