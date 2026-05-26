"""
Fetches real 15-min option bar data from Massive.com for QQQ 0DTE options.

Option ticker format: O:{underlying}{YYMMDD}{C|P}{strike*1000 zero-padded to 8 digits}
Example: O:QQQ260521C00711000  ->  QQQ Call $711, expiring 2026-05-21

Caching strategy (two layers):
  1. In-memory dict (_cache) — fastest, lasts for the process lifetime.
  2. On-disk parquet (OPTIONS_CACHE_FILE) — persists across restarts so each
     contract is fetched from Massive.com at most ONCE EVER.
"""
import time
import threading
import requests
import pandas as pd
import pytz
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import MASSIVE_API_KEY, MASSIVE_BASE_URL

EST = pytz.timezone("US/Eastern")

OPTIONS_CACHE_FILE = Path(__file__).parent / "options_cache.parquet"

# In-memory cache: option_ticker -> pd.Series(datetime -> close price)
_cache: dict[str, pd.Series] = {}
_disk_cache_loaded = False
_cache_lock = threading.Lock()   # protects _cache + disk writes


# ── Disk-cache helpers ─────────────────────────────────────────────────────────

def _load_disk_cache() -> None:
    """Load the on-disk parquet into _cache (called once at first use)."""
    global _disk_cache_loaded
    if _disk_cache_loaded:
        return
    _disk_cache_loaded = True
    if not OPTIONS_CACHE_FILE.exists():
        return
    try:
        df = pd.read_parquet(OPTIONS_CACHE_FILE)
        for ticker, grp in df.groupby("ticker"):
            series = grp.set_index("datetime")["close"]
            if hasattr(series.index, "tz") and series.index.tz is None:
                series.index = series.index.tz_localize("UTC").tz_convert(EST)
            else:
                series.index = series.index.tz_convert(EST)
            _cache[ticker] = series
        print(f"  [options] Loaded {len(_cache)} contracts from disk cache.")
    except Exception as e:
        print(f"  [options] Disk cache corrupted — deleting and starting fresh. ({e})")
        try:
            OPTIONS_CACHE_FILE.unlink(missing_ok=True)
        except Exception:
            pass


def _append_to_disk_cache(option_ticker: str, series: pd.Series) -> None:
    """Append one contract's data to the on-disk parquet cache (thread-safe)."""
    if series.empty:
        return
    with _cache_lock:
        try:
            df_new = series.reset_index()
            df_new.columns = ["datetime", "close"]
            df_new["ticker"] = option_ticker

            if OPTIONS_CACHE_FILE.exists():
                try:
                    df_existing = pd.read_parquet(OPTIONS_CACHE_FILE)
                    df_existing = df_existing[df_existing["ticker"] != option_ticker]
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                except Exception:
                    # Existing file corrupt — wipe it and start fresh
                    OPTIONS_CACHE_FILE.unlink(missing_ok=True)
                    df_combined = df_new
            else:
                df_combined = df_new

            df_combined.to_parquet(OPTIONS_CACHE_FILE)
        except Exception as e:
            print(f"  [options] Could not save disk cache: {e}")


# ── Core fetch ─────────────────────────────────────────────────────────────────

def build_option_ticker(underlying: str, expiry_date, option_type: str,
                        strike: float) -> str:
    """Build a Polygon/Massive options ticker, e.g. O:QQQ260521C00711000"""
    date_str   = expiry_date.strftime("%y%m%d")
    cp         = "C" if option_type.lower() == "call" else "P"
    strike_int = int(round(strike * 1000))
    return f"O:{underlying}{date_str}{cp}{strike_int:08d}"


def fetch_option_bars(option_ticker: str, date_str: str,
                      max_retries: int = 5) -> pd.Series:
    """
    Fetch 15-min close prices for an option on a single date.
    Order of lookups: in-memory cache -> disk cache -> Massive.com API.
    Empty Series returned if data is unavailable after all retries.
    """
    _load_disk_cache()

    if option_ticker in _cache:
        return _cache[option_ticker]

    url = (f"{MASSIVE_BASE_URL}/v2/aggs/ticker/{option_ticker}"
           f"/range/15/minute/{date_str}/{date_str}")
    headers = {"Authorization": f"Bearer {MASSIVE_API_KEY}"}
    params  = {"adjusted": "true", "sort": "asc", "limit": 50000}

    results = []
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 429:
                # Honour Retry-After if present, else exponential backoff
                retry_after = resp.headers.get("Retry-After")
                wait = int(retry_after) if retry_after else 60  # per-minute window
                print(f"  [options] rate-limited, waiting {wait}s "
                      f"({option_ticker}) attempt {attempt+1}/{max_retries}...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            results = resp.json().get("results", [])
            break
        except requests.exceptions.HTTPError:
            break
        except Exception as e:
            print(f"  [options] fetch error ({option_ticker}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            break

    if not results:
        _cache[option_ticker] = pd.Series(dtype=float, name="close")
        return _cache[option_ticker]

    df = pd.DataFrame(results)
    df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(EST)
    series = df.set_index("datetime")["c"].rename("close")

    _cache[option_ticker] = series
    _append_to_disk_cache(option_ticker, series)   # persist to disk
    return series


def prefetch_options(signals: list[dict], delay: float = 2.0,
                     max_workers: int = 1, underlying: str = "QQQ") -> None:
    """
    Pre-fetch option bars for all crossover signals before the backtest.
    Contracts already in the disk cache are skipped.
    Sequential with 0.5s between calls to stay within the options endpoint
    rate limit (~2 req/sec).
    """
    _load_disk_cache()

    to_fetch = []
    for sig in signals:
        tk = build_option_ticker(underlying, sig["date"], sig["option_type"], sig["strike"])
        if tk not in _cache:
            to_fetch.append((tk, sig["date"].strftime("%Y-%m-%d")))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for tk, ds in to_fetch:
        if tk not in seen:
            seen.add(tk)
            unique.append((tk, ds))

    if not unique:
        print("  [options] All contracts already cached — no API calls needed.")
        return

    total = len(unique)
    eta   = total * delay
    print(f"  [options] Fetching {total} new contracts (~{eta:.0f}s at {delay}s/call)...")
    for i, (tk, ds) in enumerate(unique, 1):
        print(f"  [{i:3d}/{total}] {tk}        ", end="\r", flush=True)
        fetch_option_bars(tk, ds)
        if i < total:
            time.sleep(delay)

    print(f"  [options] Done. {total} contracts fetched and cached.          ")


def get_real_option_price(underlying: str, bar_dt, option_type: str,
                          strike: float) -> float | None:
    """
    Return the real market close price for a 0DTE option at bar_dt.
    Returns None if unavailable so the caller can fall back to Black-Scholes.
    """
    ticker   = build_option_ticker(underlying, bar_dt.date(), option_type, strike)
    date_str = bar_dt.date().strftime("%Y-%m-%d")
    series   = fetch_option_bars(ticker, date_str)

    if series.empty:
        return None

    available = series[series.index <= bar_dt]
    if available.empty:
        return None

    return float(available.iloc[-1])


def clear_cache() -> None:
    """Clear in-memory cache (disk cache is unaffected)."""
    _cache.clear()
