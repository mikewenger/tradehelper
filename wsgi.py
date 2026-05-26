"""
Production entry point — used by gunicorn on Railway.
Initialises data, runs backtests, wires up Basic Auth, then exposes
the Flask server object for gunicorn to serve.

Local dev: use main.py instead (runs with Dash's built-in server).
"""
import os
from flask import request, Response

import pandas as pd
from config import ADMIN_USERNAME, ADMIN_PASSWORD, BACKTEST_START
from data.fetcher import fetch_qqq_bars
from data.options_fetcher import prefetch_options
from strategy.indicators import add_emas
from strategy.signals import add_signals
from backtest.engine import run_backtest
from dashboard.app import create_app
from main import build_contract_comparison, build_timeframe_comparison, _collect_signals

# ── Boot sequence (runs once at gunicorn startup) ──────────────────────────
print("Fetching market data...")
df = fetch_qqq_bars()
df = add_emas(df)               # EMAs computed on ALL hours (matches TOS)
df = add_signals(df)            # market-hours flag + crossover signals added
df = df[df["in_market_hours"]]  # trim to 9:30-16:00 for trading/display

# Apply backtest start filter (EMA already warmed up on full history above)
if BACKTEST_START:
    df = df[df.index >= pd.Timestamp(BACKTEST_START, tz=df.index.tz)]
    print(f"Backtest range: {df.index[0].date()} to {df.index[-1].date()}")

print("Pre-fetching real option chain data from Massive.com...")
prefetch_options(_collect_signals(df))

print("Running primary backtest...")
trade_log = run_backtest(df)
trade_log = trade_log[trade_log["pricing"] == "Real"].copy().reset_index(drop=True)
trade_log["cumulative_pnl"] = trade_log["pnl"].cumsum()

print("Building contract comparison table...")
comparison = build_contract_comparison(df)

print("Building timeframe comparison (5 / 15 / 30-min, Black-Scholes)...")
tf_data = build_timeframe_comparison(df)

print("Creating Dash app...")
dash_app = create_app(df, trade_log, comparison, tf_data)

# ── Flask server (what gunicorn actually serves) ───────────────────────────
server = dash_app.server


# ── HTTP Basic Auth middleware ─────────────────────────────────────────────
def _check_auth(username: str, password: str) -> bool:
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD and password != ""


def _auth_required() -> Response:
    return Response(
        "Authentication required.\n",
        401,
        {"WWW-Authenticate": 'Basic realm="QQQ Backtest Dashboard"'},
    )


@server.before_request
def require_login():
    # Allow Dash internal asset requests through without auth
    if request.path.startswith("/_dash") or request.path.startswith("/assets"):
        return None
    auth = request.authorization
    if not auth or not _check_auth(auth.username, auth.password):
        return _auth_required()
