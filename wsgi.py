"""
Production entry point — used by gunicorn on Railway.
Initialises data, runs backtests, wires up Basic Auth, then exposes
the Flask server object for gunicorn to serve.

Local dev: use main.py instead (runs with Dash's built-in server).
"""
import os
from flask import request, Response

from config import ADMIN_USERNAME, ADMIN_PASSWORD
from data.fetcher import fetch_qqq_bars
from strategy.indicators import add_emas
from strategy.signals import add_signals
from backtest.engine import run_backtest
from dashboard.app import create_app
from main import build_contract_comparison

# ── Boot sequence (runs once at gunicorn startup) ──────────────────────────
print("Fetching market data...")
df = fetch_qqq_bars()
df = add_emas(df)
df = add_signals(df)

print("Running primary backtest...")
trade_log = run_backtest(df)

print("Building contract comparison table...")
comparison = build_contract_comparison(df)

print("Creating Dash app...")
dash_app = create_app(df, trade_log, comparison)

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
