import os
from dotenv import load_dotenv

load_dotenv()

MASSIVE_API_KEY  = os.getenv("MASSIVE_API_KEY", "")
ADMIN_USERNAME   = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD   = os.getenv("ADMIN_PASSWORD", "")
MASSIVE_BASE_URL = "https://api.massive.com"

TICKER = "QQQ"
CONTRACTS = 20
RISK_FREE_RATE = 0.05
VOL_WINDOW = 20  # bars for rolling historical volatility

MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"
FORCE_CLOSE = "15:55"

TAKE_PROFIT = 1500.0  # close trade when P&L reaches +$1500  [optimizer #2 by Calmar, real prices]
STOP_LOSS   = -500.0  # close trade when P&L reaches -$500   [optimizer #2 by Calmar, real prices]

BACKTEST_START = "2026-01-01"  # only trade signals on/after this date (EMA still uses full history)
