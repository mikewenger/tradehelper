import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc
from config import TAKE_PROFIT, STOP_LOSS, MARKET_OPEN, MARKET_CLOSE, FORCE_CLOSE


def create_app(df: pd.DataFrame, trade_log: pd.DataFrame,
               comparison: pd.DataFrame,
               tf_data: dict | None = None,
               ticker: str = "QQQ") -> dash.Dash:
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

    market_bars = df[df["in_market_hours"]].index
    date_range_str = (
        f"{market_bars.min().strftime('%b %d, %Y')} "
        f"– {market_bars.max().strftime('%b %d, %Y')}"
    )
    # Only show dates where at least one trade fired
    trade_dates = sorted(trade_log["date"].unique(), reverse=True) if not trade_log.empty else []
    date_options = [
        {"label": str(d), "value": str(d)}
        for d in trade_dates
    ]
    default_date = str(trade_dates[0]) if trade_dates else None

    app.layout = dbc.Container([
        dbc.Row(dbc.Col(html.H2(f"{ticker} EMA 8/21 Crossover — 0DTE Options Backtest",
                                className="text-center my-3"))),

        dbc.Tabs([
            # ── TAB 1: Daily view ─────────────────────────────────────────────
            dbc.Tab(label="Daily Backtest", tab_id="tab-daily", children=[
                dbc.Row(id="summary-row", className="my-3"),

                dbc.Row([
                    dbc.Col([
                        html.Label("Select Day:", className="fw-bold"),
                        dcc.Dropdown(
                            id="date-picker",
                            options=date_options,
                            value=default_date,
                            clearable=False,
                            style={"color": "#000"},
                        )
                    ], width=3)
                ], className="mb-2"),

                dbc.Row(dbc.Col(dcc.Graph(id="price-chart",
                                          style={"height": "500px"}))),
                dbc.Row(dbc.Col(html.H5("Trades", className="mt-4"))),
                dbc.Row(dbc.Col(html.Div(id="trade-table"))),
                dbc.Row(dbc.Col(html.H5("Equity Curve (All Trades)",
                                        className="mt-4"))),
                dbc.Row(dbc.Col(dcc.Graph(id="equity-curve",
                                          style={"height": "300px"}))),
            ]),

            # ── TAB 2: Contract comparison ─────────────────────────────────────
            dbc.Tab(label="Contract Comparison", tab_id="tab-contracts", children=[

                # ── Strategy criteria panel ────────────────────────────────────
                dbc.Row(dbc.Col(html.H4(
                    "Strategy Criteria & Contract Comparison",
                    className="text-center mt-4 mb-3"
                ))),

                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("INSTRUMENT", className="text-muted mb-2 small"),
                        html.P(f"{ticker}  •  0DTE Options  •  1 Strike OTM",
                               className="fw-bold mb-0"),
                    ]), color="dark", outline=True), width=12, lg=4, className="mb-3"),

                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("ENTRY SIGNAL", className="text-muted mb-2 small"),
                        html.P([
                            html.Span("BUY CALL", className="text-success fw-bold"),
                            " — EMA 8 crosses above EMA 21",
                            html.Br(),
                            html.Span("BUY PUT", className="text-danger fw-bold"),
                            " — EMA 8 crosses below EMA 21",
                        ], className="mb-0 small"),
                    ]), color="dark", outline=True), width=12, lg=4, className="mb-3"),

                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("TIMEFRAME", className="text-muted mb-2 small"),
                        html.P([
                            "15-minute bars", html.Br(),
                            f"Market hours: {MARKET_OPEN} – {MARKET_CLOSE} EST", html.Br(),
                            f"Force-close: {FORCE_CLOSE} EST",
                        ], className="mb-0 small"),
                    ]), color="dark", outline=True), width=12, lg=4, className="mb-3"),
                ], className="mb-2"),

                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("PROFIT TARGET", className="text-muted mb-2 small"),
                        html.P(f"+${TAKE_PROFIT:,.0f} per trade",
                               className="fw-bold text-success mb-0 fs-5"),
                    ]), color="success", outline=True), width=12, lg=4, className="mb-3"),

                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("STOP LOSS", className="text-muted mb-2 small"),
                        html.P(f"-${abs(STOP_LOSS):,.0f} per trade",
                               className="fw-bold text-danger mb-0 fs-5"),
                    ]), color="danger", outline=True), width=12, lg=4, className="mb-3"),

                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("MOVING AVERAGES", className="text-muted mb-2 small"),
                        html.P([
                            "Fast: EMA 8 (close, exponential)", html.Br(),
                            "Slow: EMA 21 (close, exponential)", html.Br(),
                            "Computed on all hours (matches TOS)",
                        ], className="mb-0 small"),
                    ]), color="dark", outline=True), width=12, lg=4, className="mb-3"),
                ], className="mb-2"),

                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("BACKTEST DATE RANGE", className="text-muted mb-2 small"),
                        html.P(date_range_str,
                               className="fw-bold mb-0 fs-5"),
                    ]), color="secondary", outline=True), width=12, className="mb-3"),
                ], className="mb-2"),

                html.Hr(style={"borderColor": "#444"}),

                # Summary cards: total P&L per contract size
                dbc.Row(id="contract-summary-cards", className="mb-4 mt-2"),

                # Full trade-by-trade breakdown with P&L at each contract size
                dbc.Row(dbc.Col([
                    html.H5("Trade-by-Trade P&L by Contract Size",
                            className="mb-2"),
                    html.Div(id="contract-trade-table"),
                ], width=12), className="mb-4"),
            ]),
            # ── TAB 3: Timeframe Analysis ──────────────────────────────────────
            dbc.Tab(label="Timeframe Analysis", tab_id="tab-tf", children=(
                _build_tf_tab(tf_data) if tf_data else [
                    html.P("Timeframe comparison data not available.",
                           className="text-muted m-4")]
            )),

        ], id="tabs", active_tab="tab-daily"),

    ], fluid=True)

    # ── Callback: daily tab ────────────────────────────────────────────────────
    @app.callback(
        Output("summary-row", "children"),
        Output("price-chart", "figure"),
        Output("trade-table", "children"),
        Output("equity-curve", "figure"),
        Input("date-picker", "value"),
    )
    def update_daily(selected_date):
        # selected_date is "YYYY-MM-DD"
        selected_day = selected_date[:10] if selected_date else None
        day_trades = pd.DataFrame()
        if not trade_log.empty and selected_day:
            day_trades = trade_log[trade_log["date"].astype(str) == selected_day]

        total_trades = len(trade_log)
        wins = len(trade_log[trade_log["pnl"] > 0]) if not trade_log.empty else 0
        win_rate = f"{wins / total_trades * 100:.1f}%" if total_trades else "—"
        total_pnl = f"${trade_log['pnl'].sum():,.2f}" if not trade_log.empty else "$0.00"
        best_day = worst_day = "—"
        if not trade_log.empty:
            daily = trade_log.groupby("date")["pnl"].sum()
            best_day = f"${daily.max():,.2f}"
            worst_day = f"${daily.min():,.2f}"

        summary_cards = [
            _card("Total Trades", str(total_trades)),
            _card("Win Rate", win_rate),
            _card("Total P&L", total_pnl,
                  color="success" if not trade_log.empty and trade_log["pnl"].sum() >= 0
                  else "danger"),
            _card("Best Day", best_day, color="success"),
            _card("Worst Day", worst_day, color="danger"),
        ]

        fig = _build_chart(df, day_trades, selected_day, selected_date, ticker=ticker)
        table = _build_trade_table(day_trades)
        eq_fig = _build_equity(trade_log)
        return summary_cards, fig, table, eq_fig

    # ── Callback: contract comparison tab ──────────────────────────────────────
    @app.callback(
        Output("contract-summary-cards", "children"),
        Output("contract-trade-table", "children"),
        Input("tabs", "active_tab"),
    )
    def update_comparison(_):
        from backtest.engine import run_backtest as _run_bt
        sizes = [10, 20, 30, 40, 50]

        # Re-run backtest for each contract size so TP/SL fire at correct price moves
        logs = {}
        for n in sizes:
            log = _run_bt(df, contracts=n, use_real_prices=True)
            log = log[log["pricing"] == "Real"].copy().reset_index(drop=True)
            log["cumulative_pnl"] = log["pnl"].cumsum()
            logs[n] = log

        # Summary cards: total P&L per contract size
        summary_cards = []
        for n in sizes:
            total = logs[n]["pnl"].sum() if not logs[n].empty else 0
            color = "success" if total >= 0 else "danger"
            summary_cards.append(
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.P(f"{n} Contracts", className="text-muted small mb-1"),
                    html.H5(f"${total:,.0f}",
                            className=f"fw-bold text-{'success' if total >= 0 else 'danger'} mb-0"),
                ]), color=color, outline=True), width="auto", className="mb-2")
            )

        trade_tbl = _build_contract_trade_table(logs, sizes)
        return summary_cards, trade_tbl

    return app


# ── Helpers ────────────────────────────────────────────────────────────────────

def _card(title, value, color="secondary"):
    return dbc.Col(dbc.Card([
        dbc.CardBody([
            html.P(title, className="card-title mb-1 small text-muted"),
            html.H5(value, className="card-text fw-bold"),
        ])
    ], color=color, outline=True), width="auto")


def _build_chart(df: pd.DataFrame, day_trades: pd.DataFrame,
                 selected_date: str, selected_datetime: str = None,
                 ticker: str = "QQQ") -> go.Figure:
    import pytz
    EST = pytz.timezone("US/Eastern")

    if not selected_date:
        return go.Figure()

    day_df = df[df.index.strftime("%Y-%m-%d") == selected_date]
    if day_df.empty:
        return go.Figure()

    # Convert index to naive strings for Plotly (avoids tz serialisation issues)
    x_vals = day_df.index.strftime("%Y-%m-%d %H:%M")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.05)

    fig.add_trace(go.Candlestick(
        x=x_vals, open=day_df["open"], high=day_df["high"],
        low=day_df["low"], close=day_df["close"], name=ticker,
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=x_vals, y=day_df["ema8"], name="EMA 8",
                              line=dict(color="#ff9800", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=day_df["ema21"], name="EMA 21",
                              line=dict(color="#42a5f5", width=1.5)), row=1, col=1)

    exit_colors = {"TP": "#00e676", "SL": "#ff1744",
                   "LowerHigh": "#ffffff", "HigherLow": "#ffffff",
                   "Force": "#ffab40", "EOD": "#ce93d8"}

    if not day_trades.empty:
        for _, trade in day_trades.iterrows():
            color = "#26a69a" if trade["direction"] == "CALL" else "#ef5350"
            symbol_entry = "triangle-up" if trade["direction"] == "CALL" else "triangle-down"
            exit_color = exit_colors.get(trade.get("exit_reason", ""), "#ffffff")
            pnl_str = f"${trade['pnl']:,.2f}" if isinstance(trade["pnl"], (int, float)) else trade["pnl"]
            entry_x = trade["entry_time"].strftime("%Y-%m-%d %H:%M")
            exit_x  = trade["exit_time"].strftime("%Y-%m-%d %H:%M")

            fig.add_trace(go.Scatter(
                x=[entry_x], y=[trade["entry_underlying"]],
                mode="markers",
                marker=dict(symbol=symbol_entry, size=14, color=color,
                            line=dict(color="#ffffff", width=1)),
                hovertemplate=(f"{trade['direction']} Entry<br>"
                               f"@{entry_x}<br>"
                               f"Strike: {trade['strike']}<br>"
                               f"Premium: ${trade['entry_price']:.4f}<extra></extra>"),
                showlegend=False,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[exit_x], y=[trade["exit_underlying"]],
                mode="markers",
                marker=dict(symbol="x-thin", size=14, color=exit_color,
                            line=dict(color=exit_color, width=2)),
                hovertemplate=(f"Exit ({trade.get('exit_reason', '')}) {pnl_str}<br>"
                               f"@{exit_x}<extra></extra>"),
                showlegend=False,
            ), row=1, col=1)

    bar_colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(day_df["close"], day_df["open"])]
    fig.add_trace(go.Bar(x=x_vals, y=day_df["volume"], name="Volume",
                          marker_color=bar_colors, opacity=0.6), row=2, col=1)

    # Selected bar: yellow marker line + zoom to ±1 hr window
    sel_str = selected_datetime  # already "YYYY-MM-DD HH:MM"
    x_range = [x_vals[0], x_vals[-1]]  # default: full day

    if sel_str and sel_str in x_vals:
        idx = list(x_vals).index(sel_str)
        lo = max(0, idx - 4)          # 4 bars = 1 hour back
        hi = min(len(x_vals) - 1, idx + 4)
        x_range = [x_vals[lo], x_vals[hi]]

        # Vertical dashed line using add_shape (works in all Plotly versions)
        fig.add_shape(type="line",
                      x0=sel_str, x1=sel_str,
                      y0=0, y1=1, xref="x", yref="paper",
                      line=dict(color="#ffd600", width=2, dash="dash"))

    fig.update_layout(
        title=dict(text=f"{ticker}  ·  {selected_datetime or selected_date}",
                   font=dict(size=13, color="#aaa")),
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=20, t=45, b=20),
        legend=dict(orientation="h", y=1.08),
        xaxis=dict(range=x_range, tickformat="%H:%M", tickangle=-45),
        xaxis2=dict(range=x_range, tickformat="%H:%M", tickangle=-45),
    )
    return fig


def _build_trade_table(day_trades: pd.DataFrame) -> html.Div:
    if day_trades.empty:
        return html.P("No trades on this day.", className="text-muted")

    cols = ["direction", "strike", "entry_time", "entry_price",
            "exit_time", "exit_price", "exit_reason", "contracts", "pnl", "pricing"]
    cols = [c for c in cols if c in day_trades.columns]
    display = day_trades[cols].copy()
    display["entry_time"] = display["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
    display["exit_time"]  = display["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
    display["pnl"]        = display["pnl"].apply(lambda x: f"${x:,.2f}")

    return dash_table.DataTable(
        data=display.to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in display.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#303030", "color": "white", "fontWeight": "bold"},
        style_cell={"backgroundColor": "#1a1a1a", "color": "white", "textAlign": "center"},
        style_data_conditional=[
            # P&L colors
            {"if": {"filter_query": '{pnl} contains "-"'}, "color": "#ef5350"},
            {"if": {"filter_query": '{pnl} contains "$" && !({pnl} contains "-")'}, "color": "#26a69a"},
            # Pricing source: BS rows get an amber background, Mixed gets orange
            {"if": {"filter_query": '{pricing} = "BS"'},    "backgroundColor": "#3a2800", "color": "#ffb74d"},
            {"if": {"filter_query": '{pricing} = "Mixed"'}, "backgroundColor": "#3a1e00", "color": "#ff9800"},
        ],
    )


def _build_equity(trade_log: pd.DataFrame) -> go.Figure:
    if trade_log.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trade_log["exit_time"], y=trade_log["cumulative_pnl"],
        mode="lines+markers", name="Cumulative P&L",
        line=dict(color="#42a5f5", width=2),
        fill="tozeroy", fillcolor="rgba(66,165,245,0.1)",
    ))
    fig.add_hline(y=0, line_color="gray", line_dash="dash")
    fig.update_layout(template="plotly_dark",
                      margin=dict(l=40, r=20, t=20, b=20),
                      yaxis_tickprefix="$", yaxis_tickformat=",.0f")
    return fig


def _build_contract_trade_table(logs: dict, sizes: list) -> html.Div:
    """
    One row per trade day. Each contract-size column shows the actual daily P&L
    from a properly re-run backtest at that size (so TP/SL fire at the right price moves).
    """
    if not logs or all(v.empty for v in logs.values()):
        return html.P("No trades.", className="text-muted")

    # Collect all trade dates across all contract sizes
    all_dates = sorted(set(
        str(d) for n, log in logs.items() for d in log["date"].unique()
    ))

    rows = []
    for date_str in all_dates:
        row = {"date": date_str}
        for n in sizes:
            log = logs[n]
            day = log[log["date"].astype(str) == date_str]
            row[f"trades_{n}"] = len(day)
            row[f"pnl_{n}c"]   = round(day["pnl"].sum(), 0) if not day.empty else 0
        rows.append(row)

    daily = pd.DataFrame(rows)

    # Totals row
    totals = {"date": "TOTAL"}
    for n in sizes:
        totals[f"trades_{n}"] = daily[f"trades_{n}"].sum()
        totals[f"pnl_{n}c"]   = round(daily[f"pnl_{n}c"].sum(), 0)
    daily = pd.concat([daily, pd.DataFrame([totals])], ignore_index=True)

    # Format P&L as currency
    for n in sizes:
        daily[f"pnl_{n}c"] = daily[f"pnl_{n}c"].apply(
            lambda x: f"${int(x):,}" if isinstance(x, (int, float)) else x)

    # Use trades column from 20-contract run as the reference trade count
    daily["trades"] = daily["trades_20"]
    for n in sizes:
        daily = daily.drop(columns=[f"trades_{n}"])

    columns = (
        [{"name": h, "id": i} for h, i in [("Date", "date"), ("# Trades", "trades")]] +
        [{"name": f"{n} Contracts", "id": f"pnl_{n}c"} for n in sizes]
    )

    pnl_styles = []
    for n in sizes:
        col = f"pnl_{n}c"
        pnl_styles += [
            {"if": {"filter_query": f'{{{col}}} contains "-"', "column_id": col},
             "color": "#ef5350"},
            {"if": {"filter_query": f'{{{col}}} contains "$" && !({{{col}}} contains "-")',
                    "column_id": col}, "color": "#26a69a"},
        ]

    return dash_table.DataTable(
        data=daily.to_dict("records"),
        columns=columns,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#303030", "color": "white",
                      "fontWeight": "bold", "textAlign": "center"},
        style_cell={"backgroundColor": "#1a1a1a", "color": "white",
                    "textAlign": "center", "padding": "8px 12px",
                    "fontSize": "14px"},
        style_data_conditional=pnl_styles + [
            {"if": {"filter_query": '{date} = "TOTAL"'},
             "fontWeight": "bold", "backgroundColor": "#2a2a2a", "fontSize": "15px"},
        ],
        page_size=120,
        sort_action="native",
    )


def _build_comparison_table(subset: pd.DataFrame, pnl_color: str) -> html.Div:
    if subset.empty:
        return html.P("No data.", className="text-muted")

    cols_needed = ["contracts", "trades", "win_rate",
                   "tp_hits", "sl_hits", "trend_exits", "force_exits",
                   "total_pnl", "avg_pnl", "best_trade", "worst_trade"]
    cols_needed = [c for c in cols_needed if c in subset.columns]
    display = subset[cols_needed].copy()

    # Format percentages and currency
    display["win_rate"]    = display["win_rate"].apply(lambda x: f"{x:.1f}%")
    display["total_pnl"]   = display["total_pnl"].apply(lambda x: f"${x:,.2f}")
    display["avg_pnl"]     = display["avg_pnl"].apply(lambda x: f"${x:,.2f}")
    display["best_trade"]  = display["best_trade"].apply(lambda x: f"${x:,.2f}")
    display["worst_trade"] = display["worst_trade"].apply(lambda x: f"${x:,.2f}")

    col_names = {
        "contracts":   "Contracts",
        "trades":      "Total Trades",
        "win_rate":    "Win Rate",
        "tp_hits":     f"Take Profit Hits (+${TAKE_PROFIT:,.0f})",
        "sl_hits":     f"Stop Loss Hits (-${abs(STOP_LOSS):,.0f})",
        "trend_exits": "Trend Reversal Exits",
        "force_exits": "Force / EOD",
        "total_pnl":   "Total P&L",
        "avg_pnl":     "Avg P&L / Trade",
        "best_trade":  "Best Trade",
        "worst_trade": "Worst Trade",
    }

    return dash_table.DataTable(
        data=display.to_dict("records"),
        columns=[{"name": col_names.get(c, c), "id": c} for c in display.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#303030", "color": "white",
                      "fontWeight": "bold", "textAlign": "center",
                      "whiteSpace": "normal", "height": "auto"},
        style_cell={"backgroundColor": "#1a1a1a", "color": "white",
                    "textAlign": "center", "padding": "10px",
                    "fontSize": "14px", "minWidth": "90px"},
        style_data_conditional=[
            {"if": {"filter_query": '{total_pnl} contains "-"',
                    "column_id": "total_pnl"}, "color": "#ef5350"},
            {"if": {"filter_query": '{total_pnl} contains "$" && !({total_pnl} contains "-")',
                    "column_id": "total_pnl"}, "color": "#26a69a"},
            {"if": {"filter_query": '{avg_pnl} contains "-"',
                    "column_id": "avg_pnl"}, "color": "#ef5350"},
            {"if": {"filter_query": '{avg_pnl} contains "$" && !({avg_pnl} contains "-")',
                    "column_id": "avg_pnl"}, "color": "#26a69a"},
            {"if": {"column_id": "best_trade"},  "color": "#26a69a"},
            {"if": {"column_id": "worst_trade"}, "color": "#ef5350"},
            {"if": {"column_id": "tp_hits"},     "color": "#26a69a"},
            {"if": {"column_id": "sl_hits"},     "color": "#ef5350"},
            {"if": {"column_id": "contracts"},   "fontWeight": "bold",
             "color": pnl_color},
        ],
    )


def _build_tf_tab(tf_data: dict) -> list:
    """Build the full Timeframe Analysis tab layout."""
    COLORS = {5: "#ef5350", 15: "#26a69a", 30: "#42a5f5"}
    WINNER = 15

    # ── Winner callout ────────────────────────────────────────────────────────
    w = tf_data.get(WINNER, {})
    winner_card = dbc.Alert([
        html.H4("🏆  15-Minute Bars is the Optimal Timeframe", className="alert-heading mb-2"),
        html.P([
            f"Calmar Ratio {w.get('calmar', '—')}  •  "
            f"{w.get('trades', '—')} trades  •  "
            f"{w.get('win_rate', '—')}% win rate  •  "
            f"Total P&L ${w.get('total_pnl', 0):,.2f}  •  "
            f"Max Drawdown ${abs(w.get('max_dd', 0)):,.2f}",
        ], className="mb-0 small"),
    ], color="success", className="mt-4 mb-4")

    # ── Top metric cards (one per timeframe) ──────────────────────────────────
    metric_cards = dbc.Row([
        dbc.Col(_tf_summary_card(tf, tf_data[tf], COLORS[tf], tf == WINNER), lg=4, className="mb-3")
        for tf in [5, 15, 30] if tf in tf_data
    ], className="mb-2")

    # ── Comparison table ──────────────────────────────────────────────────────
    comp_table = _build_tf_comparison_table(tf_data, COLORS, WINNER)

    # ── Equity curves ─────────────────────────────────────────────────────────
    equity_fig = _build_tf_equity_chart(tf_data, COLORS)

    # ── Bar charts row ────────────────────────────────────────────────────────
    bar_figs = _build_tf_bar_charts(tf_data, COLORS)

    # ── Exit breakdown ────────────────────────────────────────────────────────
    exit_fig = _build_tf_exit_chart(tf_data, COLORS)

    return [
        dbc.Row(dbc.Col(html.H4("Timeframe Analysis — EMA 8/21 Crossover",
                                 className="text-center mt-4 mb-0"))),
        dbc.Row(dbc.Col(html.P(
            f"TP: +${TAKE_PROFIT:,.0f}  |  SL: -${abs(STOP_LOSS):,.0f}  |  "
            "Black-Scholes pricing (equal basis for all timeframes)",
            className="text-center text-muted small mb-0"
        ))),
        dbc.Row(dbc.Col(winner_card)),
        metric_cards,
        dbc.Row(dbc.Col(html.H5("Performance Metrics Comparison", className="mt-2 mb-2"))),
        dbc.Row(dbc.Col(comp_table, className="mb-4")),
        dbc.Row(dbc.Col([
            html.H5("Equity Curves (Cumulative P&L)", className="mt-2 mb-1"),
            html.P("All three timeframes, same date range, same TP/SL rules.",
                   className="text-muted small mb-2"),
            dcc.Graph(figure=equity_fig, style={"height": "380px"}),
        ])),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=bar_figs["calmar"], style={"height": "300px"}), lg=4),
            dbc.Col(dcc.Graph(figure=bar_figs["win_rate"], style={"height": "300px"}), lg=4),
            dbc.Col(dcc.Graph(figure=bar_figs["avg_pnl"], style={"height": "300px"}), lg=4),
        ], className="mt-2"),
        dbc.Row(dbc.Col([
            html.H5("Exit Reason Breakdown", className="mt-2 mb-1"),
            dcc.Graph(figure=exit_fig, style={"height": "300px"}),
        ])),
        dbc.Row(dbc.Col(_tf_why_box(), className="mt-3 mb-4")),
    ]


def _tf_summary_card(tf: int, stats: dict, color: str, is_winner: bool) -> dbc.Card:
    border = "success" if is_winner else "secondary"
    badge = dbc.Badge("⭐ WINNER", color="success", className="ms-2") if is_winner else ""
    return dbc.Card([
        dbc.CardHeader([
            html.H5([f"{tf}-Minute Bars", badge], className="mb-0 d-flex align-items-center")
        ], style={"borderLeft": f"4px solid {color}"}),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([html.P("Calmar Ratio", className="text-muted small mb-1"),
                         html.H4(f"{stats['calmar']:.3f}", className="fw-bold",
                                 style={"color": color})]),
                dbc.Col([html.P("Total P&L", className="text-muted small mb-1"),
                         html.H5(f"${stats['total_pnl']:,.0f}",
                                 className="text-success fw-bold")]),
            ]),
            dbc.Row([
                dbc.Col([html.P("Win Rate", className="text-muted small mb-1"),
                         html.H5(f"{stats['win_rate']}%", className="fw-bold")]),
                dbc.Col([html.P("Trades", className="text-muted small mb-1"),
                         html.H5(str(stats['trades']), className="fw-bold")]),
                dbc.Col([html.P("Max DD", className="text-muted small mb-1"),
                         html.H5(f"${abs(stats['max_dd']):,.0f}",
                                 className="text-danger fw-bold")]),
            ]),
        ]),
    ], color=border, outline=True)


def _build_tf_comparison_table(tf_data: dict, colors: dict, winner: int) -> html.Div:
    metrics = [
        ("Trades",           "trades",       lambda v: str(v)),
        ("Win Rate",         "win_rate",     lambda v: f"{v}%"),
        ("Total P&L",        "total_pnl",    lambda v: f"${v:,.2f}"),
        ("Avg P&L / Trade",  "avg_pnl",      lambda v: f"${v:,.2f}"),
        ("Max Drawdown",     "max_dd",       lambda v: f"${abs(v):,.2f}"),
        ("Calmar Ratio",     "calmar",       lambda v: f"{v:.3f}"),
        ("TP Hits",          "tp_hits",      lambda v: str(v)),
        ("SL Hits",          "sl_hits",      lambda v: str(v)),
        ("Trend Exits",      "trend_exits",  lambda v: str(v)),
        ("Force / EOD",      "force_exits",  lambda v: str(v)),
    ]
    tfs = [tf for tf in [5, 15, 30] if tf in tf_data]

    header = html.Thead(html.Tr([
        html.Th("Metric", style={"width": "200px"}),
        *[html.Th(
            [f"{tf}-Min", dbc.Badge("★ BEST", color="success", className="ms-1") if tf == winner else ""],
            style={"textAlign": "center",
                   "backgroundColor": "#1a3a1a" if tf == winner else "#1a1a1a",
                   "color": colors[tf], "fontWeight": "bold"},
        ) for tf in tfs],
    ]))

    rows = []
    for label, key, fmt in metrics:
        tds = [html.Td(label, style={"fontWeight": "bold", "color": "#aaa"})]
        for tf in tfs:
            val = tf_data[tf].get(key, "—")
            text = fmt(val) if val != "—" else "—"
            # Highlight winner column and color-code specific metrics
            bg = "#1a3a1a" if tf == winner else "#1a1a1a"
            color = "#ddd"
            if key == "calmar":
                color = colors[tf]
            elif key in ("total_pnl", "avg_pnl") and isinstance(val, (int, float)):
                color = "#26a69a" if val >= 0 else "#ef5350"
            elif key == "max_dd":
                color = "#ef5350"
            elif key == "tp_hits":
                color = "#26a69a"
            elif key == "sl_hits":
                color = "#ef5350"
            elif key == "win_rate":
                color = "#ffb74d"
            tds.append(html.Td(text, style={
                "textAlign": "center", "color": color,
                "backgroundColor": bg, "fontWeight": "bold" if tf == winner else "normal",
            }))
        rows.append(html.Tr(tds))

    table = dbc.Table(
        [header, html.Tbody(rows)],
        bordered=True, hover=True, size="sm",
        style={"fontSize": "14px"},
    )
    return html.Div(table, style={"overflowX": "auto"})


def _build_tf_equity_chart(tf_data: dict, colors: dict) -> go.Figure:
    fig = go.Figure()
    for tf in [5, 15, 30]:
        if tf not in tf_data:
            continue
        log = tf_data[tf]["log"]
        if log.empty:
            continue
        fig.add_trace(go.Scatter(
            x=log["exit_time"],
            y=log["cumulative_pnl"],
            mode="lines",
            name=f"{tf}-min",
            line=dict(color=colors[tf], width=3 if tf == 15 else 1.5),
        ))
    fig.add_hline(y=0, line_color="gray", line_dash="dash", line_width=1)
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=50, r=20, t=20, b=30),
        yaxis_tickprefix="$", yaxis_tickformat=",.0f",
        legend=dict(orientation="h", y=1.05),
        hovermode="x unified",
    )
    return fig


def _build_tf_bar_charts(tf_data: dict, colors: dict) -> dict:
    tfs  = [tf for tf in [5, 15, 30] if tf in tf_data]
    lbls = [f"{tf}-min" for tf in tfs]
    clrs = [colors[tf] for tf in tfs]

    def _bar(values, title, prefix="", suffix=""):
        fig = go.Figure(go.Bar(
            x=lbls, y=values, marker_color=clrs,
            text=[f"{prefix}{v}{suffix}" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            template="plotly_dark", title=title,
            margin=dict(l=30, r=10, t=45, b=30),
            yaxis=dict(tickprefix=prefix),
            showlegend=False,
        )
        return fig

    return {
        "calmar":   _bar([tf_data[tf]["calmar"]   for tf in tfs], "Calmar Ratio (higher = better)"),
        "win_rate": _bar([tf_data[tf]["win_rate"] for tf in tfs], "Win Rate (%)", suffix="%"),
        "avg_pnl":  _bar([tf_data[tf]["avg_pnl"]  for tf in tfs], "Avg P&L / Trade", prefix="$"),
    }


def _build_tf_exit_chart(tf_data: dict, colors: dict) -> go.Figure:
    tfs  = [tf for tf in [5, 15, 30] if tf in tf_data]
    lbls = [f"{tf}-min" for tf in tfs]
    fig  = go.Figure()
    for reason, color in [("tp_hits", "#26a69a"), ("sl_hits", "#ef5350"),
                           ("trend_exits", "#ffb74d"), ("force_exits", "#9e9e9e")]:
        name = {"tp_hits": "TP Hit", "sl_hits": "SL Hit",
                "trend_exits": "Trend Reversal", "force_exits": "Force/EOD"}[reason]
        fig.add_trace(go.Bar(
            name=name, x=lbls,
            y=[tf_data[tf][reason] for tf in tfs],
            marker_color=color,
        ))
    fig.update_layout(
        template="plotly_dark", barmode="group",
        margin=dict(l=30, r=10, t=20, b=30),
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def _tf_why_box() -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.H5("Why 15-Minute Bars?", className="text-success mb-3"),
        dbc.Row([
            dbc.Col([
                html.P([html.Strong("vs 5-min: "), "Fewer false signals. "
                    "5-min generates 3× more trades but at 33.6% win rate vs 46% — most are noise. "
                    "Avg P&L per trade is only $117 (5-min) vs $820 (15-min). "
                    "The higher trade count burns more in bid/ask spread and commissions."],
                    className="small mb-2"),
            ], lg=6),
            dbc.Col([
                html.P([html.Strong("vs 30-min: "), "More opportunities. "
                    "30-min has a similar win rate (47.6%) but generates only 63 trades — "
                    "half as many as 15-min. Total P&L is lower ($90k vs $114k) and "
                    "the Calmar ratio is 49.8 vs 61.5. Fewer signals means more capital "
                    "sitting idle between trades."],
                    className="small mb-2"),
            ], lg=6),
        ]),
        html.Hr(style={"borderColor": "#444"}),
        html.P([
            html.Strong("Calmar ratio (Total P&L ÷ Max Drawdown) "),
            "is the ranking metric — it rewards high returns relative to risk taken. "
            "15-min scores highest at 61.5, meaning it earns $61.50 for every $1 of "
            "maximum drawdown experienced. That's the best risk-adjusted return of the three."
        ], className="small mb-0 text-muted"),
    ]), color="dark", outline=True)


def _build_contract_chart(comparison: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    contract_sizes = sorted(comparison["contracts"].unique())

    for direction, color in [("CALL", "#26a69a"), ("PUT", "#ef5350"),
                              ("COMBINED", "#42a5f5")]:
        subset = comparison[comparison["direction"] == direction].sort_values("contracts")
        fig.add_trace(go.Bar(
            name=direction,
            x=[str(c) for c in subset["contracts"]],
            y=subset["total_pnl"],
            marker_color=color,
            opacity=0.85,
            text=subset["total_pnl"].apply(lambda x: f"${x:,.0f}"),
            textposition="outside",
        ))

    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        title="Total P&L by Contract Size",
        xaxis_title="Contracts",
        yaxis_title="Total P&L ($)",
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", y=1.12),
    )
    return fig
