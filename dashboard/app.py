import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc


def create_app(df: pd.DataFrame, trade_log: pd.DataFrame,
               comparison: pd.DataFrame) -> dash.Dash:
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

    market_bars = df[df["in_market_hours"]].index
    date_options = [
        {"label": ts.strftime("%Y-%m-%d %H:%M"), "value": ts.strftime("%Y-%m-%d %H:%M")}
        for ts in market_bars
    ]
    default_date = market_bars[-1].strftime("%Y-%m-%d %H:%M") if len(market_bars) else None

    app.layout = dbc.Container([
        dbc.Row(dbc.Col(html.H2("QQQ EMA 8/21 Crossover — 0DTE Options Backtest",
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
                dbc.Row(dbc.Col(html.H4(
                    "0DTE | 1 Strike OTM | Calls & Puts — by Contract Size",
                    className="text-center mt-4 mb-3"
                ))),
                dbc.Row(dbc.Col(html.P(
                    "Results using the same EMA 8/21 crossover strategy and "
                    "TP/SL settings, varying only the number of contracts.",
                    className="text-center text-muted mb-4"
                ))),

                # Calls table
                dbc.Row(dbc.Col([
                    html.H5("CALLS", className="text-success mb-2"),
                    html.Div(id="calls-table"),
                ], width=12), className="mb-4"),

                # Puts table
                dbc.Row(dbc.Col([
                    html.H5("PUTS", className="text-danger mb-2"),
                    html.Div(id="puts-table"),
                ], width=12), className="mb-4"),

                # Combined table
                dbc.Row(dbc.Col([
                    html.H5("COMBINED (Calls + Puts)", className="text-info mb-2"),
                    html.Div(id="combined-table"),
                ], width=12), className="mb-4"),

                # Bar chart: Total P&L by contract size
                dbc.Row(dbc.Col(dcc.Graph(id="contract-chart",
                                          style={"height": "350px"}))),
            ]),
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
        # selected_date is "YYYY-MM-DD HH:MM" — extract just the date part
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

        fig = _build_chart(df, day_trades, selected_day, selected_date)
        table = _build_trade_table(day_trades)
        eq_fig = _build_equity(trade_log)
        return summary_cards, fig, table, eq_fig

    # ── Callback: contract comparison tab (static — runs once on load) ─────────
    @app.callback(
        Output("calls-table", "children"),
        Output("puts-table", "children"),
        Output("combined-table", "children"),
        Output("contract-chart", "figure"),
        Input("tabs", "active_tab"),
    )
    def update_comparison(_):
        calls_tbl = _build_comparison_table(comparison[comparison["direction"] == "CALL"],
                                             "#26a69a")
        puts_tbl = _build_comparison_table(comparison[comparison["direction"] == "PUT"],
                                            "#ef5350")
        comb_tbl = _build_comparison_table(comparison[comparison["direction"] == "COMBINED"],
                                            "#42a5f5")
        chart = _build_contract_chart(comparison)
        return calls_tbl, puts_tbl, comb_tbl, chart

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
                 selected_date: str, selected_datetime: str = None) -> go.Figure:
    if not selected_date:
        return go.Figure()
    day_df = df[df.index.strftime("%Y-%m-%d") == selected_date]
    if day_df.empty:
        return go.Figure()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.05)

    fig.add_trace(go.Candlestick(
        x=day_df.index, open=day_df["open"], high=day_df["high"],
        low=day_df["low"], close=day_df["close"], name="QQQ",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=day_df.index, y=day_df["ema8"], name="EMA 8",
                              line=dict(color="#ff9800", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=day_df.index, y=day_df["ema21"], name="EMA 21",
                              line=dict(color="#42a5f5", width=1.5)), row=1, col=1)

    exit_colors = {"TP": "#00e676", "SL": "#ff1744", "Cross": "#ffffff",
                   "Force": "#ffab40", "EOD": "#ce93d8"}

    if not day_trades.empty:
        for _, trade in day_trades.iterrows():
            color = "#26a69a" if trade["direction"] == "CALL" else "#ef5350"
            symbol_entry = "triangle-up" if trade["direction"] == "CALL" else "triangle-down"
            exit_color = exit_colors.get(trade.get("exit_reason", "Cross"), "#ffffff")
            pnl_str = f"${trade['pnl']:,.2f}" if isinstance(trade["pnl"], (int, float)) else trade["pnl"]

            fig.add_trace(go.Scatter(
                x=[trade["entry_time"]], y=[trade["entry_underlying"]],
                mode="markers",
                marker=dict(symbol=symbol_entry, size=14, color=color,
                            line=dict(color="#ffffff", width=1)),
                hovertemplate=(f"{trade['direction']} Entry<br>"
                               f"@{trade['entry_time'].strftime('%H:%M')}<br>"
                               f"Strike: {trade['strike']}<br>"
                               f"Premium: ${trade['entry_price']:.4f}<extra></extra>"),
                showlegend=False,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[trade["exit_time"]], y=[trade["exit_underlying"]],
                mode="markers",
                marker=dict(symbol="x-thin", size=14, color=exit_color,
                            line=dict(color=exit_color, width=2)),
                hovertemplate=(f"Exit ({trade.get('exit_reason', '')}) {pnl_str}<br>"
                               f"@{trade['exit_time'].strftime('%H:%M')}<extra></extra>"),
                showlegend=False,
            ), row=1, col=1)

    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(day_df["close"], day_df["open"])]
    fig.add_trace(go.Bar(x=day_df.index, y=day_df["volume"], name="Volume",
                          marker_color=colors, opacity=0.6), row=2, col=1)

    # Vertical marker + zoom window for the selected bar
    import pytz
    from datetime import timedelta
    EST = pytz.timezone("US/Eastern")

    x_range = None
    if selected_datetime:
        try:
            sel_ts = pd.Timestamp(selected_datetime).tz_localize(EST)
            window = timedelta(hours=1)
            x_start = max(sel_ts - window, day_df.index[0])
            x_end   = min(sel_ts + window, day_df.index[-1])
            x_range = [x_start, x_end]

            # Yellow dashed vertical line at the selected bar
            fig.add_vline(
                x=sel_ts.value / 1e6,   # milliseconds for plotly
                line_dash="dash",
                line_color="#ffd600",
                line_width=1.5,
                row=1, col=1,
            )
        except Exception:
            pass

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.05),
        xaxis=dict(range=x_range) if x_range else {},
    )
    fig.update_xaxes(tickformat="%H:%M", tickangle=-45, row=1, col=1)
    fig.update_xaxes(tickformat="%H:%M", tickangle=-45, row=2, col=1)
    return fig


def _build_trade_table(day_trades: pd.DataFrame) -> html.Div:
    if day_trades.empty:
        return html.P("No trades on this day.", className="text-muted")

    cols = ["direction", "strike", "entry_time", "entry_price",
            "exit_time", "exit_price", "exit_reason", "contracts", "pnl"]
    cols = [c for c in cols if c in day_trades.columns]
    display = day_trades[cols].copy()
    display["entry_time"] = display["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
    display["exit_time"] = display["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
    display["pnl"] = display["pnl"].apply(lambda x: f"${x:,.2f}")

    return dash_table.DataTable(
        data=display.to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in display.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#303030", "color": "white", "fontWeight": "bold"},
        style_cell={"backgroundColor": "#1a1a1a", "color": "white", "textAlign": "center"},
        style_data_conditional=[
            {"if": {"filter_query": '{pnl} contains "-"'}, "color": "#ef5350"},
            {"if": {"filter_query": '{pnl} contains "$" && !({pnl} contains "-")',
                    }, "color": "#26a69a"},
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


def _build_comparison_table(subset: pd.DataFrame, pnl_color: str) -> html.Div:
    if subset.empty:
        return html.P("No data.", className="text-muted")

    display = subset[["contracts", "trades", "win_rate",
                       "total_pnl", "avg_pnl", "best_trade", "worst_trade"]].copy()

    display["win_rate"] = display["win_rate"].apply(lambda x: f"{x:.1f}%")
    display["total_pnl"] = display["total_pnl"].apply(lambda x: f"${x:,.2f}")
    display["avg_pnl"] = display["avg_pnl"].apply(lambda x: f"${x:,.2f}")
    display["best_trade"] = display["best_trade"].apply(lambda x: f"${x:,.2f}")
    display["worst_trade"] = display["worst_trade"].apply(lambda x: f"${x:,.2f}")

    col_names = {
        "contracts": "Contracts",
        "trades": "Trades",
        "win_rate": "Win Rate",
        "total_pnl": "Total P&L",
        "avg_pnl": "Avg P&L / Trade",
        "best_trade": "Best Trade",
        "worst_trade": "Worst Trade",
    }

    return dash_table.DataTable(
        data=display.to_dict("records"),
        columns=[{"name": col_names[c], "id": c} for c in display.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#303030", "color": "white",
                      "fontWeight": "bold", "textAlign": "center"},
        style_cell={"backgroundColor": "#1a1a1a", "color": "white",
                    "textAlign": "center", "padding": "10px",
                    "fontSize": "14px"},
        style_data_conditional=[
            {"if": {"filter_query": '{total_pnl} contains "-"',
                    "column_id": "total_pnl"}, "color": "#ef5350"},
            {"if": {"filter_query": '{total_pnl} contains "$" && !({total_pnl} contains "-")',
                    "column_id": "total_pnl"}, "color": "#26a69a"},
            {"if": {"filter_query": '{avg_pnl} contains "-"',
                    "column_id": "avg_pnl"}, "color": "#ef5350"},
            {"if": {"filter_query": '{avg_pnl} contains "$" && !({avg_pnl} contains "-")',
                    "column_id": "avg_pnl"}, "color": "#26a69a"},
            {"if": {"column_id": "best_trade"}, "color": "#26a69a"},
            {"if": {"column_id": "worst_trade"}, "color": "#ef5350"},
            {"if": {"column_id": "contracts"}, "fontWeight": "bold",
             "color": pnl_color},
        ],
    )


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
