"""Sector rotation screener: S&P 500 sectors (SPDR) + Nasdaq industry / theme ETFs.

Tab 1  Rotation screener  - returns, relative strength, RRG-style quadrants, rank changes,
                            risk-on / risk-off and concentration ratios.
Tab 2  Price action       - daily Donchian channels, 200-day SMA, ADX/DI and ATR for every
                            ticker in the universe, plus a detailed chart for one ticker.

Run:  streamlit run app.py
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

import chat
import ichimoku
import indicators as ind
import stocks

st.set_page_config(page_title="Sector Rotation Screener", layout="wide")

# ---------------------------------------------------------------------------
# Universes (edit freely; ETF tickers are what Yahoo Finance expects)
# ---------------------------------------------------------------------------
SP_SECTORS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

NASDAQ_THEMES = {
    "QQEW": "Nasdaq-100 Equal Weight",
    "QTEC": "Nasdaq-100 Technology",
    "SMH": "Semiconductors (VanEck)",
    "SOXX": "Semiconductors (iShares)",
    "IGV": "Software",
    "SKYY": "Cloud Computing",
    "WCLD": "Cloud Software",
    "CIBR": "Cybersecurity (First Trust)",
    "HACK": "Cybersecurity (Amplify)",
    "FDN": "Internet",
    "XBI": "Biotech (equal weight)",
    "IBB": "Biotech (Nasdaq Biotech)",
    "ARKK": "Innovation / high-beta growth",
}

BENCH_NAMES = {"SPY": "S&P 500", "QQQ": "Nasdaq-100"}

# Ratio panels: (numerator, denominator); a list means an equal-weight basket
SP_PAIRS = {
    "XLY / XLP (discretionary vs staples)": ("XLY", "XLP"),
    "XLI / XLU (industrials vs utilities)": ("XLI", "XLU"),
    "Cyclical / defensive basket": (["XLY", "XLI", "XLB", "XLF"], ["XLP", "XLU", "XLV"]),
    "SPY / RSP (cap-weight vs equal-weight)": ("SPY", "RSP"),
}
NQ_PAIRS = {
    "QQQ / QQEW (mega-cap concentration)": ("QQQ", "QQEW"),
    "QQQ / SPY (Nasdaq-100 vs S&P 500)": ("QQQ", "SPY"),
    "SMH / IGV (semis vs software)": ("SMH", "IGV"),
    "ARKK / QQQ (speculative growth vs core)": ("ARKK", "QQQ"),
}

PERIOD_LABELS = list(ind.PERIODS)
QUAD_FILL = {
    "Leading": "rgba(46,160,67,0.12)",
    "Weakening": "rgba(210,153,34,0.14)",
    "Lagging": "rgba(218,54,51,0.12)",
    "Improving": "rgba(56,139,253,0.12)",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def show_df(df: pd.DataFrame) -> None:
    try:
        st.dataframe(df, width="stretch")
    except Exception:
        st.dataframe(df, use_container_width=True)


def show_fig(fig: go.Figure) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def _flat(side) -> list:
    return [side] if isinstance(side, str) else list(side)


@st.cache_data(ttl=900, show_spinner="Downloading daily prices from Yahoo Finance…")
def load_prices(tickers: tuple, years: int) -> dict:
    """Daily split/dividend-adjusted OHLCV per ticker."""
    raw = yf.download(
        list(tickers),
        period=f"{years}y",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    frames: dict = {}
    if raw is None or raw.empty:
        return frames
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):  # (field, ticker) layout -> swap
            raw = raw.swaplevel(axis=1)
        level0 = raw.columns.get_level_values(0)
        for t in tickers:
            if t in level0:
                df = raw[t]
                df = df[[c for c in cols if c in df.columns]].dropna(subset=["Open", "High", "Low", "Close"])
                if len(df):
                    frames[t] = df
    else:  # single ticker, flat columns
        df = raw[[c for c in cols if c in raw.columns]].dropna(subset=["Open", "High", "Low", "Close"])
        if len(df):
            frames[tickers[0]] = df
    for t, df in frames.items():
        idx = pd.to_datetime(df.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        df.index = idx
    return frames


def drop_unfinished_bar(frames: dict) -> dict:
    """If the US market may still be open, drop a bar dated today."""
    now = pd.Timestamp.now(tz="America/New_York")
    if now.weekday() >= 5 or (now.hour, now.minute) >= (16, 15):
        return frames
    today = now.tz_localize(None).normalize()
    return {
        t: (df.iloc[:-1] if len(df) and df.index[-1].normalize() == today else df)
        for t, df in frames.items()
    }


# ---------------------------------------------------------------------------
# Figures: rotation tab
# ---------------------------------------------------------------------------
def rrg_figure(ratio: pd.DataFrame, mom: pd.DataFrame, tickers: list, tail_len: int, tail_step: int) -> go.Figure:
    n = len(ratio)
    idx = [n - 1 - k * tail_step for k in range(tail_len + 1)]
    idx = [i for i in idx if i >= 0][::-1]
    xs = ratio.iloc[idx][tickers]
    ys = mom.iloc[idx][tickers]

    allx = xs.values.flatten()
    ally = ys.values.flatten()
    allx = allx[np.isfinite(allx)]
    ally = ally[np.isfinite(ally)]
    fig = go.Figure()
    if allx.size == 0 or ally.size == 0:
        return fig

    hx = max(float(np.abs(allx - 100).max()) * 1.2, 1.0)
    hy = max(float(np.abs(ally - 100).max()) * 1.2, 1.0)
    xr = [100 - hx, 100 + hx]
    yr = [100 - hy, 100 + hy]

    quads = {
        "Leading": (100, xr[1], 100, yr[1]),
        "Improving": (xr[0], 100, 100, yr[1]),
        "Weakening": (100, xr[1], yr[0], 100),
        "Lagging": (xr[0], 100, yr[0], 100),
    }
    for name, (x0, x1, y0, y1) in quads.items():
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=QUAD_FILL[name], line_width=0, layer="below")

    colors = px.colors.qualitative.Alphabet
    for i, t in enumerate(tickers):
        x = xs[t].values
        y = ys[t].values
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.any():
            continue
        x, y = x[ok], y[ok]
        col = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers+text",
                text=[""] * (len(x) - 1) + [t],
                textposition="top center",
                marker=dict(size=[5] * (len(x) - 1) + [12], color=col),
                line=dict(width=1.5, color=col),
                name=t,
                showlegend=False,
                hovertemplate=f"{t}<br>RS-Ratio %{{x:.2f}}<br>RS-Momentum %{{y:.2f}}<extra></extra>",
            )
        )

    for txt, x, y, xa, ya in [
        ("Leading", 0.99, 0.98, "right", "top"),
        ("Improving", 0.01, 0.98, "left", "top"),
        ("Weakening", 0.99, 0.02, "right", "bottom"),
        ("Lagging", 0.01, 0.02, "left", "bottom"),
    ]:
        fig.add_annotation(
            text=txt, xref="paper", yref="paper", x=x, y=y, xanchor=xa, yanchor=ya,
            showarrow=False, font=dict(size=13, color="gray"),
        )
    fig.add_vline(x=100, line_width=1, line_color="gray")
    fig.add_hline(y=100, line_width=1, line_color="gray")
    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="RS-Ratio (relative trend)", range=xr),
        yaxis=dict(title="RS-Momentum", range=yr),
        title="Relative rotation (RRG-style)",
    )
    return fig


def returns_heatmap(ret: pd.DataFrame, excess: pd.DataFrame, order: list, names: dict, mode: str) -> go.Figure:
    data = excess if mode.startswith("Excess") else ret
    z = data.loc[order, PERIOD_LABELS]
    labels = [f"{t} · {names.get(t, t)}" for t in order]
    vals = z.values.astype(float)
    cap = float(np.nanpercentile(np.abs(vals), 95)) if np.isfinite(vals).any() else 1.0
    if not np.isfinite(cap) or cap == 0:
        cap = 1.0
    fig = go.Figure(
        go.Heatmap(
            z=vals,
            x=PERIOD_LABELS,
            y=labels,
            zmin=-cap,
            zmax=cap,
            colorscale="RdYlGn",
            text=np.round(vals, 1),
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}: %{z:.1f}<extra></extra>",
            showscale=False,
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        height=max(380, 34 * len(order) + 80),
        margin=dict(l=10, r=10, t=30, b=10),
        title="Returns, % (sorted by composite rank)" if not mode.startswith("Excess") else "Excess return vs benchmark, pts",
    )
    return fig


def ratio_panel(close: pd.DataFrame, pairs: dict, days: int):
    usable = {}
    for title, (num, den) in pairs.items():
        need = _flat(num) + _flat(den)
        if all(t in close.columns for t in need):
            r = ind.pair_ratio(close, num, den)
            if len(r) > 60:
                usable[title] = r
    if not usable:
        return None
    titles = []
    for t, r in usable.items():
        state = "above" if r.iloc[-1] > r.rolling(50).mean().iloc[-1] else "below"
        titles.append(f"{t} · {state} 50d SMA")
    rows = (len(usable) + 1) // 2
    fig = make_subplots(rows=rows, cols=2, subplot_titles=titles, vertical_spacing=0.16)
    for i, (title, r) in enumerate(usable.items()):
        sma = r.rolling(50).mean()
        rw, sw = r.iloc[-days:], sma.iloc[-days:]
        base = rw.iloc[0]
        row, col = i // 2 + 1, i % 2 + 1
        fig.add_trace(
            go.Scatter(x=rw.index, y=rw / base * 100, mode="lines", line=dict(width=2), showlegend=False, name=title),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(x=sw.index, y=sw / base * 100, mode="lines", line=dict(width=1, dash="dot"), showlegend=False, name="50d SMA"),
            row=row, col=col,
        )
    fig.update_layout(height=300 * rows, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ---------------------------------------------------------------------------
# Figures: price-action tab
# ---------------------------------------------------------------------------
def price_figure(
    d: pd.DataFrame,
    ticker: str,
    window: int,
    donch_n: int,
    sma_n: int,
    adx_min: float,
    adx_strong: float,
    chand_mult: float,
    show_brk: bool,
    show_chand: bool,
    extra_smas: tuple = (),
    show_dc_sma: bool = True,
    show_ichimoku: bool = False,
) -> go.Figure:
    x = d.iloc[-window:]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.58, 0.22, 0.20])

    fig.add_trace(
        go.Candlestick(x=x.index, open=x["Open"], high=x["High"], low=x["Low"], close=x["Close"], name=ticker, showlegend=False),
        row=1, col=1,
    )
    if show_dc_sma:
        fig.add_trace(
            go.Scatter(x=x.index, y=x["dc_upper"], mode="lines", line=dict(width=1, color="#7f8c8d"), name=f"Donchian {donch_n} upper"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x.index, y=x["dc_lower"], mode="lines", line=dict(width=1, color="#7f8c8d"),
                fill="tonexty", fillcolor="rgba(127,140,141,0.10)", name=f"Donchian {donch_n} lower",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x.index, y=x["dc_mid"], mode="lines", line=dict(width=1, dash="dot", color="#95a5a6"), name="Donchian mid"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=x.index, y=x["sma"], mode="lines", line=dict(width=2, color="#f39c12"), name=f"SMA {sma_n}"),
            row=1, col=1,
        )
        sma_colors = {5: "#1abc9c", 10: "#3498db", 20: "#9b59b6", 50: "#e67e22"}
        for n in extra_smas:
            if f"sma_{n}" in x.columns:
                fig.add_trace(
                    go.Scatter(x=x.index, y=x[f"sma_{n}"], mode="lines", line=dict(width=1, color=sma_colors.get(n, "#7f8c8d")), name=f"SMA {n}"),
                    row=1, col=1,
                )
    if show_ichimoku:
        ich = ichimoku.compute(d)
        w = ich[ich.index >= x.index[0]]
        sa, sb = w["span_a"], w["span_b"]
        top, bot = np.maximum(sa, sb), np.minimum(sa, sb)
        bull = sa >= sb
        valid_cloud = sa.notna() & sb.notna()
        for is_bull, fill, edge, nm in (
            (True, "rgba(46,204,113,0.20)", "rgba(46,204,113,0.55)", "Cloud (Span A > B)"),
            (False, "rgba(231,76,60,0.20)", "rgba(231,76,60,0.55)", "Cloud (Span A < B)"),
        ):
            m = bull if is_bull else ~bull
            m = (m | m.shift(1, fill_value=False) | m.shift(-1, fill_value=False)) & valid_cloud  # overlap one bar so colours join
            fig.add_trace(
                go.Scatter(x=w.index, y=bot.where(m), mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=w.index, y=top.where(m), mode="lines", line=dict(width=0.5, color=edge),
                           fill="tonexty", fillcolor=fill, hoverinfo="skip", name=nm),
                row=1, col=1,
            )
        fig.add_trace(
            go.Scatter(x=w.index, y=w["tenkan"], mode="lines", line=dict(width=1.5, color="#2563eb"), name="Tenkan (9)"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=w.index, y=w["kijun"], mode="lines", line=dict(width=1.5, color="#dc2626"), name="Kijun (26)"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=w.index, y=w["chikou"], mode="lines", line=dict(width=1, dash="dot", color="#16a34a"), name="Chikou"),
            row=1, col=1,
        )
    if show_brk:
        up = x[x["brk_up"]]
        dn = x[x["brk_dn"]]
        fig.add_trace(
            go.Scatter(x=up.index, y=up["Low"] * 0.995, mode="markers", marker=dict(symbol="triangle-up", size=8, color="#2ecc71"),
                       name=f"Close > prior {donch_n}d high", hoverinfo="skip"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=dn.index, y=dn["High"] * 1.005, mode="markers", marker=dict(symbol="triangle-down", size=8, color="#e74c3c"),
                       name=f"Close < prior {donch_n}d low", hoverinfo="skip"),
            row=1, col=1,
        )
    if show_chand:
        fig.add_trace(
            go.Scatter(x=x.index, y=x["chand_long"], mode="lines", line=dict(width=1, dash="dash", color="#e74c3c"),
                       name=f"Long stop (channel high − {chand_mult:g}×ATR)"),
            row=1, col=1,
        )

    fig.add_trace(go.Scatter(x=x.index, y=x["adx"], mode="lines", line=dict(width=2, color="#3498db"), name="ADX"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x.index, y=x["plus_di"], mode="lines", line=dict(width=1, color="#2ecc71"), name="+DI"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x.index, y=x["minus_di"], mode="lines", line=dict(width=1, color="#e74c3c"), name="-DI"), row=2, col=1)
    fig.add_hline(y=adx_min, line_dash="dot", line_width=1, line_color="gray", row=2, col=1)
    fig.add_hline(y=adx_strong, line_dash="dot", line_width=1, line_color="gray", row=2, col=1)

    fig.add_trace(go.Scatter(x=x.index, y=x["atr"], mode="lines", line=dict(width=2, color="#9b59b6"), name="ATR"), row=3, col=1)
    fig.add_trace(
        go.Scatter(x=x.index, y=x["atr_avg"], mode="lines", line=dict(width=1, dash="dot", color="#95a5a6"), name="ATR 50d avg"),
        row=3, col=1,
    )

    fig.update_layout(
        height=860,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(rangeslider_visible=False, rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="ADX / DI", row=2, col=1)
    fig.update_yaxes(title_text="ATR", row=3, col=1)
    return fig


# ---------------------------------------------------------------------------
# Tab 1: rotation screener
# ---------------------------------------------------------------------------
def render_rotation(close: pd.DataFrame, benchmark: str, valid: list, names: dict, pairs: dict, rrg_params: tuple) -> None:
    smooth, ratio_win, mom_win, tail_len, tail_step = rrg_params
    c = close[valid]
    bench = close[benchmark]

    ret = ind.returns_table(c)
    bench_ret = ind.returns_table(close[[benchmark]]).iloc[0]
    excess = ret.sub(bench_ret, axis=1)
    rank_now = ind.rotation_rank(c, bench)
    rank_prev = ind.rotation_rank(c, bench, offset=21)
    ratio, mom = ind.rrg(c, bench, smooth, ratio_win, mom_win)
    r_last, m_last = ratio.iloc[-1], mom.iloc[-1]
    last = c.iloc[-1]

    tbl = pd.DataFrame({"Name": [names.get(t, t) for t in valid], "Price": last.round(2)}, index=valid)
    for p in PERIOD_LABELS:
        tbl[p] = ret[p].round(1)
    for p in ("1M", "3M", "6M"):
        tbl[f"Rel {p}"] = excess[p].round(1)
    tbl["Rank"] = rank_now.astype("Int64")
    tbl["Rank Δ 1M"] = (rank_prev - rank_now).astype("Int64")
    tbl["RS-Ratio"] = r_last.round(2)
    tbl["RS-Mom"] = m_last.round(2)
    tbl["Quadrant"] = [ind.quadrant(r_last[t], m_last[t]) for t in valid]
    tbl["> 50d"] = last > c.rolling(50).mean().iloc[-1]
    tbl["> 200d"] = last > c.rolling(200).mean().iloc[-1]
    ordered = tbl.sort_values("Rank")

    best, worst = ordered.index[0], ordered.index[-1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Strongest (composite)", best, f"{tbl.loc[best, 'Rel 3M']:+.1f} pts 3M vs {benchmark}")
    m2.metric("Weakest (composite)", worst, f"{tbl.loc[worst, 'Rel 3M']:+.1f} pts 3M vs {benchmark}")
    rd = tbl["Rank Δ 1M"].astype("float")
    if rd.notna().any():
        riser = rd.idxmax()
        m3.metric("Biggest 1M rank riser", riser, f"{int(rd[riser]):+d} ranks")
    else:
        m3.metric("Biggest 1M rank riser", "n/a")
    qc = tbl["Quadrant"].value_counts()
    m4.metric("Leading / Improving", f"{qc.get('Leading', 0)} / {qc.get('Improving', 0)}")

    st.subheader("Screener")
    st.caption(
        f"Rank = average rank of excess return vs {benchmark} over 1M, 3M and 6M (1 = strongest). "
        "Rank Δ 1M > 0 means the ticker moved up the ranking over the last month."
    )
    show_df(ordered)
    st.download_button("Download CSV", ordered.to_csv().encode(), "rotation_screener.csv", "text/csv")

    left, right = st.columns([1.2, 1])
    with left:
        show_fig(rrg_figure(ratio, mom, valid, tail_len, tail_step))
        st.caption(
            "RS-Ratio = smoothed relative strength vs its own average (above 100 = outperforming trend). "
            "RS-Momentum = change in RS-Ratio. Clockwise drift Improving → Leading → Weakening → Lagging is the "
            "classic rotation sequence. This is an approximation of RRG, not the proprietary JdK calculation."
        )
    with right:
        mode = st.radio("Heatmap", ["Excess vs benchmark", "Absolute return"], horizontal=True)
        show_fig(returns_heatmap(ret, excess, list(ordered.index), names, mode))

    st.subheader("Relative strength vs benchmark")
    a, b = st.columns([3, 1])
    default_sel = list(ordered.index[:3]) + list(ordered.index[-2:])
    sel = a.multiselect("Tickers", valid, default=list(dict.fromkeys(default_sel)))
    max_win = int(max(60, min(500, len(c) - 1)))
    win = b.slider("Window (days)", 30, max_win, min(126, max_win))
    if sel:
        rs = c[sel].div(bench, axis=0).iloc[-win:].dropna(axis=1, how="all")
        rs = rs.apply(lambda s: s / s.dropna().iloc[0] * 100)
        fig = go.Figure()
        for t in rs.columns:
            fig.add_trace(go.Scatter(x=rs.index, y=rs[t], mode="lines", name=t))
        fig.add_hline(y=100, line_width=1, line_dash="dot", line_color="gray")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title=f"Relative to {benchmark} (start = 100)")
        show_fig(fig)

    st.subheader("Risk appetite and concentration")
    span = st.select_slider("Window", ["3M", "6M", "12M", "24M"], value="12M")
    days = {"3M": 63, "6M": 126, "12M": 252, "24M": 504}[span]
    fig = ratio_panel(close, pairs, days)
    if fig is None:
        st.info("Not enough data for the ratio panels.")
    else:
        show_fig(fig)
        st.caption("Rising line = numerator leading. Dotted line = 50-day SMA of the ratio.")


# ---------------------------------------------------------------------------
# Tab 2: price action
# ---------------------------------------------------------------------------
def render_price_action(frames: dict, valid: list, benchmark: str, names: dict) -> None:
    tickers = [t for t in valid + [benchmark] if t in frames]
    if not tickers:
        st.info("No price data available.")
        return

    with st.expander("Indicator settings", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        donch_n = int(c1.number_input("Donchian length", 5, 200, 20))
        sma_n = int(c2.number_input("Trend SMA length", 20, 400, 200))
        adx_n = int(c3.number_input("ADX length", 5, 50, 14))
        atr_n = int(c4.number_input("ATR length", 5, 50, 14))
        c5, c6, c7, c8 = st.columns(4)
        adx_min = float(c5.number_input("ADX: trend threshold", 10, 40, 20))
        adx_strong = float(c6.number_input("ADX: strong threshold", 15, 60, 25))
        chand_mult = float(c7.number_input("ATR stop multiple", 1.0, 6.0, 3.0, step=0.5))
        window = int(c8.slider("Chart window (days)", 100, 750, 300))

    summary = ind.price_action_summary(
        {t: frames[t] for t in tickers}, donch_n, sma_n, adx_n, atr_n, adx_min, adx_strong
    )
    if summary.empty:
        st.info("Not enough history to compute indicators.")
        return
    summary.insert(0, "Name", [names.get(t, t) for t in summary.index])

    f1, f2 = st.columns(2)
    regimes = sorted(summary["Regime"].unique())
    reg_sel = f1.multiselect("Regime filter", regimes, default=regimes)
    brk_sel = f2.selectbox("Donchian filter", ["All", "Breakout ↑", "Breakdown ↓", "Any breakout / breakdown"])

    view = summary[summary["Regime"].isin(reg_sel)]
    if brk_sel in ("Breakout ↑", "Breakdown ↓"):
        view = view[view["Breakout"] == brk_sel]
    elif brk_sel.startswith("Any"):
        view = view[view["Breakout"] != "Inside"]

    cols = [
        "Name", "Close", "Regime", "Score", "vs SMA %", "SMA ATRs", "SMA slope %",
        "Breakout", "Donchian %", "Days since high", "Days since low",
        "ADX", "ADX Δ5", "+DI", "-DI", "Trend strength",
        "ATR %", "ATR pctile", "Vol regime", "ATR",
    ]
    view = view[cols].sort_values(["Score", "ADX"], ascending=False)
    view = view.round({
        "Close": 2, "vs SMA %": 1, "SMA ATRs": 1, "SMA slope %": 1, "Donchian %": 0,
        "ADX": 1, "ADX Δ5": 1, "+DI": 1, "-DI": 1, "ATR %": 2, "ATR pctile": 0, "ATR": 2,
    })
    for col in ("Days since high", "Days since low"):
        view[col] = view[col].astype("Int64")

    st.subheader("Daily price-action scan")
    show_df(view)
    st.download_button("Download CSV", view.to_csv().encode(), "price_action_scan.csv", "text/csv")

    with st.expander("How the columns are defined"):
        st.markdown(
            f"""
- **Regime** – uses the {sma_n}-day SMA, +DI vs -DI and ADX({adx_n}):
  *Uptrend* = close above SMA, +DI > -DI, ADX ≥ {adx_min:g}; *Downtrend* = the mirror image;
  *Pullback in uptrend* = above SMA but -DI > +DI; *Bounce in downtrend* = below SMA but +DI > -DI;
  *Range* = ADX below {adx_min:g}.
- **Score (-4 to +4)** – +1/-1 each for close vs SMA, +DI vs -DI, close vs Donchian midline, plus +1 for a
  breakout / -1 for a breakdown.
- **Breakout / Breakdown** – today's close is beyond the *prior* {donch_n}-day Donchian high / low.
- **Donchian %** – where the close sits inside the {donch_n}-day channel (0 = low, 100 = high).
- **SMA ATRs** – distance of close from the SMA measured in ATR({atr_n}) units.
- **Trend strength** – ADX buckets: < {adx_min:g} none, {adx_min:g}–{adx_strong:g} emerging, {adx_strong:g}–40 trending, 40+ very strong.
- **ATR pctile / Vol regime** – today's ATR% ranked against the past year (< 20 compressed, > 80 expanded).
"""
        )

    st.subheader("Chart")
    sel = st.selectbox("Ticker", tickers, index=0, format_func=lambda t: f"{t} · {names.get(t, t)}")
    d = ind.add_indicators(frames[sel], donch_n, sma_n, adx_n, atr_n, chand_mult)

    last, prev = d.iloc[-1], d.iloc[-2]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Close", f"{last['Close']:,.2f}", f"{(last['Close'] / prev['Close'] - 1) * 100:+.2f}%")
    k2.metric(f"vs {sma_n}d SMA", "n/a" if pd.isna(last["sma"]) else f"{(last['Close'] / last['sma'] - 1) * 100:+.1f}%")
    k3.metric(f"ADX({adx_n})", "n/a" if pd.isna(last["adx"]) else f"{last['adx']:.0f}")
    k4.metric(f"ATR({atr_n}) %", "n/a" if pd.isna(last["atr_pct"]) else f"{last['atr_pct']:.2f}%")
    k5.metric(f"Donchian({donch_n}) pos", "n/a" if pd.isna(last["dc_pos"]) else f"{last['dc_pos']:.0f}%")

    for line in ind.describe_latest(d, donch_n, sma_n, adx_n, atr_n, adx_min, adx_strong, chand_mult):
        st.markdown("- " + line)

    o1, o2 = st.columns(2)
    show_brk = o1.checkbox("Mark Donchian breakouts", value=True)
    show_chand = o2.checkbox("Show ATR chandelier stop", value=False)
    show_fig(price_figure(d, sel, window, donch_n, sma_n, adx_min, adx_strong, chand_mult, show_brk, show_chand))
    st.caption("Descriptive analytics on daily adjusted prices, not trading advice.")


# ---------------------------------------------------------------------------
# Tab 3: sector drill-down (top 10 stocks by market cap)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner="Looking up ETF holdings…")
def fetch_holdings(etf: str) -> dict:
    """{ticker: {"weight": pct, "name": str}} from Yahoo's ETF top holdings; {} if unavailable."""
    try:
        return stocks.parse_holdings(yf.Ticker(etf).funds_data.top_holdings)
    except Exception:
        return {}


@st.cache_data(ttl=43200, show_spinner="Fetching market caps…")
def fetch_market_caps(tickers: tuple) -> dict:
    """{ticker: market cap in USD} via Yahoo fast_info (tickers it cannot resolve are omitted)."""

    def one(t: str):
        try:
            fi = yf.Ticker(t).fast_info
            for key in ("market_cap", "marketCap"):
                try:
                    v = fi[key]
                except Exception:
                    v = None
                if v:
                    return t, float(v)
        except Exception:
            pass
        return t, None

    with ThreadPoolExecutor(max_workers=8) as ex:
        return {t: v for t, v in ex.map(one, tickers) if v}


def render_drilldown(
    close: pd.DataFrame,
    frames: dict,
    valid: list,
    benchmark: str,
    names: dict,
    rrg_params: tuple,
    years: int,
    drop_today: bool,
) -> None:
    smooth, ratio_win, mom_win, tail_len, tail_step = rrg_params

    # --- sector overview + picker -------------------------------------------------
    c = close[valid]
    s_ratio, s_mom = ind.rrg(c, close[benchmark], smooth, ratio_win, mom_win)
    quad = {t: ind.quadrant(s_ratio[t].iloc[-1], s_mom[t].iloc[-1]) for t in valid}
    s_ret = ind.returns_table(c)
    overview = pd.DataFrame(
        {
            "Name": [names.get(t, t) for t in valid],
            "Quadrant": [quad[t] for t in valid],
            "RS-Ratio": s_ratio.iloc[-1].round(2),
            "RS-Mom": s_mom.iloc[-1].round(2),
            "1M %": s_ret["1M"].round(1),
            "3M %": s_ret["3M"].round(1),
        },
        index=valid,
    )
    st.subheader("Pick a sector")
    show_df(overview)
    etf = st.selectbox(
        "Sector / theme to drill into",
        valid,
        format_func=lambda t: f"{t} · {names.get(t, t)} · {quad[t]}",
        key="dd_etf",
    )

    with st.expander("Analysis settings"):
        a1, a2, a3, a4 = st.columns(4)
        donch_n = int(a1.number_input("Donchian length", 5, 200, 20, key="dd_donch"))
        adx_n = int(a2.number_input("ADX length", 5, 50, 14, key="dd_adx"))
        atr_n = int(a3.number_input("ATR length", 5, 50, 14, key="dd_atr"))
        adx_min = float(a4.number_input("ADX trend threshold", 10, 40, 20, key="dd_adxmin"))
        custom = st.text_input(
            "Override constituents (comma-separated tickers; leave blank to use the top 10 by market cap)",
            "",
            key="dd_custom",
        )

    # --- constituents ---------------------------------------------------------------
    live = fetch_holdings(etf)
    pool = stocks.STATIC_POOLS.get(etf, [])
    custom_list = stocks.dedupe(t for t in custom.split(",") if t.strip())
    candidates = custom_list or stocks.dedupe(list(live) + pool)
    if not candidates:
        st.info(
            f"Couldn't get holdings for {etf} from Yahoo and there is no fallback list for it. "
            "Enter tickers in 'Override constituents' above."
        )
        return
    caps = fetch_market_caps(tuple(sorted(candidates)))
    top = candidates[:15] if custom_list else stocks.select_top(candidates, caps, 10)
    meta = {t: live.get(t, {}) for t in top}
    st.caption(
        ("Custom list. " if custom_list else f"Ranked by live market cap from {len(live)} Yahoo ETF holdings + {len(pool)} fallback names. ")
        + f"{len(caps)}/{len(candidates)} market caps resolved."
        + (" Names without a market cap are listed last." if len(caps) < len(candidates) and not custom_list else "")
    )

    sframes = load_prices(tuple(sorted(top)), years)
    if drop_today:
        sframes = drop_unfinished_bar(sframes)
    missing = [t for t in top if t not in sframes]
    if missing:
        st.warning("No price data for: " + ", ".join(missing))
    top = [t for t in top if t in sframes]
    if not top:
        st.info("No price data available for these stocks.")
        return

    sclose = pd.DataFrame({t: sframes[t]["Close"] for t in top}).sort_index().ffill()
    ref_choice = st.radio(
        "Rotation measured against",
        [f"Sector ETF ({etf})", f"Benchmark ({benchmark})"],
        horizontal=True,
        key="dd_ref",
    )
    ref_ticker = etf if ref_choice.startswith("Sector") else benchmark
    ref_label = ref_choice
    ref = close[ref_ticker].reindex(sclose.index).ffill()
    ratio, mom = ind.rrg(sclose, ref, smooth, ratio_win, mom_win)

    table = stocks.stock_table(
        {t: sframes[t] for t in top}, sclose, ref, ratio, mom, caps, meta,
        donch_n, adx_n, atr_n, adx_min, 25.0,
    )
    if table.empty:
        st.info("Not enough history to analyse these stocks.")
        return

    # --- summary --------------------------------------------------------------------
    st.subheader(f"{etf} · {names.get(etf, etf)}: top {len(table)} stocks")
    qc = table["Quadrant"].value_counts()
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Leading", int(qc.get("Leading", 0)))
    k2.metric("Improving", int(qc.get("Improving", 0)))
    k3.metric("Weakening", int(qc.get("Weakening", 0)))
    k4.metric("Lagging", int(qc.get("Lagging", 0)))
    k5.metric("Above 50d", f"{(table['>50'] == '✔').sum()}/{len(table)}")
    k6.metric("Above 200d", f"{(table['>200'] == '✔').sum()}/{len(table)}")

    cols = [
        "Name", "Mkt cap $B", "Wt %", "Quadrant", "RS-Ratio", "RS-Mom", "1M %", "3M %", "Rel 3M",
        "Trend", "MAs above", ">5", ">10", ">20", ">50", ">200", "Stack", "50/200", "vs 50d %", "vs 200d %",
        "Cloud", "Ichimoku", "Breakout", "Donchian %", "ADX", "ADX read", "DI bias", "ATR %", "Vol regime", "Score",
    ]
    view = table[cols].round(
        {"Mkt cap $B": 0, "Wt %": 1, "RS-Ratio": 1, "RS-Mom": 1, "1M %": 1, "3M %": 1, "Rel 3M": 1,
         "vs 50d %": 1, "vs 200d %": 1, "Donchian %": 0, "ADX": 1, "ATR %": 2}
    )
    show_df(view)
    st.download_button("Download CSV", view.to_csv().encode(), f"{etf}_top_stocks.csv", "text/csv")
    st.caption(
        f"Quadrant = RRG-style RS-Ratio / RS-Momentum of each stock vs {ref_label} (sorted Leading → Lagging). "
        "Trend comes from how many of the 5/10/20/50/200 SMAs price is above (5 > 10 > 20 > 50 > 200 = bull stack). "
        "Score = -4…+4 from close vs 200 SMA, DI direction, Donchian midline and breakout. "
        "Cloud / Ichimoku = price vs the Ichimoku cloud and an overall signal (classic 9/26/52). "
        "Weights are from the ETF's own holdings, where available."
    )

    try:  # let the AI analyst in the sidebar see this table
        st.session_state["chat_extra"] = (
            f"SECTOR DRILL-DOWN: top stocks in {etf} ({names.get(etf, etf)}); rotation measured vs {ref_ticker}\n"
            + view.to_csv()
        )
    except Exception:
        pass

    # --- rotation chart ---------------------------------------------------------------
    left, right = st.columns([1.2, 1])
    with left:
        show_fig(rrg_figure(ratio, mom, list(table.index), tail_len, tail_step))
    with right:
        q_order = ["Leading", "Improving", "Weakening", "Lagging"]
        for q in q_order:
            members = list(table.index[table["Quadrant"] == q])
            st.markdown(f"**{q}** ({len(members)}): " + (", ".join(members) if members else "none"))

    # --- one stock in detail ------------------------------------------------------------
    st.subheader("Stock detail")
    d1, d2 = st.columns([2, 1])
    sel = d1.selectbox("Stock", list(table.index), key="dd_stock",
                       format_func=lambda t: f"{t} · {table.loc[t, 'Quadrant']} · {table.loc[t, 'Trend']}")
    window = int(d2.slider("Chart window (days)", 100, 750, 300, key="dd_window"))
    d = stocks.with_smas(ind.add_indicators(sframes[sel], donch_n, 200, adx_n, atr_n, 3.0))
    for line in stocks.describe_stock(d, table.loc[sel], ref_label) + ind.describe_latest(
        d, donch_n, 200, adx_n, atr_n, adx_min, 25.0, 3.0
    ):
        st.markdown("- " + line)
    st.markdown("**Ichimoku analysis** (classic 9 / 26 / 52 settings)")
    for line in ichimoku.describe(ichimoku.read(sframes[sel])):
        st.markdown("- " + line)

    o1, o2, o3, o4 = st.columns(4)
    show_dc_sma = o1.checkbox("DC + SMA", value=True, key="dd_dcsma",
                              help="Donchian channel plus the 5/10/20/50/200 SMAs")
    show_ich = o2.checkbox("Ichimoku cloud", value=False, key="dd_ich",
                           help="Cloud, Tenkan, Kijun and Chikou (cloud projected 26 bars ahead)")
    show_brk = o3.checkbox("Mark Donchian breakouts", value=True, key="dd_brk")
    show_chand = o4.checkbox("Show ATR chandelier stop", value=False, key="dd_chand")
    show_fig(
        price_figure(d, sel, window, donch_n, 200, adx_min, 25.0, 3.0, show_brk, show_chand,
                     extra_smas=(5, 10, 20, 50), show_dc_sma=show_dc_sma, show_ichimoku=show_ich)
    )
    st.caption("Descriptive analytics on daily adjusted prices, not trading advice.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("Sector Rotation Screener")

    st.sidebar.header("Universe")
    uni_choice = st.sidebar.radio("Universe", ["S&P 500 sectors (SPDR)", "Nasdaq industry / themes"])
    is_sp = uni_choice.startswith("S&P")
    base = SP_SECTORS if is_sp else NASDAQ_THEMES
    pairs = SP_PAIRS if is_sp else NQ_PAIRS
    benchmark = st.sidebar.text_input("Benchmark", "SPY" if is_sp else "QQQ").strip().upper()

    extra_raw = st.sidebar.text_input("Extra tickers (comma-separated)", "")
    extras = [t.strip().upper() for t in extra_raw.split(",") if t.strip()]
    names = dict(base)
    for t in extras:
        names.setdefault(t, t)
    names.setdefault(benchmark, BENCH_NAMES.get(benchmark, benchmark))
    options = list(base) + [t for t in extras if t not in base]
    selected = st.sidebar.multiselect("Tickers", options, default=options)

    years = st.sidebar.selectbox("History (years)", [2, 3, 5], index=1)
    drop_today = st.sidebar.checkbox(
        "Ignore today's unfinished bar", value=False,
        help="If the US market is still open, drop today's partial daily bar so signals use completed closes only.",
    )
    with st.sidebar.expander("RRG settings"):
        smooth = st.slider("RS smoothing (days)", 1, 20, 5)
        ratio_win = st.slider("RS-Ratio baseline (days)", 20, 126, 63)
        mom_win = st.slider("RS-Momentum lookback (days)", 5, 40, 10)
        tail_len = st.slider("Tail length (points)", 0, 20, 8)
        tail_step = st.slider("Tail spacing (days)", 1, 10, 5)

    pair_tickers = {t for pair in pairs.values() for side in pair for t in _flat(side)}
    needed = sorted(set(selected) | {benchmark} | pair_tickers)
    frames = load_prices(tuple(needed), years)
    if drop_today:
        frames = drop_unfinished_bar(frames)

    if benchmark not in frames:
        st.error(f"No data returned for benchmark {benchmark}. Check the ticker or your connection.")
        st.stop()
    missing = [t for t in needed if t not in frames]
    if missing:
        st.warning("No data returned for: " + ", ".join(missing))

    close = pd.DataFrame({t: f["Close"] for t, f in frames.items()}).sort_index().ffill()
    valid = [t for t in selected if t in close.columns and t != benchmark]
    if not valid:
        st.info("Select at least one ticker with data.")
        st.stop()

    chat.render_chat(
        lambda: chat.build_context(
            close, frames, valid, benchmark, names, uni_choice, st.session_state.get("chat_extra", "")
        )
    )

    st.caption(
        f"Daily adjusted prices via Yahoo Finance · last bar {close.index[-1].date()} · benchmark {benchmark} "
        f"({names.get(benchmark, benchmark)})"
    )

    rrg_params = (smooth, ratio_win, mom_win, tail_len, tail_step)
    tab_rot, tab_pa, tab_dd = st.tabs(["Rotation screener", "Price action (daily)", "Sector drill-down"])
    with tab_rot:
        render_rotation(close, benchmark, valid, names, pairs, rrg_params)
    with tab_pa:
        render_price_action(frames, valid, benchmark, names)
    with tab_dd:
        render_drilldown(close, frames, valid, benchmark, names, rrg_params, years, drop_today)


main()
