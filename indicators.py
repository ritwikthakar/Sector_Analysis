"""Indicator and rotation maths for the sector-rotation screener.

Pure pandas / numpy, so it can be tested without Streamlit, yfinance or Plotly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Trading-day lookbacks used for the return columns
PERIODS = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "12M": 252}


# ---------------------------------------------------------------------------
# Core indicators
# ---------------------------------------------------------------------------
def wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing (RMA): an EMA with alpha = 1/n."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), n)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.DataFrame:
    """Wilder ADX with +DI / -DI."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    atr_ = atr(high, low, close, n)
    plus_di = 100 * wilder(plus_dm, n) / atr_
    minus_di = 100 * wilder(minus_dm, n) / atr_
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    return pd.DataFrame({"adx": wilder(dx, n), "plus_di": plus_di, "minus_di": minus_di})


def donchian(high: pd.Series, low: pd.Series, n: int = 20) -> pd.DataFrame:
    upper = high.rolling(n).max()
    lower = low.rolling(n).min()
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": (upper + lower) / 2})


def _bars_since_max(x: np.ndarray) -> float:
    return float(np.argmax(x[::-1]))


def _bars_since_min(x: np.ndarray) -> float:
    return float(np.argmin(x[::-1]))


def _pctile_last(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    return float((x <= x[-1]).mean() * 100)


def add_indicators(
    df: pd.DataFrame,
    donch_n: int = 20,
    sma_n: int = 200,
    adx_n: int = 14,
    atr_n: int = 14,
    chand_mult: float = 3.0,
    slope_n: int = 20,
) -> pd.DataFrame:
    """Return OHLC plus Donchian, SMA, ADX/DI and ATR columns."""
    out = df[["Open", "High", "Low", "Close"]].copy()
    h, l, c = out["High"], out["Low"], out["Close"]

    dc = donchian(h, l, donch_n)
    out["dc_upper"], out["dc_lower"], out["dc_mid"] = dc["upper"], dc["lower"], dc["mid"]
    # Breakout = close beyond the PRIOR n-day channel (today's bar excluded)
    out["brk_up"] = c > dc["upper"].shift(1)
    out["brk_dn"] = c < dc["lower"].shift(1)
    width = (dc["upper"] - dc["lower"]).replace(0, np.nan)
    out["dc_pos"] = (c - dc["lower"]) / width * 100
    out["bars_since_high"] = h.rolling(donch_n).apply(_bars_since_max, raw=True)
    out["bars_since_low"] = l.rolling(donch_n).apply(_bars_since_min, raw=True)

    out["sma"] = c.rolling(sma_n).mean()
    out["sma_slope"] = (out["sma"] / out["sma"].shift(slope_n) - 1) * 100

    out = out.join(adx(h, l, c, adx_n))

    out["atr"] = atr(h, l, c, atr_n)
    out["atr_pct"] = out["atr"] / c * 100
    out["atr_avg"] = out["atr"].rolling(50).mean()
    out["atr_pctile"] = (
        out["atr_pct"].rolling(252, min_periods=126).apply(_pctile_last, raw=True)
    )
    out["dist_sma_atr"] = (c - out["sma"]) / out["atr"]
    # Chandelier-style trailing stop for a long: highest high of the channel window minus k * ATR
    out["chand_long"] = dc["upper"] - chand_mult * out["atr"]
    return out


# ---------------------------------------------------------------------------
# Labels / scoring
# ---------------------------------------------------------------------------
def _sgn(x) -> int:
    return 0 if x is None or pd.isna(x) else int(np.sign(x))


def strength_label(adx_v: float, adx_min: float = 20, adx_strong: float = 25) -> str:
    if pd.isna(adx_v):
        return "n/a"
    if adx_v < adx_min:
        return "No trend"
    if adx_v < adx_strong:
        return "Emerging"
    if adx_v < 40:
        return "Trending"
    return "Very strong"


def vol_label(pctile: float) -> str:
    if pd.isna(pctile):
        return "n/a"
    if pctile < 20:
        return "Compressed"
    if pctile > 80:
        return "Expanded"
    return "Normal"


def classify_regime(close, sma, plus_di, minus_di, adx_v, adx_min: float = 20) -> str:
    if any(pd.isna(v) for v in (close, sma, plus_di, minus_di, adx_v)):
        return "n/a"
    above = close > sma
    bull = plus_di > minus_di
    if adx_v < adx_min:
        return "Range (ADX low)"
    if above and bull:
        return "Uptrend"
    if (not above) and (not bull):
        return "Downtrend"
    if above:
        return "Pullback in uptrend"
    return "Bounce in downtrend"


def trend_score(close, sma, plus_di, minus_di, dc_mid, brk_up, brk_dn) -> int:
    """-4..+4: close vs SMA, DI direction, close vs channel midline, breakout/breakdown."""
    return (
        _sgn(close - sma)
        + _sgn(plus_di - minus_di)
        + _sgn(close - dc_mid)
        + (1 if brk_up else 0)
        - (1 if brk_dn else 0)
    )


def latest_snapshot(d: pd.DataFrame, adx_min: float = 20, adx_strong: float = 25) -> dict:
    """One-row summary of the most recent bar of an add_indicators() frame."""
    last = d.iloc[-1]
    adx_prev = d["adx"].iloc[-6] if len(d) > 5 else np.nan
    close = last["Close"]
    vs_sma = (close / last["sma"] - 1) * 100 if pd.notna(last["sma"]) else np.nan
    if last["brk_up"]:
        brk = "Breakout ↑"
    elif last["brk_dn"]:
        brk = "Breakdown ↓"
    else:
        brk = "Inside"
    return {
        "Close": close,
        "vs SMA %": vs_sma,
        "SMA ATRs": last["dist_sma_atr"],
        "SMA slope %": last["sma_slope"],
        "Donchian %": last["dc_pos"],
        "Breakout": brk,
        "Days since high": last["bars_since_high"],
        "Days since low": last["bars_since_low"],
        "ADX": last["adx"],
        "+DI": last["plus_di"],
        "-DI": last["minus_di"],
        "ADX Δ5": last["adx"] - adx_prev,
        "Trend strength": strength_label(last["adx"], adx_min, adx_strong),
        "ATR": last["atr"],
        "ATR %": last["atr_pct"],
        "ATR pctile": last["atr_pctile"],
        "Vol regime": vol_label(last["atr_pctile"]),
        "Regime": classify_regime(close, last["sma"], last["plus_di"], last["minus_di"], last["adx"], adx_min),
        "Score": trend_score(close, last["sma"], last["plus_di"], last["minus_di"], last["dc_mid"], last["brk_up"], last["brk_dn"]),
    }


def price_action_summary(
    frames: dict,
    donch_n: int = 20,
    sma_n: int = 200,
    adx_n: int = 14,
    atr_n: int = 14,
    adx_min: float = 20,
    adx_strong: float = 25,
) -> pd.DataFrame:
    rows = {}
    for t, df in frames.items():
        if df is None or len(df) < 30:
            continue
        d = add_indicators(df, donch_n, sma_n, adx_n, atr_n)
        rows[t] = latest_snapshot(d, adx_min, adx_strong)
    return pd.DataFrame.from_dict(rows, orient="index")


def _ago(n) -> str:
    n = int(n)
    return "today" if n == 0 else "1 bar ago" if n == 1 else f"{n} bars ago"


def describe_latest(
    d: pd.DataFrame,
    donch_n: int = 20,
    sma_n: int = 200,
    adx_n: int = 14,
    atr_n: int = 14,
    adx_min: float = 20,
    adx_strong: float = 25,
    chand_mult: float = 3.0,
) -> list:
    """Plain-English read of the latest bar. Descriptive, not a trade signal."""
    s = latest_snapshot(d, adx_min, adx_strong)
    last = d.iloc[-1]
    lines = []

    if pd.notna(s["vs SMA %"]):
        slope = s["SMA slope %"]
        if pd.isna(slope):
            slope_txt = "slope n/a"
        else:
            word = "rising" if slope > 0 else "falling" if slope < 0 else "flat"
            slope_txt = f"{word}, {slope:+.1f}% over 20 days"
        lines.append(
            f"**Trend:** close {s['Close']:,.2f} is {s['vs SMA %']:+.1f}% vs the {sma_n}-day SMA "
            f"({s['SMA ATRs']:+.1f} ATR away); the SMA is {slope_txt}."
        )
    else:
        lines.append(f"**Trend:** fewer than {sma_n} bars of history, so the {sma_n}-day SMA is not available.")

    if pd.notna(s["ADX"]):
        who = "buyers (+DI)" if s["+DI"] > s["-DI"] else "sellers (-DI)"
        d5 = s["ADX Δ5"]
        mv = "rising" if d5 > 0 else "falling" if d5 < 0 else "flat" if pd.notna(d5) else ""
        mv_txt = f" and {mv}" if mv else ""
        lines.append(
            f"**Trend strength:** ADX({adx_n}) is {s['ADX']:.0f}{mv_txt} → {s['Trend strength'].lower()}; "
            f"{who} have the upper hand (+DI {s['+DI']:.0f} vs -DI {s['-DI']:.0f})."
        )

    if pd.notna(last["dc_upper"]):
        msg = (
            f"**Donchian({donch_n}):** price sits at {s['Donchian %']:.0f}% of the channel "
            f"({last['dc_lower']:,.2f} – {last['dc_upper']:,.2f}); the {donch_n}-day high was set "
            f"{_ago(s['Days since high'])} and the low {_ago(s['Days since low'])}."
        )
        if s["Breakout"] != "Inside":
            msg += f" Latest close is outside the prior {donch_n}-day range ({s['Breakout']})."
        lines.append(msg)

    if pd.notna(s["ATR"]):
        parts = [f"**Volatility:** ATR({atr_n}) is {s['ATR']:.2f} ({s['ATR %']:.2f}% of price)"]
        if pd.notna(s["ATR pctile"]):
            parts.append(f", {s['ATR pctile']:.0f}th percentile of the past year ({s['Vol regime'].lower()})")
        if pd.notna(last["atr_avg"]):
            parts.append(f", {'above' if s['ATR'] > last['atr_avg'] else 'below'} its 50-day average")
        parts.append(
            f". A {chand_mult:g}×ATR chandelier stop (channel high − {chand_mult:g}×ATR) sits at "
            f"{last['chand_long']:,.2f}."
        )
        lines.append("".join(parts))

    lines.append(f"**Regime:** {s['Regime']} · trend score {s['Score']:+d} (range -4 to +4).")
    return lines


# ---------------------------------------------------------------------------
# Rotation maths
# ---------------------------------------------------------------------------
def returns_table(close: pd.DataFrame, periods: dict = PERIODS) -> pd.DataFrame:
    """% return over each lookback; rows = tickers, columns = period labels."""
    out = {}
    for label, n in periods.items():
        if len(close) > n:
            out[label] = (close.iloc[-1] / close.iloc[-1 - n] - 1) * 100
        else:
            out[label] = pd.Series(np.nan, index=close.columns)
    return pd.DataFrame(out)


def rotation_rank(
    close: pd.DataFrame,
    bench: pd.Series,
    offset: int = 0,
    windows: tuple = ("1M", "3M", "6M"),
) -> pd.Series:
    """Composite rank (1 = strongest): average rank of excess return vs benchmark over several windows.

    offset > 0 evaluates the ranking as of `offset` bars ago (to measure rank change).
    """
    c = close.iloc[: len(close) - offset] if offset else close
    b = bench.iloc[: len(bench) - offset] if offset else bench
    cols = {}
    for w in windows:
        n = PERIODS[w]
        if len(c) <= n:
            cols[w] = pd.Series(np.nan, index=c.columns)
            continue
        tr = (c.iloc[-1] / c.iloc[-1 - n] - 1) * 100
        br = (b.iloc[-1] / b.iloc[-1 - n] - 1) * 100
        cols[w] = tr - br
    excess = pd.DataFrame(cols)
    avg_rank = excess.rank(ascending=False).mean(axis=1)
    return avg_rank.rank(method="min")


def rrg(
    close: pd.DataFrame,
    bench: pd.Series,
    smooth: int = 5,
    ratio_win: int = 63,
    mom_win: int = 10,
):
    """RRG-style RS-Ratio and RS-Momentum (an approximation, not the proprietary JdK formula).

    RS-Ratio    = 100 * smoothed relative strength / its `ratio_win`-day average
    RS-Momentum = 100 * RS-Ratio / RS-Ratio `mom_win` days ago
    """
    rs = close.div(bench, axis=0)
    rs_s = rs.rolling(smooth, min_periods=1).mean()
    ratio = 100 * rs_s / rs_s.rolling(ratio_win).mean()
    mom = 100 * ratio / ratio.shift(mom_win)
    return ratio, mom


def quadrant(ratio_v, mom_v) -> str:
    if pd.isna(ratio_v) or pd.isna(mom_v):
        return "n/a"
    if ratio_v >= 100 and mom_v >= 100:
        return "Leading"
    if ratio_v >= 100:
        return "Weakening"
    if mom_v >= 100:
        return "Improving"
    return "Lagging"


def _basket(close: pd.DataFrame, tickers) -> pd.Series:
    sub = close[list(tickers)].dropna()
    return (sub / sub.iloc[0]).mean(axis=1)


def pair_ratio(close: pd.DataFrame, num, den) -> pd.Series:
    """Ratio of two tickers or two equal-weight baskets (lists of tickers)."""
    n = _basket(close, num) if isinstance(num, (list, tuple)) else close[num]
    d = _basket(close, den) if isinstance(den, (list, tuple)) else close[den]
    return (n / d).dropna()
