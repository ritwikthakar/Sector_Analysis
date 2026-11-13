"""Constituent selection and per-stock trend analysis for the sector drill-down tab.

Pure pandas / numpy (no Streamlit or yfinance), so it can be tested offline.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import indicators as ind

MA_LENGTHS = (5, 10, 20, 50, 200)
QUAD_ORDER = {"Leading": 0, "Improving": 1, "Weakening": 2, "Lagging": 3, "n/a": 4}

# Fallback candidate pools (large caps per SPDR sector). Best-effort and can go stale:
# live ETF holdings are merged in first, and everything is re-ranked by live market cap.
# The pool only needs to contain the sector's true top 10, so it is deliberately a bit wider.
STATIC_POOLS = {
    "XLK": "NVDA AAPL MSFT AVGO ORCL PLTR AMD CSCO IBM MU CRM NOW INTU AMAT LRCX QCOM TXN ANET".split(),
    "XLF": "BRK-B JPM V MA BAC WFC GS MS C AXP BLK SCHW SPGI PGR CB".split(),
    "XLV": "LLY JNJ ABBV UNH MRK ABT ISRG TMO AMGN GILD DHR PFE BSX SYK MDT VRTX".split(),
    "XLY": "AMZN TSLA HD MCD BKNG LOW TJX SBUX NKE ORLY CMG MAR GM ABNB RCL".split(),
    "XLP": "WMT COST PG KO PEP PM MDLZ MO CL TGT KMB KR KDP STZ MNST".split(),
    "XLE": "XOM CVX COP WMB EOG SLB KMI OKE PSX MPC VLO BKR TRGP OXY FANG".split(),
    "XLI": "GE RTX CAT BA UBER HON UNP DE ETN LMT GEV ADP PH WM TT".split(),
    "XLB": "LIN SHW ECL APD FCX NEM CTVA MLM VMC DD NUE DOW PPG STLD IP".split(),
    "XLU": "NEE SO DUK CEG VST AEP SRE D EXC XEL PEG ED ETR WEC PCG".split(),
    "XLRE": "PLD AMT WELL EQIX SPG DLR O PSA CCI VICI CBRE EXR AVB IRM CSGP".split(),
    "XLC": "META GOOGL NFLX DIS T VZ TMUS CMCSA CHTR EA TTWO WBD LYV OMC".split(),
}

# Second share class of the same company: keep the first, drop the second when both are present
_SAME_COMPANY = {"GOOG": "GOOGL", "FOX": "FOXA", "NWS": "NWSA"}
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9\-]{0,5}$")


# ---------------------------------------------------------------------------
# Constituents
# ---------------------------------------------------------------------------
def normalize(sym) -> str:
    return str(sym).strip().upper().replace(".", "-")


def dedupe(tickers) -> list:
    norm = [normalize(t) for t in tickers if str(t).strip()]
    present = set(norm)
    out: list = []
    for t in norm:
        if t in out or not _TICKER_RE.match(t):
            continue
        if t in _SAME_COMPANY and _SAME_COMPANY[t] in present:
            continue
        out.append(t)
    return out


def parse_holdings(df) -> dict:
    """Yahoo `funds_data.top_holdings` frame -> {ticker: {"weight": pct, "name": str}}."""
    if df is None or len(df) == 0:
        return {}
    d = df.copy()
    cols = {str(c).lower(): c for c in d.columns}
    sym_col = next((cols[k] for k in cols if "symbol" in k), None)
    pct_col = next((cols[k] for k in cols if "percent" in k or "weight" in k), None)
    name_col = next((cols[k] for k in cols if "name" in k), None)
    symbols = d[sym_col].tolist() if sym_col is not None else list(d.index)
    vals = pd.to_numeric(d[pct_col], errors="coerce") if pct_col is not None else pd.Series(np.nan, index=d.index)
    scale = 100.0 if vals.notna().any() and vals.max() <= 1.5 else 1.0  # fractions -> percent
    out: dict = {}
    for i, sym in enumerate(symbols):
        t = normalize(sym)
        if t in out or not _TICKER_RE.match(t):
            continue
        w = vals.iloc[i]
        out[t] = {
            "weight": float(w * scale) if pd.notna(w) else np.nan,
            "name": str(d[name_col].iloc[i]) if name_col is not None else "",
        }
    return out


def select_top(candidates, caps: dict, n: int = 10) -> list:
    """Top n by market cap. Candidates with no cap data go after those with one, in input order."""
    uniq = dedupe(candidates)
    with_cap = [t for t in uniq if caps.get(t) and caps[t] > 0]
    without = [t for t in uniq if t not in with_cap]
    return (sorted(with_cap, key=lambda t: -caps[t]) + without)[:n]


# ---------------------------------------------------------------------------
# Moving-average state
# ---------------------------------------------------------------------------
def with_smas(d: pd.DataFrame, lengths=MA_LENGTHS) -> pd.DataFrame:
    out = d.copy()
    for n in lengths:
        out[f"sma_{n}"] = out["Close"].rolling(n).mean()
    return out


def stack_label(smas: dict) -> str:
    vals = [smas[n] for n in MA_LENGTHS]
    if any(pd.isna(v) for v in vals):
        return "n/a"
    if all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
        return "Bull stack"  # 5 > 10 > 20 > 50 > 200
    if all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
        return "Bear stack"  # 5 < 10 < 20 < 50 < 200
    return "Mixed"


def trend_label(count: int, above200: bool, stack: str) -> str:
    """Trend read from how many of the five SMAs price is above, and whether it is above the 200."""
    if count == 5:
        return "Strong uptrend" if stack == "Bull stack" else "Uptrend"
    if count == 4:
        return "Uptrend" if above200 else "Recovering"
    if count == 0:
        return "Strong downtrend" if stack == "Bear stack" else "Downtrend"
    if count == 1:
        return "Deep pullback" if above200 else "Downtrend"
    return "Pullback in uptrend" if above200 else "Bounce in downtrend"


def stock_snapshot(
    df: pd.DataFrame,
    donch_n: int = 20,
    adx_n: int = 14,
    atr_n: int = 14,
    adx_min: float = 20,
    adx_strong: float = 25,
) -> dict:
    """Latest-bar trend read: SMAs 5/10/20/50/200, Donchian, ADX/DI and ATR."""
    d = ind.add_indicators(df, donch_n=donch_n, sma_n=200, adx_n=adx_n, atr_n=atr_n)
    snap = ind.latest_snapshot(d, adx_min, adx_strong)
    close = d["Close"]
    c = close.iloc[-1]
    smas = {n: close.rolling(n).mean().iloc[-1] for n in MA_LENGTHS}
    flags = {n: bool(pd.notna(smas[n]) and c > smas[n]) for n in MA_LENGTHS}
    count = sum(flags.values())
    stack = stack_label(smas)
    if pd.isna(smas[200]):
        trend, cross, count_txt = "n/a (<200 bars)", "n/a", "n/a"
    else:
        trend = trend_label(count, flags[200], stack)
        cross = "50 > 200" if smas[50] > smas[200] else "50 < 200"
        count_txt = f"{count}/5"
    tick = {True: "✔", False: "✘"}
    row = {
        "Trend": trend,
        "MAs above": count_txt,
        **{f">{n}": tick[flags[n]] if pd.notna(smas[n]) else "–" for n in MA_LENGTHS},
        "Stack": stack,
        "50/200": cross,
        "vs 50d %": (c / smas[50] - 1) * 100 if pd.notna(smas[50]) else np.nan,
        "vs 200d %": (c / smas[200] - 1) * 100 if pd.notna(smas[200]) else np.nan,
        "Donchian %": snap["Donchian %"],
        "Breakout": snap["Breakout"],
        "ADX": snap["ADX"],
        "ADX read": snap["Trend strength"],
        "DI bias": "+DI" if snap["+DI"] > snap["-DI"] else "-DI",
        "ATR %": snap["ATR %"],
        "Vol regime": snap["Vol regime"],
        "Score": snap["Score"],
    }
    return row


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
def stock_table(
    frames: dict,
    close: pd.DataFrame,
    ref: pd.Series,
    ratio: pd.DataFrame,
    mom: pd.DataFrame,
    caps: dict | None = None,
    meta: dict | None = None,
    donch_n: int = 20,
    adx_n: int = 14,
    atr_n: int = 14,
    adx_min: float = 20,
    adx_strong: float = 25,
) -> pd.DataFrame:
    """One row per stock: rotation quadrant vs `ref` plus the trend read, sorted Leading -> Lagging."""
    caps = caps or {}
    meta = meta or {}
    ret = ind.returns_table(close)
    ref_ret = ind.returns_table(ref.to_frame("ref")).iloc[0]
    rows: dict = {}
    for t, df in frames.items():
        if df is None or len(df) < 30 or t not in close.columns:
            continue
        m = meta.get(t, {})
        cap = caps.get(t)
        rows[t] = {
            "Name": m.get("name", ""),
            "Mkt cap $B": cap / 1e9 if cap else np.nan,
            "Wt %": m.get("weight", np.nan),
            "Quadrant": ind.quadrant(ratio[t].iloc[-1], mom[t].iloc[-1]),
            "RS-Ratio": ratio[t].iloc[-1],
            "RS-Mom": mom[t].iloc[-1],
            "1M %": ret.loc[t, "1M"],
            "3M %": ret.loc[t, "3M"],
            "Rel 3M": ret.loc[t, "3M"] - ref_ret["3M"],
            **stock_snapshot(df, donch_n, adx_n, atr_n, adx_min, adx_strong),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    if out.empty:
        return out
    out["_q"] = out["Quadrant"].map(QUAD_ORDER)
    return out.sort_values(["_q", "RS-Ratio"], ascending=[True, False]).drop(columns="_q")


QUAD_MEANING = {
    "Leading": "outperforming with improving or steady relative momentum",
    "Improving": "still behind on relative trend but momentum is turning up",
    "Weakening": "ahead on relative trend but momentum is fading",
    "Lagging": "behind on relative trend with weak momentum",
    "n/a": "not enough data",
}


def describe_stock(d: pd.DataFrame, row: pd.Series, ref_label: str) -> list:
    """Plain-English lines for one stock (rotation + moving averages)."""
    lines = [
        f"**Rotation vs {ref_label}:** {row['Quadrant']} ({QUAD_MEANING.get(row['Quadrant'], '')}); "
        f"RS-Ratio {row['RS-Ratio']:.1f}, RS-Momentum {row['RS-Mom']:.1f}, 3M excess return {row['Rel 3M']:+.1f} pts."
    ]
    last = d.iloc[-1]
    above, below = [], []
    for n in MA_LENGTHS:
        v = last.get(f"sma_{n}")
        if pd.notna(v):
            (above if last["Close"] > v else below).append(str(n))
    if above or below:
        a = f"above the {', '.join(above)} SMA{'s' if len(above) > 1 else ''}" if above else "below every SMA"
        b = f"; below the {', '.join(below)} SMA{'s' if len(below) > 1 else ''}" if below and above else ""
        tail = (
            f" Alignment: {row['Stack'].lower()}; {row['50/200']} "
            f"({row['vs 50d %']:+.1f}% vs 50d, {row['vs 200d %']:+.1f}% vs 200d)."
            if pd.notna(row["vs 200d %"])
            else ""
        )
        lines.append(f"**Moving averages:** {str(row['Trend']).lower()}; price is {a}{b}.{tail}")
    return lines
