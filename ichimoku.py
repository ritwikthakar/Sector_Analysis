"""Ichimoku Kinko Hyo (classic 9 / 26 / 52, displacement 26) for the sector drill-down.

Pure pandas / numpy. `compute` returns lines on an index that extends `disp` business days past the
last bar (the cloud is projected forward); `read` summarises the latest bar; `describe` turns that
into plain-English lines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TENKAN, KIJUN, SENKOU_B, DISP = 9, 26, 52, 26


def _mid(high: pd.Series, low: pd.Series, n: int) -> pd.Series:
    return (high.rolling(n).max() + low.rolling(n).min()) / 2


def compute(
    df: pd.DataFrame,
    tenkan: int = TENKAN,
    kijun: int = KIJUN,
    senkou_b: int = SENKOU_B,
    disp: int = DISP,
) -> pd.DataFrame:
    """tenkan, kijun, span_a, span_b, chikou on [df.index + `disp` future business days].

    span_a / span_b are plotted `disp` bars ahead of the bar that produced them;
    chikou is the close plotted `disp` bars back.
    """
    h, l, c = df["High"], df["Low"], df["Close"]
    idx = df.index
    n = len(idx)
    t = _mid(h, l, tenkan)
    k = _mid(h, l, kijun)
    raw_a = (t + k) / 2
    raw_b = _mid(h, l, senkou_b)

    future = pd.bdate_range(idx[-1] + pd.Timedelta(days=1), periods=disp)
    ext = idx.append(future)
    out = pd.DataFrame(index=ext)
    out["tenkan"] = t.reindex(ext)
    out["kijun"] = k.reindex(ext)

    span_a = np.full(len(ext), np.nan)
    span_b = np.full(len(ext), np.nan)
    span_a[disp: disp + n] = raw_a.values
    span_b[disp: disp + n] = raw_b.values
    out["span_a"], out["span_b"] = span_a, span_b

    chik = np.full(len(ext), np.nan)
    if n > disp:
        chik[: n - disp] = c.values[disp:]
    out["chikou"] = chik
    return out


def _sgn(x: float) -> int:
    return int(np.sign(x)) if pd.notna(x) else 0


def _where(price: float, top: float, bot: float) -> str:
    if pd.isna(top) or pd.isna(bot):
        return "n/a"
    return "above" if price > top else "below" if price < bot else "inside"


def read(
    df: pd.DataFrame,
    tenkan: int = TENKAN,
    kijun: int = KIJUN,
    senkou_b: int = SENKOU_B,
    disp: int = DISP,
) -> dict | None:
    """Latest-bar Ichimoku state, or None when there is not enough history (needs senkou_b + disp bars)."""
    n = len(df)
    if n < senkou_b + disp:
        return None
    ich = compute(df, tenkan, kijun, senkou_b, disp)
    close = float(df["Close"].iloc[-1])
    a_now, b_now = ich["span_a"].iloc[n - 1], ich["span_b"].iloc[n - 1]
    if pd.isna(a_now) or pd.isna(b_now):
        return None
    top, bot = max(a_now, b_now), min(a_now, b_now)
    pos = _where(close, top, bot)

    # cloud projected 26 bars ahead, and whether its colour flips before then
    a_f, b_f = ich["span_a"].iloc[-1], ich["span_b"].iloc[-1]
    future_green = bool(a_f >= b_f)
    s_now = _sgn(a_now - b_now)
    twist_in = None
    if s_now != 0:
        diff = (ich["span_a"].iloc[n:] - ich["span_b"].iloc[n:]).values
        for i, v in enumerate(diff, start=1):
            if pd.notna(v) and _sgn(v) not in (0, s_now):
                twist_in = i
                break

    t_now, k_now = ich["tenkan"].iloc[n - 1], ich["kijun"].iloc[n - 1]

    # last Tenkan / Kijun cross
    sign = np.sign((ich["tenkan"].iloc[:n] - ich["kijun"].iloc[:n]).dropna())
    tk = {"since": None, "dir": None, "loc": "n/a", "strength": "n/a"}
    if len(sign) > 1 and sign.iloc[-1] != 0:
        cur = sign.iloc[-1]
        diff_pos = np.where(sign.values != cur)[0]
        if diff_pos.size:
            cross_i = diff_pos[-1] + 1
            cross_bar = sign.index[cross_i]
            tk["since"] = int(len(sign) - 1 - cross_i)
            tk["dir"] = "bullish" if cur > 0 else "bearish"
            ci = ich.index.get_loc(cross_bar)
            loc = _where(ich["tenkan"].iloc[ci], max(ich["span_a"].iloc[ci], ich["span_b"].iloc[ci]),
                         min(ich["span_a"].iloc[ci], ich["span_b"].iloc[ci]))
            tk["loc"] = loc
            if loc == "inside":
                tk["strength"] = "neutral"
            elif loc != "n/a":
                with_trend = (cur > 0 and loc == "above") or (cur < 0 and loc == "below")
                tk["strength"] = "strong" if with_trend else "weak"

    # Chikou: today's close vs price (and cloud) 26 bars ago
    c26 = float(df["Close"].iloc[n - 1 - disp])
    a26, b26 = ich["span_a"].iloc[n - 1 - disp], ich["span_b"].iloc[n - 1 - disp]
    chik_cloud = _where(close, max(a26, b26), min(a26, b26)) if pd.notna(a26) and pd.notna(b26) else "n/a"

    comp = [
        1 if pos == "above" else -1 if pos == "below" else 0,
        _sgn(t_now - k_now),
        _sgn(close - c26),
        1 if future_green else -1,
    ]
    score = int(sum(comp))
    if pos == "inside":
        label = "Neutral (inside cloud)"
    elif score >= 3:
        label = "Strong bullish"
    elif score >= 1:
        label = "Bullish"
    elif score == 0:
        label = "Mixed"
    elif score >= -2:
        label = "Bearish"
    else:
        label = "Strong bearish"

    return {
        "close": close,
        "pos": pos,
        "top": float(top),
        "bot": float(bot),
        "thick_pct": float((top - bot) / close * 100),
        "now_green": bool(a_now >= b_now),
        "future_green": future_green,
        "twist_in": twist_in,
        "tenkan": float(t_now),
        "kijun": float(k_now),
        "tk_bull": bool(t_now > k_now),
        "tk": tk,
        "close_vs_kijun": float((close / k_now - 1) * 100),
        "c26": c26,
        "chikou_above_price": bool(close > c26),
        "chikou_cloud": chik_cloud,
        "score": score,
        "label": label,
    }


def describe(r: dict | None) -> list:
    """Plain-English Ichimoku read (descriptive, not a trade signal)."""
    if r is None:
        return ["**Ichimoku:** not enough history (needs about 78 daily bars)."]
    lines = []
    thick = "thin" if r["thick_pct"] < 2.5 else "thick" if r["thick_pct"] >= 6 else "average-thickness"
    support = {"thin": "weak", "average-thickness": "moderate", "thick": "strong"}[thick]

    if r["pos"] == "above":
        gap = (r["close"] / r["top"] - 1) * 100
        lines.append(
            f"**Price vs cloud:** close {r['close']:,.2f} is above the cloud (top {r['top']:,.2f}, bottom {r['bot']:,.2f}), "
            f"{gap:+.1f}% over the cloud top. The cloud is {thick} ({r['thick_pct']:.1f}% of price), so the support under price is {support}."
        )
    elif r["pos"] == "below":
        gap = (r["close"] / r["bot"] - 1) * 100
        lines.append(
            f"**Price vs cloud:** close {r['close']:,.2f} is below the cloud (bottom {r['bot']:,.2f}, top {r['top']:,.2f}), "
            f"{gap:+.1f}% from the cloud bottom. The cloud is {thick} ({r['thick_pct']:.1f}% of price), so the resistance overhead is {support}."
        )
    else:
        lines.append(
            f"**Price vs cloud:** close {r['close']:,.2f} is inside the cloud ({r['bot']:,.2f} – {r['top']:,.2f}). "
            "Ichimoku treats this as a neutral, no-trend zone until price closes above the top or below the bottom."
        )

    ahead = "green (Span A above Span B, bullish)" if r["future_green"] else "red (Span A below Span B, bearish)"
    twist = f" The cloud changes colour in about {r['twist_in']} bars." if r["twist_in"] else ""
    lines.append(f"**Cloud ahead (26 bars):** projected {ahead}; the current cloud is {'green' if r['now_green'] else 'red'}.{twist}")

    tk = r["tk"]
    rel = "above" if r["tk_bull"] else "below"
    cross = (
        f" The last cross was {tk['since']} bars ago ({tk['dir']}, made {tk['loc']} the cloud, a {tk['strength']} signal)."
        if tk["since"] is not None
        else " No Tenkan/Kijun cross in the available history."
    )
    lines.append(
        f"**Tenkan / Kijun:** Tenkan (9) {r['tenkan']:,.2f} is {rel} Kijun (26) {r['kijun']:,.2f}, "
        f"{'bullish' if r['tk_bull'] else 'bearish'} momentum.{cross} "
        f"Close is {r['close_vs_kijun']:+.1f}% {'above' if r['close_vs_kijun'] >= 0 else 'below'} Kijun."
    )

    cp = "above" if r["chikou_above_price"] else "below"
    cc = r["chikou_cloud"]
    if r["chikou_above_price"] and cc == "above":
        verdict = "confirming the uptrend (clear space behind it)"
    elif (not r["chikou_above_price"]) and cc == "below":
        verdict = "confirming the downtrend"
    else:
        verdict = "a mixed confirmation (the lagging span is obstructed)"
    lines.append(
        f"**Chikou span:** today's close is {cp} the price 26 bars ago ({r['c26']:,.2f})"
        + (f" and {cc} the cloud from that time" if cc != "n/a" else "")
        + f", {verdict}."
    )

    if r["pos"] == "above":
        levels = (
            f"Watch Kijun {r['kijun']:,.2f} and the cloud top {r['top']:,.2f}: a close below Kijun is the first sign of "
            "weakness, and a close back inside the cloud would neutralise the read."
        )
    elif r["pos"] == "below":
        levels = (
            f"Watch Kijun {r['kijun']:,.2f} and the cloud bottom {r['bot']:,.2f} as resistance: a close above Kijun is the first "
            "sign of recovery, and a close back inside the cloud would neutralise the read."
        )
    else:
        levels = f"Bullish above {r['top']:,.2f} (cloud top), bearish below {r['bot']:,.2f} (cloud bottom)."
    lines.append(
        f"**Ichimoku read:** {r['label']} (score {r['score']:+d} of ±4, from price vs cloud, Tenkan vs Kijun, "
        f"Chikou vs price and cloud colour ahead). {levels}"
    )
    return lines
