"""Sidebar AI analyst for the sector-rotation screener.

The current screener tables (rotation ranking + daily price-action scan) are rebuilt
on every question and passed to Claude as context, so answers always reflect the
universe and data you have loaded.

API key: put ANTHROPIC_API_KEY in .streamlit/secrets.toml, or set it as an
environment variable, or paste it into the sidebar box (kept in this browser session only).
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

import indicators as ind

MODELS = ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"]
MAX_HISTORY = 20
MAX_TOKENS = 4096

SYSTEM_PROMPT = """You are a markets analyst embedded in a sector-rotation screener used by an active options and equity trader.

You are given the screener's current data: a rotation table (returns, relative strength vs the benchmark, composite rank, RRG-style quadrant, position vs 50d/200d SMA) and a daily price-action scan (Donchian channel position and breakouts, 200d SMA distance, ADX with +DI/-DI, ATR and volatility regime, a regime label and a -4..+4 trend score).

The data may also include a SECTOR DRILL-DOWN table: the top stocks inside one sector ETF, each with an RRG-style quadrant (vs the sector ETF or the benchmark, as stated), the 5/10/20/50/200 SMA state, Donchian, ADX and ATR. Use it for stock-level questions inside that sector.

Rules:
- Ground every claim in the supplied data and cite the specific numbers. If the data does not answer the question (news, flows, options positioning, fundamentals, earnings), say so rather than guessing.
- Relative-strength columns are measured against the stated benchmark. Note when the benchmark itself is falling, because relative strength can then mean "falling less".
- When discussing long or short candidates, give the evidence for and against, name what would confirm or invalidate the setup (for example a Donchian breakout, a reclaim of the 50d/200d, ADX and DI direction), and mention ATR-based risk. This is analysis, not personalised financial advice, so keep any disclaimer to one short line.
- RS-Ratio and RS-Momentum are an approximation of RRG, not the proprietary JdK formula.
- Be concise: lead with the answer, use short bullets or a small table when comparing, no filler."""

QUICK_PROMPTS = {
    "Long / short ideas": "Based on the current data, which tickers are the best long and short candidates and what would confirm each?",
    "Explain the rotation": "Summarise what the rotation data says: which groups are gaining, which are fading, and how the benchmark context changes the read.",
    "Best breakouts": "Which tickers have a Donchian breakout or are near one, and does ADX/DI support the move?",
}


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------
def build_context(
    close: pd.DataFrame,
    frames: dict,
    valid: list,
    benchmark: str,
    names: dict,
    universe_label: str,
    extra: str = "",
) -> str:
    """Compact text snapshot of the screener state for the model."""
    c = close[valid]
    bench = close[benchmark]
    ret = ind.returns_table(c)
    bench_ret = ind.returns_table(close[[benchmark]]).iloc[0]
    excess = ret.sub(bench_ret, axis=1)

    rot = pd.DataFrame({"Name": [names.get(t, t) for t in valid]}, index=valid)
    for p in ind.PERIODS:
        rot[p] = ret[p].round(1)
    for p in ("1M", "3M", "6M"):
        rot[f"Rel{p}"] = excess[p].round(1)
    try:
        now = ind.rotation_rank(c, bench)
        prev = ind.rotation_rank(c, bench, offset=21)
        rot["Rank"] = now
        rot["RankChg1M"] = prev - now
        ratio, mom = ind.rrg(c, bench)
        rot["RS-Ratio"] = ratio.iloc[-1].round(1)
        rot["RS-Mom"] = mom.iloc[-1].round(1)
        rot["Quadrant"] = [ind.quadrant(ratio[t].iloc[-1], mom[t].iloc[-1]) for t in valid]
    except Exception:
        pass
    rot["Above50d"] = c.iloc[-1] > c.rolling(50).mean().iloc[-1]
    rot["Above200d"] = c.iloc[-1] > c.rolling(200).mean().iloc[-1]
    if "Rank" in rot:
        rot = rot.sort_values("Rank")

    pa_frames = {t: frames[t] for t in valid + [benchmark] if t in frames}
    pa = ind.price_action_summary(pa_frames)
    if not pa.empty:
        keep = [
            "Close", "Regime", "Score", "vs SMA %", "SMA ATRs", "Breakout", "Donchian %",
            "Days since high", "Days since low", "ADX", "ADX Δ5", "+DI", "-DI",
            "ATR %", "ATR pctile", "Vol regime",
        ]
        pa = pa[[k for k in keep if k in pa.columns]].round(1)

    bench_line = ", ".join(f"{p} {bench_ret[p]:+.1f}%" for p in ind.PERIODS)
    parts = [
        f"Universe: {universe_label}",
        f"Benchmark: {benchmark} ({names.get(benchmark, benchmark)}); benchmark returns: {bench_line}",
        f"Last daily bar: {close.index[-1].date()}",
        "Price-action defaults: Donchian 20, SMA 200, ADX 14, ATR 14 (sidebar/tab settings may differ).",
        "Returns are % over 1W/1M/3M/6M/12M trading-day lookbacks; RelXM = excess return vs benchmark in points; "
        "RankChg1M > 0 = moved up the ranking over the last month; Rank 1 = strongest.",
        "",
        "ROTATION TABLE (CSV)",
        rot.to_csv(),
        "DAILY PRICE-ACTION SCAN (CSV)",
        pa.to_csv() if not pa.empty else "(not enough history)",
    ]
    if extra:
        parts += ["", extra]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def _secret_key() -> str:
    try:
        k = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        k = ""
    return k or os.environ.get("ANTHROPIC_API_KEY", "")


def _stream(api_key: str, model: str, context: str, messages: list, info: dict):
    """Yield reply text; fill `info` with stop reason / block types for diagnostics."""
    import anthropic  # imported lazily so the rest of the app works without it

    client = anthropic.Anthropic(api_key=api_key)
    system = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {"type": "text", "text": context, "cache_control": {"type": "ephemeral"}},
    ]
    with client.messages.stream(model=model, max_tokens=MAX_TOKENS, system=system, messages=messages) as s:
        for text in s.text_stream:
            yield text
        final = s.get_final_message()
        info["stop_reason"] = final.stop_reason
        info["blocks"] = [b.type for b in final.content]
        info["output_tokens"] = getattr(final.usage, "output_tokens", None)


def _api_messages(history: list) -> list:
    """Last MAX_HISTORY turns, starting on a user message (the API requires that).

    Empty messages (e.g. from a reply that came back blank) are dropped, since the API rejects them.
    """
    msgs = [m for m in history if str(m.get("content", "")).strip()][-MAX_HISTORY:]
    while msgs and msgs[0]["role"] != "user":
        msgs = msgs[1:]
    return msgs


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def render_chat(context_fn) -> None:
    """Render the chat in the sidebar. context_fn() is called only when a question is sent."""
    with st.sidebar:
        st.divider()
        st.header("AI analyst")

        key = _secret_key()
        if not key:
            key = st.text_input(
                "Anthropic API key",
                type="password",
                help="Or set ANTHROPIC_API_KEY in .streamlit/secrets.toml. Kept for this browser session only.",
            ).strip()

        with st.expander("Chat settings"):
            model = st.selectbox("Model", MODELS, index=0)
            if st.button("Clear chat"):
                st.session_state["chat_msgs"] = []

        history = st.session_state.setdefault("chat_msgs", [])

        box = st.container(height=430)
        with box:
            if not history:
                st.caption("Ask about the data currently loaded: rotation, breakouts, ADX/ATR, long/short candidates.")
            for m in history:
                if not str(m["content"]).strip():
                    continue
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

        for i, (label, text) in enumerate(QUICK_PROMPTS.items()):
            if st.button(label, key=f"quick_{i}", use_container_width=True):
                st.session_state["chat_pending"] = text

        prompt = st.chat_input("Ask about the screener…") or st.session_state.pop("chat_pending", None)
        if not prompt:
            return
        if not key:
            st.warning("Add your Anthropic API key above to chat.")
            return

        history.append({"role": "user", "content": prompt})
        with box:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                try:
                    info: dict = {}
                    reply = st.write_stream(_stream(key, model, context_fn(), _api_messages(history), info))
                    if isinstance(reply, str) and reply.strip():
                        history.append({"role": "assistant", "content": reply})
                    else:
                        history.pop()
                        st.warning(
                            "The model returned no text "
                            f"(stop reason: {info.get('stop_reason')}, blocks: {info.get('blocks')}, "
                            f"output tokens: {info.get('output_tokens')}). Try again or pick another model."
                        )
                except ImportError:
                    history.pop()
                    st.error("The `anthropic` package is missing. Run: pip install anthropic")
                except Exception as e:  # keep the conversation consistent if the call fails
                    history.pop()
                    st.error(f"{type(e).__name__}: {e}")
