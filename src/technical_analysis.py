"""Technical analysis module for Picsou trading agent.

Computes standard technical indicators from OHLCV candle data and produces
both a structured dict of numeric values (with per-indicator signal labels)
and a concise English summary suitable for injection into an LLM prompt.

No external dependencies — pure-stdlib math only.
"""

from typing import Any, Dict, List, Tuple

# ── Type aliases ──────────────────────────────────────────────────────────
Candle = Dict[str, float]
Signal = str  # "bullish" | "bearish" | "neutral"


# ═══════════════════════════════════════════════════════════════════════════
#  Helper utilities
# ═══════════════════════════════════════════════════════════════════════════

def _sma(values: List[float], period: int) -> List[float]:
    """Simple moving average returning None-padded list of same length."""
    result: List[float] = []
    for i in range(len(values)):
        if i < period - 1:
            result.append(float("nan"))
        else:
            result.append(sum(values[i - period + 1 : i + 1]) / period)
    return result


def _ema(values: List[float], period: int) -> List[float]:
    """Exponential moving average (standard multiplier)."""
    if not values or len(values) < period:
        return [float("nan")] * len(values)
    result: List[float] = [float("nan")] * len(values)
    multiplier = 2.0 / (period + 1)
    seed = sum(values[:period]) / period
    result[period - 1] = seed
    for i in range(period, len(values)):
        result[i] = (values[i] - result[i - 1]) * multiplier + result[i - 1]
    return result


def _last(values: List[float]) -> float:
    """Return the last non-NaN value or NaN."""
    for v in reversed(values):
        if v == v:  # NaN check (NaN != NaN)
            return v
    return float("nan")


# ═══════════════════════════════════════════════════════════════════════════
#  Indicator calculators
# ═══════════════════════════════════════════════════════════════════════════

def calc_rsi(closes: List[float], period: int = 14) -> Dict[str, Any]:
    """Compute RSI (Wilder's smoothed version).

    Returns:
        dict with keys: value, signal, period, overbought, oversold
    """
    n = len(closes)
    if n < period + 1:
        return {"value": float("nan"), "signal": "neutral", "period": period}

    deltas = [closes[i] - closes[i - 1] for i in range(1, n)]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]

    # Initial averages
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Wilder's smoothing
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        rsi_val = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_val = 100.0 - (100.0 / (1.0 + rs))

    # Signal
    if rsi_val >= 70:
        signal: Signal = "bearish"
    elif rsi_val <= 30:
        signal = "bullish"
    else:
        signal = "neutral"

    return {
        "value": round(rsi_val, 2),
        "signal": signal,
        "period": period,
        "overbought": rsi_val >= 70,
        "oversold": rsi_val <= 30,
    }


def calc_macd(closes: List[float],
              fast: int = 12,
              slow: int = 26,
              signal_period: int = 9) -> Dict[str, Any]:
    """Compute MACD line, signal line, and histogram.

    Returns:
        dict with keys: macd, signal, histogram, signal_label
    """
    n = len(closes)
    if n < slow + signal_period:
        return {
            "macd": float("nan"),
            "signal": float("nan"),
            "histogram": float("nan"),
            "signal_label": "neutral",
            "fast_period": fast,
            "slow_period": slow,
            "signal_period": signal_period,
        }

    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)

    # MACD line = fast EMA - slow EMA
    macd_line = [f - s if (f == f and s == s) else float("nan")
                 for f, s in zip(ema_fast, ema_slow)]

    # Signal line = EMA of MACD line (using valid values only)
    valid_macd = [v for v in macd_line if v == v]
    if len(valid_macd) < signal_period:
        return {
            "macd": float("nan"),
            "signal": float("nan"),
            "histogram": float("nan"),
            "signal_label": "neutral",
            "fast_period": fast,
            "slow_period": slow,
            "signal_period": signal_period,
        }

    signal_line = _ema(valid_macd, signal_period)

    # Align signal line with macd_line
    offset = len(macd_line) - len(signal_line)
    padded_signal = [float("nan")] * offset + signal_line

    histogram = [
        (m - s) if (m == m and s == s) else float("nan")
        for m, s in zip(macd_line, padded_signal)
    ]

    macd_val = _last(macd_line)
    signal_val = _last(padded_signal)
    hist_val = _last(histogram)

    # Determine signal from histogram trend (last 2 values)
    recent_hist = [v for v in histogram if v == v]
    sig: Signal = "neutral"
    if len(recent_hist) >= 2:
        if recent_hist[-1] > recent_hist[-2] and recent_hist[-1] > 0:
            sig = "bullish"
        elif recent_hist[-1] < recent_hist[-2] and recent_hist[-1] < 0:
            sig = "bearish"
        elif recent_hist[-1] > 0:
            sig = "bullish"
        elif recent_hist[-1] < 0:
            sig = "bearish"

    return {
        "macd": round(macd_val, 6),
        "signal": round(signal_val, 6),
        "histogram": round(hist_val, 6),
        "signal_label": sig,
        "fast_period": fast,
        "slow_period": slow,
        "signal_period": signal_period,
    }


def calc_bollinger_bands(closes: List[float],
                         period: int = 20,
                         num_std: float = 2.0) -> Dict[str, Any]:
    """Compute Bollinger Bands (middle, upper, lower, %B, bandwidth).

    Returns:
        dict with band values, %B, bandwidth, and signal label.
    """
    n = len(closes)
    if n < period:
        return {
            "upper": float("nan"), "middle": float("nan"),
            "lower": float("nan"), "pct_b": float("nan"),
            "bandwidth": float("nan"), "signal": "neutral",
            "period": period, "num_std": num_std,
        }

    sma_vals = _sma(closes, period)
    middle = sma_vals[-1]

    # Standard deviation over last `period` values
    recent = closes[-period:]
    variance = sum((x - middle) ** 2 for x in recent) / period
    std = variance ** 0.5

    upper = middle + num_std * std
    lower = middle - num_std * std

    band_width = upper - lower
    pct_b = (closes[-1] - lower) / band_width if band_width > 0 else float("nan")

    # Signal: price near upper band → bearish (overbought),
    #         near lower band → bullish (oversold)
    signal: Signal = "neutral"
    if pct_b == pct_b:  # not NaN
        if pct_b >= 1.0:
            signal = "bearish"
        elif pct_b <= 0.0:
            signal = "bullish"
        elif pct_b >= 0.8:
            signal = "bearish"
        elif pct_b <= 0.2:
            signal = "bullish"

    return {
        "upper": round(upper, 6),
        "middle": round(middle, 6),
        "lower": round(lower, 6),
        "pct_b": round(pct_b, 6) if pct_b == pct_b else float("nan"),
        "bandwidth": round(band_width, 6),
        "signal": signal,
        "period": period,
        "num_std": num_std,
    }


def calc_stochastic(candles: List[Candle],
                    k_period: int = 14,
                    d_period: int = 3) -> Dict[str, Any]:
    """Compute Stochastic Oscillator (%K and %D).

    Returns:
        dict with k, d, signal.
    """
    n = len(candles)
    if n < k_period:
        return {
            "k": float("nan"), "d": float("nan"),
            "signal": "neutral", "k_period": k_period, "d_period": d_period,
        }

    k_values: List[float] = []
    for i in range(k_period - 1, n):
        highs = [candles[j]["high"] for j in range(i - k_period + 1, i + 1)]
        lows = [candles[j]["low"] for j in range(i - k_period + 1, i + 1)]
        hh = max(highs)
        ll = min(lows)
        close = candles[i]["close"]
        k_val = ((close - ll) / (hh - ll)) * 100.0 if hh != ll else 50.0
        k_values.append(k_val)

    # %D = SMA of %K
    d_values = _sma(k_values, d_period) if len(k_values) >= d_period else [float("nan")] * len(k_values)

    current_k = k_values[-1] if k_values else float("nan")
    current_d = _last(d_values) if d_values else float("nan")

    # Signal interpretation
    signal: Signal = "neutral"
    if current_k == current_k:  # not NaN
        if current_k >= 80:
            signal = "bearish"
        elif current_k <= 20:
            signal = "bullish"
        # Cross signals (need previous values)
        if len(k_values) >= 2 and len(d_values) >= 2:
            prev_k = k_values[-2]
            prev_d = _last(d_values[:-1]) if len(d_values) > 1 else d_values[-1]
            if prev_d == prev_d:
                if prev_k < prev_d and current_k > current_d:
                    signal = "bullish"
                elif prev_k > prev_d and current_k < current_d:
                    signal = "bearish"

    return {
        "k": round(current_k, 2),
        "d": round(current_d, 2) if current_d == current_d else float("nan"),
        "signal": signal,
        "k_period": k_period,
        "d_period": d_period,
    }


def calc_atr(candles: List[Candle], period: int = 14) -> Dict[str, Any]:
    """Compute Average True Range (ATR) — volatility measure.

    Returns:
        dict with atr value, signal (neutral always — ATR is descriptive), period.
    """
    n = len(candles)
    if n < period + 1:
        return {"value": float("nan"), "signal": "neutral", "period": period}

    true_ranges: List[float] = []
    for i in range(1, n):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_close = candles[i - 1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)

    # Wilder's smoothing for ATR
    atr = sum(true_ranges[:period]) / period
    for i in range(period, len(true_ranges)):
        atr = (atr * (period - 1) + true_ranges[i]) / period

    # Compare to recent average for context
    recent_avg_tr = sum(true_ranges[-3:]) / 3 if len(true_ranges) >= 3 else atr
    volatility = "expanding" if recent_avg_tr > atr else "contracting"

    return {
        "value": round(atr, 6),
        "signal": "neutral",
        "period": period,
        "volatility": volatility,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Candlestick pattern detection (last 3 candles)
# ═══════════════════════════════════════════════════════════════════════════

def detect_candlestick_patterns(candles: List[Candle]) -> Dict[str, Any]:
    """Detect common candlestick patterns on the last 3 candles.

    Patterns detected:
    - doji
    - hammer
    - bullish_engulfing
    - bearish_engulfing
    - morning_star
    - evening_star

    Returns:
        dict with each pattern name → bool, plus "patterns" list and
        overall "signal" label.
    """
    n = len(candles)
    result: Dict[str, Any] = {
        "doji": False,
        "hammer": False,
        "bullish_engulfing": False,
        "bearish_engulfing": False,
        "morning_star": False,
        "evening_star": False,
        "patterns": [],
        "signal": "neutral",
    }

    if n < 1:
        return result

    # Helper to compute body and shadows
    def _body(c: Candle) -> float:
        return abs(c["close"] - c["open"])

    def _upper_shadow(c: Candle) -> float:
        top = max(c["open"], c["close"])
        return c["high"] - top

    def _lower_shadow(c: Candle) -> float:
        bottom = min(c["open"], c["close"])
        return bottom - c["low"]

    def _full_range(c: Candle) -> float:
        return c["high"] - c["low"]

    def _is_bullish(c: Candle) -> bool:
        return c["close"] > c["open"]

    def _is_bearish(c: Candle) -> bool:
        return c["close"] < c["open"]

    # ── Doji (last candle) ───────────────────────────────────────────
    last = candles[-1]
    fr = _full_range(last)
    if fr > 0 and _body(last) / fr < 0.1:
        result["doji"] = True
        result["patterns"].append("doji")

    # ── Hammer (last candle) ────────────────────────────────────────
    # Small upper shadow, long lower shadow (≥2× body), small body
    if fr > 0:
        body = _body(last)
        ls = _lower_shadow(last)
        us = _upper_shadow(last)
        if ls >= 2 * body and us <= body * 0.3 and body > 0:
            result["hammer"] = True
            result["patterns"].append("hammer")

    # ── Engulfing patterns (need last 2 candles) ────────────────────
    if n >= 2:
        prev = candles[-2]
        # Bullish engulfing: prev bearish, last bullish, last body engulfs prev
        if _is_bearish(prev) and _is_bullish(last):
            if last["open"] <= prev["close"] and last["close"] >= prev["open"]:
                result["bullish_engulfing"] = True
                result["patterns"].append("bullish_engulfing")
        # Bearish engulfing: prev bullish, last bearish, last body engulfs prev
        if _is_bullish(prev) and _is_bearish(last):
            if last["open"] >= prev["close"] and last["close"] <= prev["open"]:
                result["bearish_engulfing"] = True
                result["patterns"].append("bearish_engulfing")

    # ── Morning / Evening star (need last 3 candles) ─────────────────
    if n >= 3:
        c1 = candles[-3]
        c2 = candles[-2]
        c3 = candles[-3]  # c3 not used; fix index
        c3 = candles[-1]

        body1 = _body(c1)
        body2 = _body(c2)
        body3 = _body(c3)

        # Morning star: large bearish → small body (gap) → large bullish
        if _is_bearish(c1) and body2 < body1 * 0.5 and _is_bullish(c3) and body3 > body2:
            # c3 close should recover into c1 body
            mid1 = (c1["open"] + c1["close"]) / 2
            if c3["close"] > mid1:
                result["morning_star"] = True
                result["patterns"].append("morning_star")

        # Evening star: large bullish → small body (gap) → large bearish
        if _is_bullish(c1) and body2 < body1 * 0.5 and _is_bearish(c3) and body3 > body2:
            mid1 = (c1["open"] + c1["close"]) / 2
            if c3["close"] < mid1:
                result["evening_star"] = True
                result["patterns"].append("evening_star")

    # Overall pattern signal
    bullish_patterns = sum([
        result["hammer"],
        result["bullish_engulfing"],
        result["morning_star"],
    ])
    bearish_patterns = sum([
        result["bearish_engulfing"],
        result["evening_star"],
    ])
    # Doji is neutral but can shift context
    if bullish_patterns > bearish_patterns:
        result["signal"] = "bullish"
    elif bearish_patterns > bullish_patterns:
        result["signal"] = "bearish"
    elif result["doji"]:
        result["signal"] = "neutral"  # indecision
    else:
        result["signal"] = "neutral"

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Support / Resistance (pivot-based)
# ═══════════════════════════════════════════════════════════════════════════

def calc_support_resistance(candles: List[Candle],
                           lookback: int = 20) -> Dict[str, Any]:
    """Compute support/resistance levels using pivot points and recent extremes.

    Uses classic pivot point math on the last completed candle, plus
    the highest high / lowest low over the lookback window.

    Returns:
        dict with pivot, supports (S1-S3), resistances (R1-R3),
        nearest_support, nearest_resistance, and signal.
    """
    n = len(candles)
    if n < 2:
        return {
            "pivot": float("nan"),
            "s1": float("nan"), "s2": float("nan"), "s3": float("nan"),
            "r1": float("nan"), "r2": float("nan"), "r3": float("nan"),
            "nearest_support": float("nan"), "nearest_resistance": float("nan"),
            "signal": "neutral",
        }

    # Use the second-to-last candle as the "completed" candle for pivot calc
    prev = candles[-2]
    h = prev["high"]
    l = prev["low"]
    c = prev["close"]

    pivot = (h + l + c) / 3.0
    s1 = 2 * pivot - h
    r1 = 2 * pivot - l
    s2 = pivot - (h - l)
    r2 = pivot + (h - l)
    s3 = l - 2 * (h - pivot)
    r3 = h + 2 * (pivot - l)

    current_price = candles[-1]["close"]

    # Find nearest support (below price) and resistance (above price)
    levels_below = sorted([lv for lv in [s1, s2, s3] if lv < current_price], reverse=True)
    levels_above = sorted([lv for lv in [r1, r2, r3] if lv > current_price])

    nearest_support = levels_below[0] if levels_below else min(s1, s2, s3)
    nearest_resistance = levels_above[0] if levels_above else max(r1, r2, r3)

    # Signal: price near support → bullish, near resistance → bearish
    price_range = nearest_resistance - nearest_support
    signal: Signal = "neutral"
    if price_range > 0 and price_range == price_range:
        pos = (current_price - nearest_support) / price_range
        if pos <= 0.2:
            signal = "bullish"
        elif pos >= 0.8:
            signal = "bearish"

    return {
        "pivot": round(pivot, 6),
        "s1": round(s1, 6), "s2": round(s2, 6), "s3": round(s3, 6),
        "r1": round(r1, 6), "r2": round(r2, 6), "r3": round(r3, 6),
        "nearest_support": round(nearest_support, 6),
        "nearest_resistance": round(nearest_resistance, 6),
        "signal": signal,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Volume Profile
# ═══════════════════════════════════════════════════════════════════════════

def calc_volume_profile(candles: List[Candle],
                       avg_period: int = 20,
                       recent_period: int = 3) -> Dict[str, Any]:
    """Compare recent volume to average volume.

    Returns:
        dict with avg_volume, recent_volume, ratio, and signal.
    """
    n = len(candles)
    if n < recent_period:
        return {
            "avg_volume": float("nan"), "recent_volume": float("nan"),
            "ratio": float("nan"), "signal": "neutral",
            "avg_period": avg_period, "recent_period": recent_period,
        }

    volumes = [c["volume"] for c in candles]

    # Average volume over lookback
    lookback = min(avg_period, n)
    avg_vol = sum(volumes[-lookback:]) / lookback

    # Recent volume
    recent = min(recent_period, n)
    recent_vol = sum(volumes[-recent:]) / recent

    ratio = recent_vol / avg_vol if avg_vol > 0 else float("nan")

    # Signal: volume spike = conviction in current price move
    # We need price direction context
    signal: Signal = "neutral"
    if ratio == ratio:  # not NaN
        if ratio >= 1.5:
            # High volume — direction depends on price trend
            recent_closes = [c["close"] for c in candles[-recent:]]
            if recent_closes[-1] > recent_closes[0]:
                signal = "bullish"  # high volume rally
            else:
                signal = "bearish"  # high volume sell-off
        elif ratio <= 0.5:
            signal = "neutral"  # low volume = low conviction

    return {
        "avg_volume": round(avg_vol, 2),
        "recent_volume": round(recent_vol, 2),
        "ratio": round(ratio, 4) if ratio == ratio else float("nan"),
        "signal": signal,
        "avg_period": lookback,
        "recent_period": recent,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Composite analysis per symbol
# ═══════════════════════════════════════════════════════════════════════════

def analyze_symbol(candles: List[Candle]) -> Dict[str, Any]:
    """Run all technical indicators on one symbol's candle list.

    Args:
        candles: list of OHLCV dicts sorted chronologically (oldest first).

    Returns:
        dict with keys for each indicator plus a composite "signal" vote.
    """
    closes = [c["close"] for c in candles]

    rsi = calc_rsi(closes, period=14)
    macd = calc_macd(closes)
    boll = calc_bollinger_bands(closes, period=20, num_std=2.0)
    stoch = calc_stochastic(candles, k_period=14, d_period=3)
    atr = calc_atr(candles, period=14)
    patterns = detect_candlestick_patterns(candles)
    sr = calc_support_resistance(candles)
    vol_profile = calc_volume_profile(candles)

    # Composite signal: majority vote
    signals = [
        rsi["signal"],
        macd["signal_label"],
        boll["signal"],
        stoch["signal"],
        patterns["signal"],
        sr["signal"],
        vol_profile["signal"],
    ]
    bull = sum(1 for s in signals if s == "bullish")
    bear = sum(1 for s in signals if s == "bearish")
    neut = sum(1 for s in signals if s == "neutral")

    if bull > bear and bull > neut:
        composite: Signal = "bullish"
    elif bear > bull and bear > neut:
        composite = "bearish"
    else:
        composite = "neutral"

    return {
        "rsi": rsi,
        "macd": macd,
        "bollinger": boll,
        "stochastic": stoch,
        "atr": atr,
        "patterns": patterns,
        "support_resistance": sr,
        "volume_profile": vol_profile,
        "composite_signal": composite,
        "bullish_count": bull,
        "bearish_count": bear,
        "neutral_count": neut,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Text summary generator (for LLM prompt injection)
# ═══════════════════════════════════════════════════════════════════════════

def _format_signal(sig: str) -> str:
    """Pretty-print signal label for LLM consumption."""
    return {"bullish": "▲ BULLISH", "bearish": "▼ BEARISH", "neutral": "— NEUTRAL"}.get(sig, sig)


def _build_symbol_summary(symbol: str, indicators: Dict[str, Any],
                          price: float) -> str:
    """Build a concise English summary for one symbol's technical analysis."""
    lines: List[str] = []
    lines.append(f"  {symbol} @ {price:.2f}")

    # Composite
    comp = indicators["composite_signal"]
    lines.append(f"    Overall: {_format_signal(comp)} "
                 f"(bull={indicators['bullish_count']}, "
                 f"bear={indicators['bearish_count']}, "
                 f"neut={indicators['neutral_count']})")

    # RSI
    rsi = indicators["rsi"]
    rsi_note = ""
    if rsi.get("overbought"):
        rsi_note = " [OVERBOUGHT]"
    elif rsi.get("oversold"):
        rsi_note = " [OVERSOLD]"
    lines.append(f"    RSI(14): {rsi['value']:.1f}{rsi_note} → {_format_signal(rsi['signal'])}")

    # MACD
    macd = indicators["macd"]
    hist_dir = "rising" if macd["histogram"] > 0 else "falling"
    lines.append(f"    MACD(12/26/9): {macd['macd']:.4f} / "
                 f"signal={macd['signal']:.4f} / "
                 f"hist={macd['histogram']:.4f} ({hist_dir}) → "
                 f"{_format_signal(macd['signal_label'])}")

    # Bollinger
    boll = indicators["bollinger"]
    pct_b = boll["pct_b"]
    pct_b_str = f"{pct_b:.2f}" if pct_b == pct_b else "N/A"
    lines.append(f"    Bollinger(20,2σ): "
                 f"upper={boll['upper']:.2f} mid={boll['middle']:.2f} "
                 f"lower={boll['lower']:.2f} %B={pct_b_str} → "
                 f"{_format_signal(boll['signal'])}")

    # Stochastic
    stoch = indicators["stochastic"]
    lines.append(f"    Stoch(14,3): %K={stoch['k']:.1f} %D={stoch['d']:.1f} → "
                 f"{_format_signal(stoch['signal'])}")

    # ATR
    atr = indicators["atr"]
    vol_desc = atr.get("volatility", "N/A")
    lines.append(f"    ATR(14): {atr['value']:.4f} (vol {vol_desc})")

    # Patterns
    pats = indicators["patterns"]
    pat_list = pats["patterns"]
    pat_str = ", ".join(pat_list) if pat_list else "none"
    lines.append(f"    Candlestick patterns: {pat_str} → {_format_signal(pats['signal'])}")

    # Support / Resistance
    sr = indicators["support_resistance"]
    lines.append(f"    Pivot={sr['pivot']:.2f} | "
                 f"S1={sr['s1']:.2f} S2={sr['s2']:.2f} | "
                 f"R1={sr['r1']:.2f} R2={sr['r2']:.2f} → "
                 f"{_format_signal(sr['signal'])}")

    # Volume profile
    vp = indicators["volume_profile"]
    ratio_str = f"{vp['ratio']:.2f}" if vp["ratio"] == vp["ratio"] else "N/A"
    lines.append(f"    Volume: avg={vp['avg_volume']:.0f} recent={vp['recent_volume']:.0f} "
                 f"ratio={ratio_str} → {_format_signal(vp['signal'])}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def generate_technical_summary(
    candles_dict: Dict[str, List[Dict[str, float]]],
) -> Tuple[str, Dict[str, Any]]:
    """Generate a technical analysis summary for all symbols.

    This is the primary entry point, designed to be called from brain.py
    to enrich the LLM prompt with technical indicator data.

    Args:
        candles_dict: Mapping of symbol key (e.g. "okx:BTC") to a list of
            OHLCV candle dicts. Each dict must have keys:
            "open", "high", "low", "close", "volume" (all float).
            Candles should be sorted chronologically (oldest first).

    Returns:
        A tuple of (summary_text, indicators_dict) where:
        - summary_text: concise English string ready for LLM prompt injection
        - indicators_dict: nested dict of all numeric values and signals,
          keyed by symbol, suitable for programmatic use.
    """
    all_indicators: Dict[str, Any] = {}
    summary_lines: List[str] = []

    summary_lines.append("TECHNICAL ANALYSIS:")
    summary_lines.append("─" * 40)

    for symbol, candles in candles_dict.items():
        if not candles or len(candles) < 15:
            summary_lines.append(f"  {symbol}: insufficient data ({len(candles)} candles, need ≥15)")
            all_indicators[symbol] = {"error": "insufficient_data", "candle_count": len(candles)}
            continue

        indicators = analyze_symbol(candles)
        all_indicators[symbol] = indicators

        current_price = candles[-1]["close"]
        summary_lines.append(_build_symbol_summary(symbol, indicators, current_price))

    # Overall market bias
    bull_count = sum(1 for v in all_indicators.values()
                     if isinstance(v, dict) and v.get("composite_signal") == "bullish")
    bear_count = sum(1 for v in all_indicators.values()
                     if isinstance(v, dict) and v.get("composite_signal") == "bearish")
    total = sum(1 for v in all_indicators.values()
                if isinstance(v, dict) and "composite_signal" in v)

    summary_lines.append("─" * 40)
    if total > 0:
        if bull_count > bear_count:
            summary_lines.append(f"Market bias: BULLISH ({bull_count}/{total} symbols bullish)")
        elif bear_count > bull_count:
            summary_lines.append(f"Market bias: BEARISH ({bear_count}/{total} symbols bearish)")
        else:
            summary_lines.append(f"Market bias: NEUTRAL ({bull_count}B/{bear_count}B/{total} total)")
    else:
        summary_lines.append("Market bias: INSUFFICIENT DATA")

    summary_text = "\n".join(summary_lines)
    return summary_text, all_indicators