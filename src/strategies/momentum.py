"""Momentum strategy using EMA crossover with volume confirmation."""

import logging
from typing import Any, Dict, List

from .base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


def calculate_ema(prices: List[float], period: int) -> List[float]:
    """Calculate Exponential Moving Average.

    Args:
        prices: List of closing prices (oldest first).
        period: EMA period.

    Returns:
        List of EMA values, same length as prices.
        First values will be NaN-like (0.0) until enough data.
    """
    if not prices or len(prices) < period:
        return [0.0] * len(prices)

    ema = [0.0] * len(prices)
    multiplier = 2.0 / (period + 1)

    # Seed EMA with SMA of first 'period' values
    sma = sum(prices[:period]) / period
    ema[period - 1] = sma

    # Calculate EMA from there
    for i in range(period, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1]

    return ema


def calculate_volume_sma(volumes: List[float], period: int) -> List[float]:
    """Calculate Simple Moving Average of volume."""
    result = [0.0] * len(volumes)
    for i in range(period - 1, len(volumes)):
        result[i] = sum(volumes[i - period + 1:i + 1]) / period
    return result


class MomentumStrategy(BaseStrategy):
    """Momentum strategy using EMA crossover.

    - Buy signal: EMA9 crosses above EMA21 with volume confirmation
    - Sell signal: EMA9 crosses below EMA21
    - Confidence based on crossover strength + volume
    """

    def __init__(self, fast_period: int = 9, slow_period: int = 21,
                 volume_period: int = 20) -> None:
        super().__init__(name="momentum")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.volume_period = volume_period

    def analyze(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Analyze candles for EMA crossover signals."""
        candles = market_data.get("candles", [])
        symbol = market_data.get("symbol", "UNKNOWN")
        exchange = market_data.get("exchange", "unknown")

        if not self._validate_candles(candles, min_count=self.slow_period + 5):
            return []

        # Extract price and volume series (oldest first)
        closes = [c["close"] for c in candles]
        volumes = [c["volume"] for c in candles]

        # Calculate EMAs
        ema_fast = calculate_ema(closes, self.fast_period)
        ema_slow = calculate_ema(closes, self.slow_period)

        # Calculate volume SMA for confirmation
        vol_sma = calculate_volume_sma(volumes, self.volume_period)

        signals: List[Signal] = []

        # Need at least 2 data points with valid EMAs to detect crossover
        # Find the last two valid EMA points
        n = len(closes)

        # Check if we have valid crossover data
        # Find latest index where both EMAs are valid
        valid_start = max(self.fast_period, self.slow_period)
        if n < valid_start + 2:
            return []

        # Current and previous EMA values
        curr_fast = ema_fast[n - 1]
        curr_slow = ema_slow[n - 1]
        prev_fast = ema_fast[n - 2]
        prev_slow = ema_slow[n - 2]

        if curr_fast == 0.0 or curr_slow == 0.0:
            return []

        curr_price = closes[-1]

        # Volume confirmation: current volume > SMA volume
        curr_vol = volumes[-1] if volumes else 0
        avg_vol = vol_sma[-1] if vol_sma[-1] > 0 else 1.0
        volume_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0

        # Detect crossovers
        # Bullish: fast crosses above slow
        bullish_cross = prev_fast <= prev_slow and curr_fast > curr_slow
        # Bearish: fast crosses below slow
        bearish_cross = prev_fast >= prev_slow and curr_fast < curr_slow

        # Crossover strength: distance between EMAs relative to price
        separation = abs(curr_fast - curr_slow) / curr_price if curr_price > 0 else 0

        if bullish_cross:
            # Volume confirmation boosts confidence
            vol_boost = min(volume_ratio, 2.0) / 2.0  # 0 to 1
            confidence = min(0.3 + separation * 50 + vol_boost * 0.3, 0.95)

            reasoning = (
                f"EMA{self.fast_period} crossed above EMA{self.slow_period} "
                f"(separation={separation:.4f}, vol_ratio={volume_ratio:.2f})"
            )
            signals.append(Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                reasoning=reasoning,
                strategy_name=self.name,
                metadata={
                    "exchange": exchange,
                    "ema_fast": round(curr_fast, 2),
                    "ema_slow": round(curr_slow, 2),
                    "separation": round(separation, 6),
                    "volume_ratio": round(volume_ratio, 2),
                    "price": curr_price,
                },
            ))

        elif bearish_cross:
            confidence = min(0.3 + separation * 50, 0.90)

            reasoning = (
                f"EMA{self.fast_period} crossed below EMA{self.slow_period} "
                f"(separation={separation:.4f})"
            )
            signals.append(Signal(
                symbol=symbol,
                side="sell",
                confidence=confidence,
                reasoning=reasoning,
                strategy_name=self.name,
                metadata={
                    "exchange": exchange,
                    "ema_fast": round(curr_fast, 2),
                    "ema_slow": round(curr_slow, 2),
                    "separation": round(separation, 6),
                    "price": curr_price,
                },
            ))

        # Also check trend direction for weaker signals
        # If fast EMA is significantly above/below slow (no crossover, but trending)
        elif curr_fast > curr_slow:
            # Uptrend - hold or mild buy
            trend_strength = (curr_fast - curr_slow) / curr_price if curr_price > 0 else 0
            if trend_strength > 0.01:  # More than 1% trend
                signals.append(Signal(
                    symbol=symbol,
                    side="buy",
                    confidence=min(0.2 + trend_strength * 10, 0.5),  # Weak signal
                    reasoning=f"Uptrend: EMA{self.fast_period} > EMA{self.slow_period} "
                              f"by {trend_strength*100:.2f}%",
                    strategy_name=self.name,
                    metadata={"exchange": exchange, "trend": "up",
                              "strength": round(trend_strength, 4)},
                ))

        elif curr_fast < curr_slow:
            trend_strength = (curr_slow - curr_fast) / curr_price if curr_price > 0 else 0
            if trend_strength > 0.01:
                signals.append(Signal(
                    symbol=symbol,
                    side="sell",
                    confidence=min(0.2 + trend_strength * 10, 0.5),
                    reasoning=f"Downtrend: EMA{self.fast_period} < EMA{self.slow_period} "
                              f"by {trend_strength*100:.2f}%",
                    strategy_name=self.name,
                    metadata={"exchange": exchange, "trend": "down",
                              "strength": round(trend_strength, 4)},
                ))

        return signals