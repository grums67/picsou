"""Grid trading strategy for ranging markets."""

import logging
from typing import Any, Dict, List

from .base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


def calculate_atr(candles: List[Dict[str, Any]], period: int = 14) -> List[float]:
    """Calculate Average True Range.

    True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    ATR is the SMA of True Range.
    """
    if len(candles) < period + 1:
        return [0.0] * len(candles)

    true_ranges: List[float] = [0.0] * len(candles)

    for i in range(1, len(candles)):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_close = candles[i - 1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges[i] = tr

    # Calculate ATR as SMA of true ranges
    atr = [0.0] * len(candles)
    for i in range(period, len(candles)):
        atr[i] = sum(true_ranges[i - period + 1:i + 1]) / period

    return atr


def calculate_adx(candles: List[Dict[str, Any]], period: int = 14) -> List[float]:
    """Simplified ADX calculation for trend detection.

    Returns ADX values. Higher ADX = stronger trend.
    Values above 25 generally indicate a trending market.
    """
    if len(candles) < period * 2 + 1:
        return [0.0] * len(candles)

    # Calculate directional movement
    plus_dm: List[float] = [0.0] * len(candles)
    minus_dm: List[float] = [0.0] * len(candles)
    tr: List[float] = [0.0] * len(candles)

    for i in range(1, len(candles)):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_high = candles[i - 1]["high"]
        prev_low = candles[i - 1]["low"]
        prev_close = candles[i - 1]["close"]

        up_move = high - prev_high
        down_move = prev_low - low

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

        tr[i] = max(high - low, abs(high - prev_close), abs(low - prev_close))

    # Smooth with Wilder's method
    atr = [0.0] * len(candles)
    smooth_plus_dm = [0.0] * len(candles)
    smooth_minus_dm = [0.0] * len(candles)

    # Initial values
    atr[period] = sum(tr[1:period + 1]) / period
    smooth_plus_dm[period] = sum(plus_dm[1:period + 1]) / period
    smooth_minus_dm[period] = sum(minus_dm[1:period + 1]) / period

    for i in range(period + 1, len(candles)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        smooth_plus_dm[i] = (smooth_plus_dm[i - 1] * (period - 1) + plus_dm[i]) / period
        smooth_minus_dm[i] = (smooth_minus_dm[i - 1] * (period - 1) + minus_dm[i]) / period

    # Calculate DI and DX
    adx = [0.0] * len(candles)
    dx_values: List[float] = [0.0] * len(candles)

    for i in range(period, len(candles)):
        if atr[i] > 0:
            plus_di = 100 * smooth_plus_dm[i] / atr[i]
            minus_di = 100 * smooth_minus_dm[i] / atr[i]
            di_sum = plus_di + minus_di
            if di_sum > 0:
                dx_values[i] = 100 * abs(plus_di - minus_di) / di_sum

    # ADX is smoothed DX
    adx_start = period * 2
    if len(candles) > adx_start:
        adx[adx_start] = sum(dx_values[period:adx_start]) / period
        for i in range(adx_start + 1, len(candles)):
            adx[i] = (adx[i - 1] * (period - 1) + dx_values[i]) / period

    return adx


class GridStrategy(BaseStrategy):
    """Grid trading strategy for ranging markets.

    - Divides price range into N levels
    - Buys at lower levels, sells at upper levels
    - Only activates in ranging (low ADX) markets
    """

    def __init__(self, num_levels: int = 10, atr_period: int = 14,
                 adx_threshold: float = 25.0,
                 grid_range_atr_multiplier: float = 1.5) -> None:
        super().__init__(name="grid")
        self.num_levels = num_levels
        self.atr_period = atr_period
        self.adx_threshold = adx_threshold
        self.grid_range_atr_multiplier = grid_range_atr_multiplier

    def analyze(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Analyze market for grid trading opportunities."""
        candles = market_data.get("candles", [])
        symbol = market_data.get("symbol", "UNKNOWN")
        exchange = market_data.get("exchange", "unknown")

        if not self._validate_candles(candles, min_count=self.atr_period * 2 + 5):
            return []

        # Calculate ATR and ADX
        atr_values = calculate_atr(candles, self.atr_period)
        adx_values = calculate_adx(candles, self.atr_period)

        curr_price = candles[-1]["close"]
        curr_atr = atr_values[-1] if atr_values[-1] > 0 else 0.0
        curr_adx = adx_values[-1]

        if curr_atr == 0:
            return []

        # Only trade in ranging markets (ADX below threshold)
        if curr_adx > self.adx_threshold:
            logger.debug(
                "Grid: skipping %s - trending market (ADX=%.1f > %.1f)",
                symbol, curr_adx, self.adx_threshold,
            )
            return []

        # Define grid range based on ATR
        grid_range = curr_atr * self.grid_range_atr_multiplier
        grid_center = curr_price
        grid_upper = grid_center + grid_range
        grid_lower = grid_center - grid_range

        # Calculate grid levels
        level_size = (grid_upper - grid_lower) / self.num_levels
        levels = [grid_lower + i * level_size for i in range(self.num_levels + 1)]

        # Find current position in grid
        if curr_price <= grid_lower:
            price_position = 0.0
        elif curr_price >= grid_upper:
            price_position = 1.0
        else:
            price_position = (curr_price - grid_lower) / (grid_upper - grid_lower)

        # Find nearest grid levels
        level_index = int(price_position * self.num_levels)
        level_index = max(0, min(level_index, self.num_levels - 1))

        signals: List[Signal] = []

        # Buy signal: price is in lower half of grid (near support levels)
        if price_position < 0.4:
            # Closer to bottom = stronger buy signal
            confidence = min(0.3 + (0.4 - price_position) * 1.2, 0.8)

            target_level = levels[max(0, level_index)]
            reasoning = (
                f"Grid buy: price at {price_position:.1%} of range "
                f"(price={curr_price:.2f}, range={grid_lower:.2f}-{grid_upper:.2f}, "
                f"ATR={curr_atr:.2f}, ADX={curr_adx:.1f})"
            )
            signals.append(Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                reasoning=reasoning,
                strategy_name=self.name,
                metadata={
                    "exchange": exchange,
                    "price": curr_price,
                    "grid_lower": round(grid_lower, 2),
                    "grid_upper": round(grid_upper, 2),
                    "grid_center": round(grid_center, 2),
                    "price_position": round(price_position, 4),
                    "atr": round(curr_atr, 2),
                    "adx": round(curr_adx, 2),
                    "target_level": round(target_level, 2),
                    "num_levels": self.num_levels,
                },
            ))

        # Sell signal: price is in upper half of grid (near resistance levels)
        elif price_position > 0.6:
            confidence = min(0.3 + (price_position - 0.6) * 1.2, 0.8)

            target_level = levels[min(self.num_levels, level_index + 1)]
            reasoning = (
                f"Grid sell: price at {price_position:.1%} of range "
                f"(price={curr_price:.2f}, range={grid_lower:.2f}-{grid_upper:.2f}, "
                f"ATR={curr_atr:.2f}, ADX={curr_adx:.1f})"
            )
            signals.append(Signal(
                symbol=symbol,
                side="sell",
                confidence=confidence,
                reasoning=reasoning,
                strategy_name=self.name,
                metadata={
                    "exchange": exchange,
                    "price": curr_price,
                    "grid_lower": round(grid_lower, 2),
                    "grid_upper": round(grid_upper, 2),
                    "grid_center": round(grid_center, 2),
                    "price_position": round(price_position, 4),
                    "atr": round(curr_atr, 2),
                    "adx": round(curr_adx, 2),
                    "target_level": round(target_level, 2),
                    "num_levels": self.num_levels,
                },
            ))

        return signals