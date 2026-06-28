"""Mean reversion strategy using Bollinger Bands and RSI."""

import logging
from typing import Any, Dict, List

from .base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


def calculate_sma(prices: List[float], period: int) -> List[float]:
    """Calculate Simple Moving Average."""
    result = [0.0] * len(prices)
    for i in range(period - 1, len(prices)):
        result[i] = sum(prices[i - period + 1:i + 1]) / period
    return result


def calculate_std(prices: List[float], period: int) -> List[float]:
    """Calculate rolling standard deviation."""
    result = [0.0] * len(prices)
    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1:i + 1]
        mean = sum(window) / period
        variance = sum((p - mean) ** 2 for p in window) / period
        result[i] = variance ** 0.5
    return result


def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """Calculate Relative Strength Index.

    Returns:
        List of RSI values. First values are 0.0 until enough data.
    """
    if len(prices) < period + 1:
        return [50.0] * len(prices)

    result = [0.0] * len(prices)
    gains = [0.0] * len(prices)
    losses = [0.0] * len(prices)

    # Calculate price changes
    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains[i] = change
            losses[i] = 0.0
        else:
            gains[i] = 0.0
            losses[i] = abs(change)

    # First RSI: use SMA of gains/losses
    avg_gain = sum(gains[1:period + 1]) / period
    avg_loss = sum(losses[1:period + 1]) / period

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))

    # Subsequent RSI values use smoothed averages
    for i in range(period + 1, len(prices)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))

    return result


def calculate_bollinger_bands(prices: List[float], period: int = 20,
                               num_std: float = 2.0) -> Dict[str, List[float]]:
    """Calculate Bollinger Bands.

    Returns dict with keys: middle, upper, lower.
    """
    sma = calculate_sma(prices, period)
    std = calculate_std(prices, period)

    upper = [0.0] * len(prices)
    lower = [0.0] * len(prices)

    for i in range(len(prices)):
        if sma[i] > 0:
            upper[i] = sma[i] + num_std * std[i]
            lower[i] = sma[i] - num_std * std[i]

    return {"middle": sma, "upper": upper, "lower": lower}


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using Bollinger Bands and RSI.

    - Buy when price touches lower band AND RSI < 30
    - Sell when price reaches middle band OR RSI > 70
    - Confidence based on distance from mean
    """

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0,
                 rsi_period: int = 14, rsi_oversold: float = 30.0,
                 rsi_overbought: float = 70.0) -> None:
        super().__init__(name="mean_reversion")
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def analyze(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Analyze candles for mean reversion signals."""
        candles = market_data.get("candles", [])
        symbol = market_data.get("symbol", "UNKNOWN")
        exchange = market_data.get("exchange", "unknown")

        min_candles = max(self.bb_period, self.rsi_period) + 5
        if not self._validate_candles(candles, min_count=min_candles):
            return []

        closes = [c["close"] for c in candles]

        # Calculate indicators
        bb = calculate_bollinger_bands(closes, self.bb_period, self.bb_std)
        rsi = calculate_rsi(closes, self.rsi_period)

        signals: List[Signal] = []
        n = len(closes)
        curr_price = closes[-1]
        curr_rsi = rsi[-1] if rsi[-1] > 0 else 50.0
        curr_upper = bb["upper"][-1]
        curr_lower = bb["lower"][-1]
        curr_middle = bb["middle"][-1]

        if curr_upper == 0.0 or curr_lower == 0.0 or curr_middle == 0.0:
            return []

        # Calculate distance from middle band as percentage
        bb_width = curr_upper - curr_lower
        price_position = (curr_price - curr_lower) / bb_width if bb_width > 0 else 0.5

        # Buy signal: price at/near lower band + RSI oversold
        if curr_price <= curr_lower * 1.005 and curr_rsi < self.rsi_oversold:
            # Confidence increases with how oversold we are
            rsi_confidence = (self.rsi_oversold - curr_rsi) / self.rsi_oversold
            bb_confidence = max(0, 1.0 - price_position)  # Closer to 0 = more oversold
            confidence = min(0.4 + rsi_confidence * 0.3 + bb_confidence * 0.2, 0.95)

            reasoning = (
                f"Price near lower BB ({curr_price:.2f} vs {curr_lower:.2f}) "
                f"with RSI={curr_rsi:.1f} (oversold)"
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
                    "rsi": round(curr_rsi, 2),
                    "bb_lower": round(curr_lower, 2),
                    "bb_upper": round(curr_upper, 2),
                    "bb_middle": round(curr_middle, 2),
                    "price_position": round(price_position, 4),
                },
            ))

        # Sell signal: price near upper band + RSI overbought
        elif curr_price >= curr_upper * 0.995 and curr_rsi > self.rsi_overbought:
            rsi_confidence = (curr_rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            bb_confidence = price_position  # Closer to 1 = more overbought
            confidence = min(0.4 + rsi_confidence * 0.3 + bb_confidence * 0.2, 0.95)

            reasoning = (
                f"Price near upper BB ({curr_price:.2f} vs {curr_upper:.2f}) "
                f"with RSI={curr_rsi:.1f} (overbought)"
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
                    "rsi": round(curr_rsi, 2),
                    "bb_lower": round(curr_lower, 2),
                    "bb_upper": round(curr_upper, 2),
                    "bb_middle": round(curr_middle, 2),
                    "price_position": round(price_position, 4),
                },
            ))

        # Moderate signals: RSI extreme without BB confirmation
        elif curr_rsi < 25:  # Deeply oversold
            confidence = min(0.25 + (30 - curr_rsi) / 100, 0.6)
            signals.append(Signal(
                symbol=symbol,
                side="buy",
                confidence=confidence,
                reasoning=f"RSI deeply oversold at {curr_rsi:.1f}",
                strategy_name=self.name,
                metadata={"exchange": exchange, "rsi": round(curr_rsi, 2)},
            ))
        elif curr_rsi > 75:  # Deeply overbought
            confidence = min(0.25 + (curr_rsi - 70) / 100, 0.6)
            signals.append(Signal(
                symbol=symbol,
                side="sell",
                confidence=confidence,
                reasoning=f"RSI deeply overbought at {curr_rsi:.1f}",
                strategy_name=self.name,
                metadata={"exchange": exchange, "rsi": round(curr_rsi, 2)},
            ))

        return signals