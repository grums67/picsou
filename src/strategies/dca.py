"""Dollar Cost Averaging (DCA) strategy with martingale-lite enhancement."""

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


def calculate_price_change(prices: List[float]) -> List[float]:
    """Calculate percentage change between current and N periods ago."""
    result = [0.0] * len(prices)
    for i in range(1, len(prices)):
        if prices[i - 1] > 0:
            result[i] = (prices[i] - prices[i - 1]) / prices[i - 1]
    return result


class DCAStrategy(BaseStrategy):
    """Dollar Cost Averaging strategy with martingale-lite enhancement.

    - Buy fixed amount at regular intervals
    - Increase buy size when price drops (martingale lite)
    - Sell when portfolio is up X% from average entry
    """

    def __init__(self, buy_interval_hours: int = 4,
                 base_buy_pct: float = 0.02,
                 drop_multiplier: float = 1.5,
                 drop_threshold: float = -0.03,
                 take_profit_pct: float = 0.10,
                 sma_period: int = 24) -> None:
        """Initialize DCA strategy.

        Args:
            buy_interval_hours: Minimum hours between DCA buys.
            base_buy_pct: Base buy as % of capital (0.02 = 2%).
            drop_multiplier: Multiplier for buy size when price drops.
            drop_threshold: Price drop % to trigger increased buy (-0.03 = -3%).
            take_profit_pct: Take profit when up this % from avg entry (0.10 = 10%).
            sma_period: SMA period for trend detection.
        """
        super().__init__(name="dca")
        self.buy_interval_hours = buy_interval_hours
        self.base_buy_pct = base_buy_pct
        self.drop_multiplier = drop_multiplier
        self.drop_threshold = drop_threshold
        self.take_profit_pct = take_profit_pct
        self.sma_period = sma_period

    def analyze(self, market_data: Dict[str, Any]) -> List[Signal]:
        """Analyze market for DCA signals."""
        candles = market_data.get("candles", [])
        symbol = market_data.get("symbol", "UNKNOWN")
        exchange = market_data.get("exchange", "unknown")

        if not self._validate_candles(candles, min_count=self.sma_period + 5):
            return []

        closes = [c["close"] for c in candles]
        volumes = [c["volume"] for c in candles]

        curr_price = closes[-1]

        # Calculate SMA for trend reference
        sma = calculate_sma(closes, self.sma_period)
        curr_sma = sma[-1] if sma[-1] > 0 else curr_price

        # Calculate recent price change (drop detection)
        # Use 24h (or available) percentage change
        lookback = min(24, len(closes) - 1)
        if lookback > 0 and closes[-1 - lookback] > 0:
            price_change_24h = (curr_price - closes[-1 - lookback]) / closes[-1 - lookback]
        else:
            price_change_24h = 0.0

        # Shorter term change (4h or available)
        short_lookback = min(4, len(closes) - 1)
        if short_lookback > 0 and closes[-1 - short_lookback] > 0:
            price_change_4h = (curr_price - closes[-1 - short_lookback]) / closes[-1 - short_lookback]
        else:
            price_change_4h = 0.0

        # Price vs SMA
        price_vs_sma = (curr_price - curr_sma) / curr_sma if curr_sma > 0 else 0.0

        # Volume analysis (is this a high-volume move?)
        vol_sma = calculate_sma(volumes, self.sma_period) if volumes else [0.0] * len(volumes)
        vol_ratio = volumes[-1] / vol_sma[-1] if vol_sma[-1] > 0 else 1.0

        signals: List[Signal] = []

        # BUY SIGNAL: Price has dropped
        if price_change_24h < self.drop_threshold:
            # Increased buy - martingale lite
            drop_severity = abs(price_change_24h) / abs(self.drop_threshold)
            buy_multiplier = min(self.drop_multiplier, 1.0 + drop_severity)
            confidence = min(0.4 + drop_severity * 0.2, 0.85)

            reasoning = (
                f"DCA buy (enhanced): price dropped {price_change_24h*100:.1f}% in 24h "
                f"(multiplier={buy_multiplier:.1f}x, vs SMA={price_vs_sma*100:.1f}%)"
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
                    "price_change_24h": round(price_change_24h, 4),
                    "price_change_4h": round(price_change_4h, 4),
                    "price_vs_sma": round(price_vs_sma, 4),
                    "buy_multiplier": round(buy_multiplier, 2),
                    "buy_pct": round(self.base_buy_pct * buy_multiplier, 4),
                    "vol_ratio": round(vol_ratio, 2),
                },
            ))

        # Regular DCA buy (smaller, periodic)
        elif price_vs_sma < -0.02:  # Below SMA by 2%+
            confidence = min(0.25 + abs(price_vs_sma) * 5, 0.55)
            reasoning = (
                f"DCA buy (regular): price {price_vs_sma*100:.1f}% below SMA "
                f"(change_24h={price_change_24h*100:.1f}%)"
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
                    "price_vs_sma": round(price_vs_sma, 4),
                    "buy_pct": round(self.base_buy_pct, 4),
                    "vol_ratio": round(vol_ratio, 2),
                },
            ))

        # SELL SIGNAL: Significant profit above SMA
        elif price_vs_sma > self.take_profit_pct:
            confidence = min(0.4 + price_vs_sma * 2, 0.85)
            reasoning = (
                f"DCA sell: price {price_vs_sma*100:.1f}% above SMA "
                f"(take_profit={self.take_profit_pct*100:.0f}%)"
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
                    "price_vs_sma": round(price_vs_sma, 4),
                    "profit_pct": round(price_vs_sma * 100, 2),
                },
            ))

        # Moderate sell: well above SMA but not at take profit
        elif price_vs_sma > 0.05:
            confidence = 0.3
            reasoning = (
                f"DCA partial sell: price {price_vs_sma*100:.1f}% above SMA"
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
                    "price_vs_sma": round(price_vs_sma, 4),
                },
            ))

        return signals