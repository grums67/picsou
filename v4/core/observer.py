"""Picsou v4 — Market observer.

Collects market data from exchanges: prices, candles, sentiment.
This is the OBSERVE step — deterministic, no LLM.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import requests

from .config import PicsouConfig

logger = logging.getLogger(__name__)


class Observer:
    """Collects market data, sentiment, and portfolio state."""

    def __init__(self, config: PicsouConfig, exchanges: Dict):
        self.config = config
        self.exchanges = exchanges
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 60  # seconds
        self._last_fetch = 0

    def fetch_market_data(self) -> Dict[str, Dict]:
        """Fetch current market data for all configured symbols.

        Returns dict keyed by "EXCHANGE:BASE" with candles + ticker.
        """
        data = {}
        for exc_name, exchange in self.exchanges.items():
            for symbol in self.config.symbols:
                key = f"{exc_name}:{symbol}"
                try:
                    formatted = exchange.format_symbol(symbol)
                    ticker = exchange.get_ticker(formatted)
                    candles = exchange.get_candles(formatted, self.config.candle_interval, limit=100)

                    data[key] = {
                        "exchange": exc_name,
                        "symbol": symbol,
                        "formatted_symbol": formatted,
                        "ticker": ticker,
                        "candles": candles or [],
                    }
                except Exception as e:
                    logger.warning("Failed to fetch %s: %s", key, e)

        logger.info("Fetched market data for %d pairs", len(data))
        return data

    def fetch_sentiment(self) -> Dict[str, Any]:
        """Fetch Fear & Greed Index and basic sentiment data."""
        sentiment = {"fear_and_greed": {}, "headlines": []}

        # Fear & Greed Index
        try:
            resp = requests.get(
                "https://api.alternative.me/fng/?limit=1",
                timeout=10,
            )
            if resp.status_code == 200:
                fng = resp.json().get("data", [{}])[0]
                sentiment["fear_and_greed"] = {
                    "value": int(fng.get("value", 0)),
                    "classification": fng.get("value_classification", "Unknown"),
                }
                logger.info("Fear & Greed: %s (%s)", fng.get("value"), fng.get("value_classification"))
        except Exception as e:
            logger.warning("Failed to fetch F&G: %s", e)

        return sentiment

    def build_context(self, market_data: Dict, portfolio_state: Dict,
                      memory_context: Dict) -> Dict[str, Any]:
        """Build the full context dict for the LLM brain.

        This is what gets sent to the LLM every brain cycle.
        """
        # Compact market summary
        market_summary = {}
        for key, md in market_data.items():
            ticker = md.get("ticker", {})
            candles = md.get("candles", [])
            last_price = ticker.get("last", 0) if ticker else 0
            vol_24h = ticker.get("volume_24h", 0) if ticker else 0
            change_24h = ticker.get("change_24h", 0) if ticker else 0

            market_summary[key] = {
                "price": last_price,
                "volume_24h": vol_24h,
                "change_24h": change_24h,
                "candles_count": len(candles),
                "last_5_candles": candles[-5:] if len(candles) >= 5 else candles,
            }

        return {
            "market": market_summary,
            "sentiment": self.fetch_sentiment(),
            "portfolio": portfolio_state,
            "memory": memory_context,
            "symbols": self.config.symbols,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }