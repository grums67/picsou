"""Kraken exchange adapter for Picsou."""

import logging
from typing import Any, Dict, List

import requests

from .base import BaseExchange

logger = logging.getLogger(__name__)

# Kraken OHLC interval in minutes
INTERVAL_MAP = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}

# Kraken uses XBT instead of BTC
SYMBOL_ALIASES = {"BTC": "XBT"}


class KrakenExchange(BaseExchange):
    """Kraken exchange adapter using public REST API (no auth required)."""

    def __init__(self, rest_url: str = "https://api.kraken.com/0/public",
                 fee_rate: float = 0.0026) -> None:
        super().__init__(
            name="kraken",
            rest_url=rest_url,
            fee_rate=fee_rate,
            symbol_format="{base}USDT",
        )
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Picsou/1.0",
            "Accept": "application/json",
        })

    def _kraken_symbol(self, symbol: str) -> str:
        """Convert to Kraken symbol format. BTC-USDT -> BTCUSDT (query) -> XBTUSDT (internal key).
        Kraken API query uses BTCUSDT but returns XBTUSDT as key."""
        base = symbol.upper().replace("-", "").replace("USDT", "").replace("USD", "")
        if not base:
            base = symbol.upper().replace("-", "")
            return base
        # Kraken uses XBT internally but BTC in query params
        # Query with BTC, API returns XBT key
        return f"{base}USDT"

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker from Kraken. Query format: BTCUSDT."""
        # Normalize: accept BTC-USDT, BTC, btcusdt etc -> BTCUSDT
        pair = self._kraken_symbol(symbol)
        url = f"{self.rest_url}/Ticker"
        params = {"pair": pair}
        try:
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("error"):
                logger.warning("Kraken ticker error for %s: %s", pair, data["error"])
                return {}
            result = data.get("result", {})
            # Find the pair key (Kraken uses XXBTZUSDT etc.)
            pair_key = None
            for k in result:
                if k != "last" and not k.startswith("."):
                    pair_key = k
                    break
            if not pair_key:
                logger.warning("Kraken ticker: no pair data for %s", pair)
                return {}
            t = result[pair_key]
            close_arr = t.get("c", ["0", "0"])
            return {
                "symbol": pair,
                "bid": float(t.get("b", ["0"])[0]),
                "ask": float(t.get("a", ["0"])[0]),
                "last": float(close_arr[0]),
                "volume": float(t.get("v", ["0", "0"])[1]),
                "high_24h": float(t.get("h", ["0", "0"])[1]),
                "low_24h": float(t.get("l", ["0", "0"])[1]),
            }
        except Exception as e:
            logger.error("Kraken get_ticker failed for %s: %s", pair, e)
            return {}

    def get_candles(self, symbol: str, interval: str = "1h",
                    limit: int = 100) -> List[Dict[str, Any]]:
        """Get OHLC candles from Kraken. Returns oldest first."""
        pair = self._kraken_symbol(symbol)
        minutes = INTERVAL_MAP.get(interval, 60)
        url = f"{self.rest_url}/OHLC"
        params = {"pair": pair, "interval": minutes}
        try:
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("error"):
                logger.warning("Kraken OHLC error for %s: %s", pair, data["error"])
                return []
            result = data.get("result", {})
            # Find the pair key
            pair_key = None
            for k in result:
                if k != "last" and not k.startswith("."):
                    pair_key = k
                    break
            if not pair_key:
                return []
            candles_raw = result[pair_key]
            # Kraken returns oldest first already, but limit to requested count
            candles = []
            for c in candles_raw[-limit:]:
                candles.append({
                    "timestamp": int(c[0]),
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5]),
                })
            return candles
        except Exception as e:
            logger.error("Kraken get_candles failed for %s: %s", pair, e)
            return []

    def get_order_book(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """Get order book from Kraken."""
        pair = self._kraken_symbol(symbol)
        url = f"{self.rest_url}/Depth"
        params = {"pair": pair, "count": depth}
        try:
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("error"):
                logger.warning("Kraken Depth error for %s: %s", pair, data["error"])
                return {"bids": [], "asks": []}
            result = data.get("result", {})
            pair_key = None
            for k in result:
                if k != "last" and not k.startswith("."):
                    pair_key = k
                    break
            if not pair_key:
                return {"bids": [], "asks": []}
            book = result[pair_key]
            bids = [[float(p[0]), float(p[1])] for p in book.get("bids", [])]
            asks = [[float(p[0]), float(p[1])] for p in book.get("asks", [])]
            return {"bids": bids, "asks": asks}
        except Exception as e:
            logger.error("Kraken get_order_book failed for %s: %s", pair, e)
            return {"bids": [], "asks": []}