"""Bitstamp exchange adapter for Picsou."""

import logging
from typing import Any, Dict, List

import requests

from .base import BaseExchange

logger = logging.getLogger(__name__)

# Bitstamp OHLC interval in seconds
INTERVAL_MAP = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

# Bitstamp doesn't have USDT pairs for all coins — fallback to USDC or USD
PAIR_FALLBACKS = {
    "solusdt": "solusdc",
    "adausdt": "adausd",
    "dotusdt": "dotusd",
    "avaxusdt": "avaxusd",
    "xrpusdt": "xrpusd",
}


class BitstampExchange(BaseExchange):
    """Bitstamp exchange adapter using public REST API (no auth required)."""

    def __init__(self, rest_url: str = "https://www.bitstamp.net/api/v2",
                 fee_rate: float = 0.0025) -> None:
        super().__init__(
            name="bitstamp",
            rest_url=rest_url,
            fee_rate=fee_rate,
            symbol_format="{base}usdt",
        )
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Picsou/1.0",
            "Accept": "application/json",
        })

    def _bitstamp_symbol(self, symbol: str) -> str:
        """Convert to Bitstamp symbol format. BTC-USDT -> btcusdt (no dash)."""
        base = symbol.lower().replace("-", "")
        if base.endswith("usdt") or base.endswith("usd"):
            return base
        return f"{base}usdt"

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker from Bitstamp. Handles list responses from the API."""
        pair = self._bitstamp_symbol(symbol)
        url = f"{self.rest_url}/ticker/{pair}"
        try:
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 404:
                # Try fallback pair
                fallback = PAIR_FALLBACKS.get(pair)
                if fallback:
                    logger.info("Bitstamp: %s not found, trying %s", pair, fallback)
                    url = f"{self.rest_url}/ticker/{fallback}"
                    resp = self.session.get(url, timeout=10)
            if resp.status_code == 404:
                logger.warning("Bitstamp ticker: pair not found %s", pair)
                return {}
            resp.raise_for_status()
            data = resp.json()

            # Bitstamp sometimes returns a list of all tickers instead of a single one
            if isinstance(data, list):
                # Find matching pair in the list
                base = symbol.upper().replace("-", "").replace("USDT", "").replace("USD", "")
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    item_pair = item.get("pair", "")
                    if f"{base}/USDT" in item_pair or f"{base}/USD" in item_pair or f"{base}/USDC" in item_pair:
                        return {
                            "symbol": pair,
                            "bid": float(item.get("bid", 0)),
                            "ask": float(item.get("ask", 0)),
                            "last": float(item.get("last", 0)),
                            "volume": float(item.get("volume", 0)),
                            "high_24h": float(item.get("high", 0)),
                            "low_24h": float(item.get("low", 0)),
                        }
                logger.warning("Bitstamp ticker: pair %s not found in list response", pair)
                return {}

            return {
                "symbol": pair,
                "bid": float(data.get("bid", 0)),
                "ask": float(data.get("ask", 0)),
                "last": float(data.get("last", 0)),
                "volume": float(data.get("volume", 0)),
                "high_24h": float(data.get("high", 0)),
                "low_24h": float(data.get("low", 0)),
            }
        except Exception as e:
            logger.error("Bitstamp get_ticker failed for %s: %s", pair, e)
            return {}

    def get_candles(self, symbol: str, interval: str = "1h",
                    limit: int = 100) -> List[Dict[str, Any]]:
        """Get OHLC candles from Bitstamp. Returns oldest first."""
        pair = self._bitstamp_symbol(symbol)
        step = INTERVAL_MAP.get(interval, 3600)
        url = f"{self.rest_url}/ohlc/{pair}"
        params = {"step": step, "limit": limit}
        try:
            resp = self.session.get(url, params=params, timeout=10)
            if resp.status_code == 404:
                fallback = PAIR_FALLBACKS.get(pair)
                if fallback:
                    url = f"{self.rest_url}/ohlc/{fallback}"
                    resp = self.session.get(url, params=params, timeout=10)
            if resp.status_code == 404:
                logger.warning("Bitstamp OHLC: pair not found %s", pair)
                return []
            resp.raise_for_status()
            data = resp.json()
            pair_key = None
            for k in data.get("data", {}):
                if k != "ohlc" and not k.startswith("_"):
                    pair_key = k
                    break

            ohlc_data = data.get("data", {}).get("ohlc", [])
            if not ohlc_data and pair_key:
                ohlc_data = data.get("data", {}).get(pair_key, [])

            candles = []
            for c in ohlc_data:
                if isinstance(c, dict):
                    candles.append({
                        "timestamp": int(c.get("timestamp", 0)),
                        "open": float(c.get("open", 0)),
                        "high": float(c.get("high", 0)),
                        "low": float(c.get("low", 0)),
                        "close": float(c.get("close", 0)),
                        "volume": float(c.get("volume", 0)),
                    })
                elif isinstance(c, (list, tuple)) and len(c) >= 6:
                    candles.append({
                        "timestamp": int(c[0]),
                        "open": float(c[1]),
                        "high": float(c[2]),
                        "low": float(c[3]),
                        "close": float(c[4]),
                        "volume": float(c[5]),
                    })
            return candles[-limit:]
        except Exception as e:
            logger.error("Bitstamp get_candles failed for %s: %s", pair, e)
            return []

    def get_order_book(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """Get order book from Bitstamp."""
        pair = self._bitstamp_symbol(symbol)
        url = f"{self.rest_url}/order_book/{pair}"
        params = {"group": 1}
        try:
            resp = self.session.get(url, params=params, timeout=10)
            if resp.status_code == 404:
                fallback = PAIR_FALLBACKS.get(pair)
                if fallback:
                    url = f"{self.rest_url}/order_book/{fallback}"
                    resp = self.session.get(url, params=params, timeout=10)
            if resp.status_code == 404:
                logger.warning("Bitstamp order_book: pair not found %s", pair)
                return {"bids": [], "asks": []}
            resp.raise_for_status()
            data = resp.json()
            bids = [[float(p[0]), float(p[1])] for p in data.get("bids", [])[:depth]]
            asks = [[float(p[0]), float(p[1])] for p in data.get("asks", [])[:depth]]
            return {"bids": bids, "asks": asks}
        except Exception as e:
            logger.error("Bitstamp get_order_book failed for %s: %s", pair, e)
            return {"bids": [], "asks": []}