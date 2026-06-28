"""OKX exchange adapter for Picsou."""

import logging
import time
from typing import Any, Dict, List

import requests

from .base import BaseExchange

logger = logging.getLogger(__name__)

# Interval mapping for OKX candle API
INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
}


class OKXExchange(BaseExchange):
    """OKX exchange adapter using public REST API (no auth required)."""

    def __init__(self, rest_url: str = "https://www.okx.com/api/v5",
                 fee_rate: float = 0.0008) -> None:
        super().__init__(
            name="okx",
            rest_url=rest_url,
            fee_rate=fee_rate,
            symbol_format="{base}-USDT",
        )
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Picsou/1.0",
            "Accept": "application/json",
        })

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker from OKX. Symbol format: BTC-USDT."""
        inst_id = self.format_symbol(symbol) if "-" not in symbol else symbol
        url = f"{self.rest_url}/market/ticker"
        params = {"instId": inst_id}
        try:
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != "0" or not data.get("data"):
                logger.warning("OKX ticker error for %s: %s", inst_id, data.get("msg"))
                return {}
            t = data["data"][0]
            return {
                "symbol": inst_id,
                "bid": float(t.get("bid", 0)),
                "ask": float(t.get("ask", 0)),
                "last": float(t.get("last", 0)),
                "volume": float(t.get("vol24h", 0)),
                "high_24h": float(t.get("high24h", 0)),
                "low_24h": float(t.get("low24h", 0)),
            }
        except Exception as e:
            logger.error("OKX get_ticker failed for %s: %s", inst_id, e)
            return {}

    def get_candles(self, symbol: str, interval: str = "1h",
                    limit: int = 100) -> List[Dict[str, Any]]:
        """Get candles from OKX. Returns oldest first."""
        inst_id = self.format_symbol(symbol) if "-" not in symbol else symbol
        bar = INTERVAL_MAP.get(interval, "1H")
        url = f"{self.rest_url}/market/candles"
        params = {"instId": inst_id, "bar": bar, "limit": str(limit)}
        try:
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != "0" or not data.get("data"):
                logger.warning("OKX candles error for %s: %s", inst_id, data.get("msg"))
                return []
            # OKX returns newest first, we reverse to oldest first
            candles = []
            for c in reversed(data["data"]):
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
            logger.error("OKX get_candles failed for %s: %s", inst_id, e)
            return []

    def get_order_book(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """Get order book from OKX."""
        inst_id = self.format_symbol(symbol) if "-" not in symbol else symbol
        url = f"{self.rest_url}/market/books"
        params = {"instId": inst_id, "sz": str(depth)}
        try:
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != "0" or not data.get("data"):
                logger.warning("OKX order_book error for %s: %s", inst_id, data.get("msg"))
                return {"bids": [], "asks": []}
            book = data["data"][0]
            bids = [[float(p[0]), float(p[1])] for p in book.get("bids", [])]
            asks = [[float(p[0]), float(p[1])] for p in book.get("asks", [])]
            return {"bids": bids, "asks": asks}
        except Exception as e:
            logger.error("OKX get_order_book failed for %s: %s", inst_id, e)
            return {"bids": [], "asks": []}