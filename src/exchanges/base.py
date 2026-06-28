"""Abstract base class for exchange adapters."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseExchange(ABC):
    """Abstract base exchange that all adapters must implement.

    Public data methods work without API keys.
    Order and balance methods are simulated for paper trading.
    """

    def __init__(self, name: str, rest_url: str, fee_rate: float,
                 symbol_format: str) -> None:
        self.name = name
        self.rest_url = rest_url
        self.fee_rate = fee_rate
        self.symbol_format = symbol_format

    def format_symbol(self, base: str) -> str:
        """Format a base currency (e.g. BTC) into the exchange-specific symbol."""
        return self.symbol_format.format(base=base)

    @abstractmethod
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data for a symbol.

        Returns dict with keys: symbol, bid, ask, last, volume, high_24h, low_24h.
        """
        ...

    @abstractmethod
    def get_candles(self, symbol: str, interval: str = "1h",
                    limit: int = 100) -> List[Dict[str, Any]]:
        """Get OHLCV candle data for a symbol.

        Returns list of dicts with keys:
            timestamp, open, high, low, close, volume
        Oldest candle first.
        """
        ...

    @abstractmethod
    def get_order_book(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """Get order book for a symbol.

        Returns dict with keys: bids, asks (each a list of [price, amount]).
        """
        ...

    def place_order(self, symbol: str, side: str, amount: float,
                    price: float, portfolio: Any = None) -> Dict[str, Any]:
        """Place a simulated order (paper trading mode).

        In paper trading, this delegates to the portfolio manager.
        Returns dict with keys: order_id, symbol, side, amount, price, fee, total.
        """
        logger.info(
            "PAPER ORDER: %s %s %s amount=%.6f price=%.2f on %s",
            side.upper(), symbol, self.name, amount, price, self.name,
        )
        return {
            "order_id": f"paper_{self.name}_{symbol}_{side}",
            "exchange": self.name,
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": price,
            "fee": amount * price * self.fee_rate,
            "total": amount * price,
            "status": "filled",
        }

    def get_balance(self) -> Dict[str, float]:
        """Get simulated balance (paper trading mode).

        Returns placeholder; real balances come from PortfolioManager.
        """
        logger.debug("PAPER BALANCE requested on %s", self.name)
        return {"EUR": 0.0}

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"