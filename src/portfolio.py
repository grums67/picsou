"""Paper trading portfolio manager for Picsou."""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Exchange fee rates
EXCHANGE_FEES = {
    "okx": 0.0008,       # 0.08%
    "kraken": 0.0026,     # 0.26%
    "bitstamp": 0.0025,   # 0.25%
}


class Position:
    """Represents an open position."""

    def __init__(self, position_id: str, exchange: str, symbol: str,
                 side: str, amount: float, entry_price: float,
                 fee: float, timestamp: str) -> None:
        self.id = position_id
        self.exchange = exchange
        self.symbol = symbol
        self.side = side  # "long" or "short"
        self.amount = amount
        self.entry_price = entry_price
        self.fee = fee
        self.open_time = timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "side": self.side,
            "amount": self.amount,
            "entry_price": self.entry_price,
            "fee": self.fee,
            "open_time": self.open_time,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Position":
        return cls(
            position_id=d["id"],
            exchange=d["exchange"],
            symbol=d["symbol"],
            side=d["side"],
            amount=d["amount"],
            entry_price=d["entry_price"],
            fee=d["fee"],
            timestamp=d["open_time"],
        )


class Trade:
    """Represents a closed trade."""

    def __init__(self, trade_id: str, position: Position, close_price: float,
                 close_fee: float, pnl: float, close_time: str) -> None:
        self.id = trade_id
        self.position = position
        self.close_price = close_price
        self.close_fee = close_fee
        self.pnl = pnl
        self.close_time = close_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "position": self.position.to_dict(),
            "close_price": self.close_price,
            "close_fee": self.close_fee,
            "pnl": self.pnl,
            "close_time": self.close_time,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trade":
        return cls(
            trade_id=d["id"],
            position=Position.from_dict(d["position"]),
            close_price=d["close_price"],
            close_fee=d["close_fee"],
            pnl=d["pnl"],
            close_time=d["close_time"],
        )


class PortfolioManager:
    """Manages simulated portfolio for paper trading."""

    def __init__(self, data_path: Path = Path("/root/PROJECTS/picsou/data"),
                 starting_capital: float = 10000.0) -> None:
        self.data_path = data_path
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.file_path = data_path / "portfolio.json"
        self.starting_capital = starting_capital
        self.balance: float = starting_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self._load()

    def _load(self) -> None:
        """Load portfolio state from disk if available."""
        if self.file_path.exists():
            try:
                data = json.loads(self.file_path.read_text())
                self.balance = data.get("balance", self.starting_capital)
                self.positions = {
                    k: Position.from_dict(v)
                    for k, v in data.get("positions", {}).items()
                }
                self.trades = [
                    Trade.from_dict(t) for t in data.get("trades", [])
                ]
                logger.info("Loaded portfolio: balance=%.2f, positions=%d",
                            self.balance, len(self.positions))
            except Exception as e:
                logger.error("Failed to load portfolio: %s", e)
                self.balance = self.starting_capital
        else:
            logger.info("No existing portfolio, starting fresh with %.2f",
                        self.starting_capital)

    def _save(self) -> None:
        """Save portfolio state to disk."""
        data = {
            "balance": self.balance,
            "starting_capital": self.starting_capital,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "trades": [t.to_dict() for t in self.trades],
        }
        self.file_path.write_text(json.dumps(data, indent=2))
        logger.debug("Portfolio saved to %s", self.file_path)

    def open_position(self, exchange: str, symbol: str, side: str,
                      amount: float, price: float) -> Optional[Position]:
        """Open a new position (paper trade).

        Args:
            exchange: Exchange name (okx, kraken, bitstamp).
            symbol: Trading pair symbol.
            side: "long" or "short".
            amount: Quantity in base currency.
            price: Entry price.

        Returns:
            The Position object or None if insufficient balance.
        """
        fee_rate = EXCHANGE_FEES.get(exchange, 0.001)
        cost = amount * price
        fee = cost * fee_rate
        total_cost = cost + fee

        if side == "long":
            if total_cost > self.balance:
                logger.warning(
                    "Insufficient balance: need %.2f, have %.2f",
                    total_cost, self.balance,
                )
                return None
            self.balance -= total_cost

        pos_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now(timezone.utc).isoformat()
        position = Position(
            position_id=pos_id,
            exchange=exchange,
            symbol=symbol,
            side=side,
            amount=amount,
            entry_price=price,
            fee=fee,
            timestamp=timestamp,
        )
        self.positions[pos_id] = position
        self._save()
        logger.info("Opened %s position: %s %s @ %.2f on %s (cost=%.2f, fee=%.2f)",
                     side, symbol, amount, price, exchange, total_cost, fee)
        return position

    def close_position(self, position_id: str, price: float) -> Optional[Trade]:
        """Close an existing position (paper trade).

        Args:
            position_id: The position ID to close.
            price: Exit price.

        Returns:
            The Trade object or None if position not found.
        """
        position = self.positions.pop(position_id, None)
        if position is None:
            logger.warning("Position %s not found", position_id)
            return None

        fee_rate = EXCHANGE_FEES.get(position.exchange, 0.001)
        cost = position.amount * price
        close_fee = cost * fee_rate

        if position.side == "long":
            entry_cost = position.amount * position.entry_price
            pnl = cost - entry_cost - position.fee - close_fee
            self.balance += cost - close_fee
        else:
            # Short: profit if price goes down
            entry_value = position.amount * position.entry_price
            close_value = position.amount * price
            pnl = entry_value - close_value - position.fee - close_fee
            self.balance += entry_value + pnl

        trade_id = str(uuid.uuid4())[:8]
        close_time = datetime.now(timezone.utc).isoformat()
        trade = Trade(
            trade_id=trade_id,
            position=position,
            close_price=price,
            close_fee=close_fee,
            pnl=pnl,
            close_time=close_time,
        )
        self.trades.append(trade)
        self._save()
        logger.info("Closed position %s: PnL=%.2f (fee=%.4f)",
                     position_id, pnl, close_fee)
        return trade

    def get_open_positions(self) -> List[Position]:
        """Return list of currently open positions."""
        return list(self.positions.values())

    def get_balance(self) -> float:
        """Return current simulated balance in EUR."""
        return self.balance

    def get_pnl(self) -> Dict[str, float]:
        """Calculate PnL statistics.

        Returns dict with: total_pnl, realized_pnl, unrealized_pnl,
        win_rate, total_trades, winning_trades.
        """
        realized_pnl = sum(t.pnl for t in self.trades)
        winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        total_trades = len(self.trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Unrealized PnL based on open positions (using entry price as estimate)
        unrealized_pnl = 0.0
        for pos in self.positions.values():
            if pos.side == "long":
                # Would need current price for accurate unrealized PnL
                unrealized_pnl -= pos.fee  # At minimum, fees will reduce PnL

        total_pnl = (self.balance - self.starting_capital) + sum(
            pos.amount * pos.entry_price for pos in self.positions.values()
        )

        return {
            "total_pnl": total_pnl,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "balance": self.balance,
            "starting_capital": self.starting_capital,
            "return_pct": ((total_pnl) /
                           self.starting_capital * 100) if self.starting_capital else 0.0,
        }

    def get_position_count(self) -> int:
        """Return number of open positions."""
        return len(self.positions)