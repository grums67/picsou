"""Picsou v4 — Portfolio manager.

Manages positions, balance, PnL for paper trading.
Reused from v3 but simplified — no JSON, state goes to SQLite via Memory.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Exchange fee rates
EXCHANGE_FEES = {
    "okx": 0.0008,
    "kraken": 0.0026,
    "bitstamp": 0.0025,
}


class Position:
    """An open position."""
    def __init__(self, position_id: str, exchange: str, symbol: str,
                 side: str, amount: float, entry_price: float,
                 fee: float, timestamp: str, strategy: str = ""):
        self.id = position_id
        self.exchange = exchange
        self.symbol = symbol
        self.side = side
        self.amount = amount
        self.entry_price = entry_price
        self.fee = fee
        self.open_time = timestamp
        self.strategy = strategy

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "exchange": self.exchange, "symbol": self.symbol,
            "side": self.side, "amount": self.amount, "entry_price": self.entry_price,
            "fee": self.fee, "open_time": self.open_time, "strategy": self.strategy,
        }

    def cost_basis(self) -> float:
        return self.amount * self.entry_price + self.fee


class Portfolio:
    """Paper trading portfolio — state persisted via Memory (SQLite)."""

    def __init__(self, starting_capital: float = 10000.0, memory=None):
        self.starting_capital = starting_capital
        self.balance = starting_capital
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Dict] = []
        if memory is not None:
            self._restore_from_memory(memory)

    def _restore_from_memory(self, memory):
        """Restore portfolio state from trades stored in Memory (SQLite)."""
        # Reload open positions
        open_trades = memory.get_open_trades()
        for t in open_trades:
            pos = Position(
                position_id=str(t.get("id", "")),
                exchange=t.get("exchange", "okx"),
                symbol=t.get("symbol", ""),
                side=t.get("side", "buy"),
                amount=t.get("amount", 0),
                entry_price=t.get("price", 0),
                fee=t.get("fee", 0),
                timestamp=t.get("timestamp", ""),
                strategy=t.get("strategy", ""),
            )
            self.positions[pos.id] = pos
            # Deduct cost from balance (simulates the buy)
            cost = pos.amount * pos.entry_price + pos.fee
            self.balance -= cost
            logger.info("Restored position %s %s %s @ %.2f (cost=%.2f)",
                        pos.id, pos.side.upper(), pos.symbol, pos.entry_price, cost)

        # Reload closed trades PnL
        closed_trades = memory.get_closed_trades()
        for t in closed_trades:
            self.closed_trades.append(t)
            # Add close proceeds back to balance
            if t.get("side") == "buy" and t.get("close_price"):
                close_proceeds = t.get("amount", 0) * t.get("close_price", 0)
                close_fee = close_proceeds * EXCHANGE_FEES.get(t.get("exchange", "okx"), 0.001)
                self.balance += close_proceeds - close_fee

        logger.info("Portfolio restored: balance=$%.2f, %d open positions, %d closed trades",
                    self.balance, len(self.positions), len(self.closed_trades))

    def get_state(self) -> Dict[str, Any]:
        """Get full portfolio state for context/serialization."""
        return {
            "balance": round(self.balance, 2),
            "starting_capital": self.starting_capital,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "open_position_count": len(self.positions),
            "pnl": self.get_pnl(),
        }

    def get_pnl(self) -> Dict[str, float]:
        """Calculate PnL statistics."""
        realized_pnl = sum(t["pnl"] for t in self.closed_trades if "pnl" in t)
        wins = sum(1 for t in self.closed_trades if t.get("pnl", 0) > 0)
        total = len(self.closed_trades)
        unrealized = sum(
            p.amount * p.entry_price for p in self.positions.values()
        )
        total_pnl = self.balance - self.starting_capital + unrealized
        return {
            "realized_pnl": round(realized_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "unrealized_value": round(unrealized, 2),
            "win_rate": round(wins / total, 4) if total > 0 else 0,
            "total_trades": total,
            "winning_trades": wins,
            "return_pct": round(total_pnl / self.starting_capital * 100, 2) if self.starting_capital else 0,
        }

    def open_position(self, exchange: str, symbol: str, side: str,
                      amount: float, price: float, strategy: str = "") -> Optional[Position]:
        """Open a new position."""
        fee_rate = EXCHANGE_FEES.get(exchange, 0.001)
        cost = amount * price
        fee = cost * fee_rate
        total_cost = cost + fee

        if side == "long" and total_cost > self.balance:
            logger.warning("Insufficient balance: need %.2f, have %.2f", total_cost, self.balance)
            return None

        if side == "long":
            self.balance -= total_cost

        pos_id = str(uuid.uuid4())[:8]
        pos = Position(
            position_id=pos_id, exchange=exchange, symbol=symbol,
            side=side, amount=amount, entry_price=price,
            fee=fee, timestamp=datetime.now(timezone.utc).isoformat(),
            strategy=strategy,
        )
        self.positions[pos_id] = pos
        logger.info("OPEN %s %s %s %.6f @ %.2f on %s (cost=%.2f fee=%.2f)",
                     side.upper(), symbol, amount, amount, price, exchange, total_cost, fee)
        return pos

    def close_position(self, position_id: str, price: float) -> Optional[Dict]:
        """Close a position and return trade record."""
        pos = self.positions.pop(position_id, None)
        if pos is None:
            logger.warning("Position %s not found", position_id)
            return None

        fee_rate = EXCHANGE_FEES.get(pos.exchange, 0.001)
        close_cost = pos.amount * price
        close_fee = close_cost * fee_rate

        if pos.side == "long":
            entry_cost = pos.amount * pos.entry_price
            pnl = close_cost - entry_cost - pos.fee - close_fee
            self.balance += close_cost - close_fee
        else:
            # Short position PnL
            entry_cost = pos.amount * pos.entry_price
            pnl = entry_cost - close_cost - pos.fee - close_fee
            self.balance += entry_cost + pnl

        trade = {
            "id": pos.id, "exchange": pos.exchange, "symbol": pos.symbol,
            "side": pos.side, "amount": pos.amount, "entry_price": pos.entry_price,
            "close_price": price, "fee": pos.fee + close_fee,
            "pnl": round(pnl, 4), "pnl_pct": round(pnl / entry_cost * 100, 2) if entry_cost else 0,
            "strategy": pos.strategy,
            "open_time": pos.open_time,
            "close_time": datetime.now(timezone.utc).isoformat(),
        }
        self.closed_trades.append(trade)
        logger.info("CLOSE %s %s PnL=%.2f (%.2f%%)", pos.symbol, pos.side, pnl, trade["pnl_pct"])
        return trade

    def get_open_positions(self) -> List[Position]:
        return list(self.positions.values())

    def get_position_count(self) -> int:
        return len(self.positions)