"""Picsou v4 — Executor. Takes decisions and turns them into trades.

Runs safety checks, size capping, then delegates to portfolio manager.
"""

import logging
from typing import Any, Dict, List, Optional

from .portfolio import Portfolio
from .safety import Safety, SafetyCheck
from .memory import Memory

logger = logging.getLogger(__name__)


class Executor:
    """Executes trading decisions with hard safety limits."""

    def __init__(self, safety: Safety, portfolio: Portfolio, memory: Memory):
        self.safety = safety
        self.portfolio = portfolio
        self.memory = memory

    def execute(self, decisions: List[Dict], exchanges: Dict) -> List[Dict]:
        """Execute a list of trading decisions.

        Each decision dict should have:
            action: "buy" | "sell" | "hold"
            symbol: str (e.g. "BTC")
            exchange: str (optional, defaults to first exchange)
            size_pct: float 0.0-1.0 (% of balance)
            confidence: float 0.0-1.0
            strategy: str
            reasoning: str

        Returns list of executed trade dicts.
        """
        executed = []

        for decision in decisions:
            action = decision.get("action", "hold").lower()

            if action == "hold":
                logger.info("HOLD %s: %s", decision.get("symbol", "?"), decision.get("reasoning", ""))
                continue

            if action not in ("buy", "sell"):
                logger.warning("Unknown action: %s", action)
                continue

            result = self._execute_one(decision, exchanges)
            if result:
                executed.append(result)

        return executed

    def _execute_one(self, decision: Dict, exchanges: Dict) -> Optional[Dict]:
        """Execute a single trade decision."""
        action = decision["action"]
        symbol = decision.get("symbol", "BTC")
        exchange_name = decision.get("exchange", list(exchanges.keys())[0])
        size_pct = decision.get("size_pct", 0.05)
        confidence = decision.get("confidence", 0.5)
        strategy = decision.get("strategy", "unknown")
        reasoning = decision.get("reasoning", "")

        # Validate exchange
        if exchange_name not in exchanges:
            logger.warning("Unknown exchange %s, using %s", exchange_name, list(exchanges.keys())[0])
            exchange_name = list(exchanges.keys())[0]

        exchange = exchanges[exchange_name]
        formatted_symbol = exchange.format_symbol(symbol)

        # Get current price
        ticker = exchange.get_ticker(formatted_symbol)
        if not ticker or ticker.get("last", 0) == 0:
            logger.warning("No price for %s on %s, skipping", formatted_symbol, exchange_name)
            return None
        price = ticker["last"]

        # Get portfolio state for safety checks
        pnl = self.portfolio.get_pnl()
        balance = self.portfolio.balance
        open_positions = self.portfolio.get_position_count()

        # ── BUY ──────────────────────────────────────────────────────────
        if action == "buy":
            amount_usd = balance * size_pct

            # Safety checks
            check = self.safety.check_trade(
                side="buy", amount_usd=amount_usd, balance=balance,
                open_positions=open_positions,
                starting_capital=self.portfolio.starting_capital,
                current_pnl_pct=pnl["return_pct"] / 100,
            )
            if not check.allowed:
                logger.warning("SAFETY BLOCKED buy %s: %s", symbol, check.reason)
                return None

            # Cap position size
            amount_usd = self.safety.cap_position_size(amount_usd, balance)
            amount = amount_usd / price

            # Open position
            pos = self.portfolio.open_position(
                exchange=exchange_name, symbol=formatted_symbol,
                side="long", amount=amount, price=price,
                strategy=strategy,
            )
            if pos is None:
                return None

            # Log to memory
            self.memory.log_trade(
                exchange=exchange_name, symbol=formatted_symbol, side="buy",
                amount=amount, price=price, fee=pos.fee,
                strategy=strategy, confidence=confidence, reasoning=reasoning,
            )

            return {
                "action": "buy", "symbol": formatted_symbol, "exchange": exchange_name,
                "amount": round(amount, 8), "price": price, "size_usd": round(amount_usd, 2),
                "confidence": confidence, "strategy": strategy, "reasoning": reasoning,
                "position_id": pos.id,
            }

        # ── SELL ─────────────────────────────────────────────────────────
        elif action == "sell":
            # Find matching open position
            import re
            def _base_symbol(sym):
                return re.sub(r'[-_/]?[Uu][Ss][Dd][Tt]$', '', sym)

            base = _base_symbol(formatted_symbol)
            matching = [
                p for p in self.portfolio.get_open_positions()
                if _base_symbol(p.symbol) == base and p.side == "long"
            ]

            if not matching:
                logger.info("No open position to sell for %s", symbol)
                return None

            # Close the first matching position
            pos = matching[0]
            trade = self.portfolio.close_position(pos.id, price)

            if trade:
                self.memory.close_trade(
                    trade_id=0,  # We'll use the memory log approach instead
                    close_price=price,
                    pnl=trade["pnl"],
                )
                # Update the latest trade in memory with close info
                self.memory.log_trade(
                    exchange=exchange_name, symbol=formatted_symbol, side="sell",
                    amount=trade["amount"], price=price, fee=trade["fee"],
                    strategy=strategy, confidence=confidence, reasoning=reasoning,
                )

            return trade

        return None