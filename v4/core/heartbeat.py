"""Picsou v4 — Heartbeat loop (fast, deterministic, no LLM).

Runs every 5 minutes. Executes active strategies on current market data.
This is the "Actor" in the Actor/Critic architecture.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from .config import PicsouConfig
from .memory import Memory
from .portfolio import Portfolio
from .safety import Safety
from .observer import Observer
from .executor import Executor
from .strategy_loader import StrategyLoader

logger = logging.getLogger(__name__)


class Heartbeat:
    """Fast loop: observe market → run strategies → execute trades.

    No LLM calls. Pure deterministic strategy execution.
    Strategies are loaded dynamically from v4/strategies/.
    """

    def __init__(self, config: PicsouConfig, portfolio: Portfolio,
                 memory: Memory, exchanges: Dict):
        self.config = config
        self.portfolio = portfolio
        self.memory = memory
        self.exchanges = exchanges
        self.safety = Safety(config)
        self.observer = Observer(config, exchanges)
        self.executor = Executor(self.safety, portfolio, memory)
        self.strategy_loader = StrategyLoader(config.strategies_path)
        self.cycle_count = 0

    def run_once(self) -> Dict[str, Any]:
        """Run one heartbeat cycle.

        1. Fetch market data
        2. Load and run all active strategies
        3. Execute decisions with safety checks
        4. Save snapshot to memory

        Returns summary dict.
        """
        self.cycle_count += 1
        logger.info("=== Heartbeat cycle #%d ===", self.cycle_count)

        # 1. Observe market
        market_data = self.observer.fetch_market_data()
        if not market_data:
            logger.warning("No market data, skipping cycle")
            return {"cycle": self.cycle_count, "actions": 0, "status": "no_data"}

        # 2. Get active strategies from memory
        active_strategies = self.memory.get_active_strategies()
        if not active_strategies:
            logger.info("No active strategies — heartbeat idle")
            decisions = []
        else:
            # 3. Run each active strategy
            decisions = self._run_strategies(active_strategies, market_data)

        # 4. Execute decisions
        executed = self.executor.execute(decisions, self.exchanges)

        # 5. Save snapshot
        pnl = self.portfolio.get_pnl()
        self.memory.save_snapshot(
            balance=self.portfolio.balance,
            positions_count=self.portfolio.get_position_count(),
            total_pnl=pnl["total_pnl"],
            return_pct=pnl["return_pct"],
            active_strategies=[s["name"] for s in active_strategies],
            cycle_number=self.cycle_count,
        )

        # 6. Check for positions that need closing (stop-loss, take-profit)
        self._check_open_positions(market_data)

        # 7. Enforce max positions per asset (2 per symbol)
        self._enforce_max_per_asset(market_data)

        summary = {
            "cycle": self.cycle_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_pairs": len(market_data),
            "active_strategies": len(active_strategies),
            "decisions_made": len(decisions),
            "trades_executed": len(executed),
            "balance": round(self.portfolio.balance, 2),
            "pnl": pnl,
            "executed_trades": executed,
        }
        logger.info("Heartbeat #%d done: %d strategies, %d trades",
                     self.cycle_count, len(active_strategies), len(executed))

        return summary

    def _run_strategies(self, active_strategies: List[Dict],
                        market_data: Dict) -> List[Dict]:
        """Run all active strategies and collect their signals."""
        all_decisions = []

        portfolio_state = self.portfolio.get_state()
        memory_context = self.memory.get_context_for_llm()

        for strategy_info in active_strategies:
            name = strategy_info["name"]
            weight = strategy_info.get("weight", 0.1)

            # Load and run the strategy
            module = self.strategy_loader.load(name)
            if module is None:
                logger.warning("Strategy %s failed to load, setting dormant", name)
                self.memory.set_strategy_status(name, "dormant")
                continue

            try:
                signal = module.signal(market_data, portfolio_state, memory_context)
                if signal and signal.get("action") != "hold":
                    # Apply weight to size
                    signal["size_pct"] = signal.get("size_pct", 0.05) * weight * 10  # Scale by weight
                    signal["size_pct"] = min(signal["size_pct"], 0.20)  # Hard cap
                    signal["strategy"] = name
                    all_decisions.append(signal)

                    logger.info("Strategy %s: %s %s (conf=%.2f, size=%.1f%%)",
                                name, signal.get("action"), signal.get("symbol", "?"),
                                signal.get("confidence", 0), signal.get("size_pct", 0) * 100)

            except Exception as e:
                logger.error("Strategy %s signal() error: %s", name, e)
                # Don't kill the strategy — it might work next cycle
                self.memory.add_observation(
                    category="error",
                    content=f"Strategy {name} crashed: {e}",
                    relevance="high"
                )

        return all_decisions

    def _check_open_positions(self, market_data: Dict):
        """Check open positions for potential close signals.

        Simple checks: if a position has been open for too long with
        significant loss, or if we have current price data for take-profit.
        """
        for pos in list(self.portfolio.positions.values()):
            # Find current price for this position's symbol
            current_price = None
            for key, md in market_data.items():
                if md.get("formatted_symbol") == pos.symbol or md.get("symbol", "").upper() in pos.symbol.upper():
                    current_price = md.get("ticker", {}).get("last")
                    break

            if current_price is None:
                continue

            # Calculate unrealized PnL
            if pos.side == "long":
                unrealized_pnl_pct = (current_price - pos.entry_price) / pos.entry_price

                # Hard stop-loss at -10%
                if unrealized_pnl_pct <= -0.10:
                    logger.warning("STOP-LOSS: %s at %.2f%% (entry=%.2f, current=%.2f)",
                                   pos.symbol, unrealized_pnl_pct * 100, pos.entry_price, current_price)
                    trade = self.portfolio.close_position(pos.id, current_price)
                    if trade:
                        self.memory.add_observation(
                            category="stop_loss",
                            content=f"Stop-loss triggered for {pos.symbol} at {unrealized_pnl_pct*100:.1f}%",
                            relevance="high"
                        )
                        self.memory.log_trade(
                            exchange=pos.exchange, symbol=pos.symbol, side="sell",
                            amount=pos.amount, price=current_price, fee=trade.get("fee", 0),
                            strategy=pos.strategy, confidence=0, reasoning="Stop-loss at -10%",
                        )

                # Take-profit at +10%
                elif unrealized_pnl_pct >= 0.10:
                    logger.info("TAKE-PROFIT: %s at +%.2f%% (entry=%.2f, current=%.2f)",
                                pos.symbol, unrealized_pnl_pct * 100, pos.entry_price, current_price)
                    trade = self.portfolio.close_position(pos.id, current_price)
                    if trade:
                        self.memory.add_observation(
                            category="take_profit",
                            content=f"Take-profit triggered for {pos.symbol} at +{unrealized_pnl_pct*100:.1f}%",
                            relevance="high"
                        )
                        self.memory.add_lesson(
                            lesson=f"Take-profit à +{unrealized_pnl_pct*100:.1f}% sur {pos.symbol} — bien joué de prendre ses gains",
                        )
                        self.memory.log_trade(
                            exchange=pos.exchange, symbol=pos.symbol, side="sell",
                            amount=pos.amount, price=current_price, fee=trade.get("fee", 0),
                            strategy=pos.strategy, confidence=0.8, reasoning=f"Take-profit at +{unrealized_pnl_pct*100:.1f}%",
                        )
    def _enforce_max_per_asset(self, market_data: Dict):
        """Enforce max 2 positions per asset. Close excess positions (oldest first)."""
        import re
        MAX_PER_ASSET = 2

        # Group positions by base symbol
        def _base(sym):
            return re.sub(r'[-_/]?[Uu][Ss][Dd][Tt]$', '', sym)

        positions = list(self.portfolio.positions.values())
        by_symbol = {}
        for pos in positions:
            base = _base(pos.symbol)
            by_symbol.setdefault(base, []).append(pos)

        for symbol, pos_list in by_symbol.items():
            if len(pos_list) > MAX_PER_ASSET:
                # Sort by entry price descending (close worst entries first)
                excess = sorted(pos_list, key=lambda p: p.entry_price, reverse=True)
                to_close = excess[MAX_PER_ASSET:]  # Keep the 2 best entries

                for pos in to_close:
                    # Find current price
                    current_price = None
                    for key, md in market_data.items():
                        if md.get("formatted_symbol") == pos.symbol or md.get("symbol", "").upper() in pos.symbol.upper():
                            current_price = md.get("ticker", {}).get("last")
                            break

                    if current_price is None:
                        logger.warning("MAX-PER-ASSET: Cannot find price for %s, skipping close", pos.symbol)
                        continue

                    logger.warning("MAX-PER-ASSET: Closing excess position #%s %s @ %.2f (current=%.2f)",
                                  pos.id, pos.symbol, pos.entry_price, current_price)
                    trade = self.portfolio.close_position(pos.id, current_price)
                    if trade:
                        self.memory.add_observation(
                            category="safety",
                            content=f"Position {pos.symbol} #{pos.id} fermée: max {MAX_PER_ASSET} par actif dépassé ({len(pos_list)} positions)",
                            relevance="high"
                        )
                        self.memory.log_trade(
                            exchange=pos.exchange, symbol=pos.symbol, side="sell",
                            amount=pos.amount, price=current_price, fee=trade.get("fee", 0),
                            strategy=pos.strategy, confidence=0,
                            reasoning=f"Safety: max {MAX_PER_ASSET} positions par actif dépassé",
                        )
