"""Picsou v4 — Brain loop (slow, LLM-driven, every N heartbeats).

This is the "Critic" in the Actor/Critic architecture.
The LLM analyzes performance, creates/modifies strategies, and adjusts weights.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from .config import PicsouConfig
from .memory import Memory
from .portfolio import Portfolio
from .observer import Observer
from .brain import Brain, TOOL_DEFINITIONS
from .strategy_loader import StrategyLoader
from .backtest import Backtester
from .executor import Executor
from .safety import Safety

logger = logging.getLogger(__name__)


class BrainLoop:
    """Slow loop: LLM analyzes, creates strategies, adjusts weights.

    Called every N heartbeat cycles (configurable, default ~1h).
    """

    def __init__(self, config: PicsouConfig, portfolio: Portfolio,
                 memory: Memory, exchanges: Dict):
        self.config = config
        self.portfolio = portfolio
        self.memory = memory
        self.exchanges = exchanges
        self.brain = Brain(config)
        self.observer = Observer(config, exchanges)
        self.strategy_loader = StrategyLoader(config.strategies_path)
        self.backtester = Backtester()
        self.safety = Safety(config)
        self.executor = Executor(self.safety, portfolio, memory)
        self.cycle_count = 0

    def should_run(self, heartbeat_cycle: int) -> bool:
        """Check if the brain should run this cycle."""
        interval = self.config.safety.brain_interval_cycles  # Default 12 = ~1h
        return heartbeat_cycle % interval == 0

    def run_once(self) -> Dict[str, Any]:
        """Run one brain cycle.

        1. Build full context
        2. Send to LLM
        3. Process tool calls (write strategy, test strategy, adjust weights, etc.)
        4. Update memory with observations and lessons
        """
        self.cycle_count += 1
        logger.info("=== Brain cycle #%d ===", self.cycle_count)

        # 1. Build context
        market_data = self.observer.fetch_market_data()
        sentiment = self.observer.fetch_sentiment()
        portfolio_state = self.portfolio.get_state()
        memory_context = self.memory.get_context_for_llm()

        # Discover new strategy files that appeared on disk
        self._discover_new_strategies()

        # Evaluate and adjust existing strategies
        self._evaluate_strategies()

        context = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market": {},
            "sentiment": sentiment,
            "portfolio": portfolio_state,
            "memory": memory_context,
            "symbols": self.config.symbols,
            "cycle": self.cycle_count,
        }

        # Compact market data for context
        for key, md in market_data.items():
            ticker = md.get("ticker", {})
            candles = md.get("candles", [])
            context["market"][key] = {
                "price": ticker.get("last", 0) if ticker else 0,
                "volume_24h": ticker.get("volume_24h", 0) if ticker else 0,
                "change_24h": ticker.get("change_24h", 0) if ticker else 0,
                "candles_count": len(candles),
                "last_5_candles": candles[-5:] if len(candles) >= 5 else candles,
            }

        # 2. Call LLM
        decision = self.brain.think(context)

        # 3. Process the decision
        result = self._process_decision(decision, market_data)

        logger.info("Brain cycle #%d: action=%s, result=%s",
                     self.cycle_count, decision.get("action", "?"), result)

        return result

    def _process_decision(self, decision: Dict, market_data: Dict) -> Dict:
        """Process an LLM decision and execute any tool calls."""
        action = decision.get("action", "hold")
        results = {"action": action}

        if action == "hold":
            # Record observations and lessons
            observations = decision.get("observations", [])
            lessons = decision.get("lessons", [])
            for obs in observations if isinstance(observations, list) else []:
                self.memory.add_observation(category="brain", content=str(obs))
            for lesson in lessons if isinstance(lessons, list) else []:
                self.memory.add_lesson(lesson=str(lesson))

        elif action in ("buy", "sell"):
            # LLM wants to trade — execute via the Executor
            trade_decision = {
                "action": action,
                "symbol": decision.get("symbol", "BTC"),
                "size_pct": decision.get("size_pct", 0.05),
                "confidence": decision.get("confidence", 0.5),
                "strategy": decision.get("strategy", "brain"),
                "reasoning": decision.get("reasoning", ""),
            }
            executed = self.executor.execute([trade_decision], self.exchanges)
            if executed:
                logger.info("Brain trade executed: %s %s — %s",
                            action.upper(), trade_decision["symbol"], executed)
                results["trades"] = executed
            else:
                logger.info("Brain trade not executed (safety or no match): %s %s",
                            action, trade_decision["symbol"])
                results["trades"] = []

        elif action == "create_strategy":
            # LLM wants to create a new strategy
            name = decision.get("name", "")
            code = decision.get("code", "")
            reasoning = decision.get("reasoning", "")

            if not name or not code:
                logger.warning("Brain wants to create strategy but missing name/code")
                return {"action": "create_strategy", "result": "missing_name_or_code"}

            # Validate and write
            is_valid, error = self.strategy_loader.validate_code(code)
            if not is_valid:
                logger.error("Strategy code invalid: %s", error)
                self.memory.add_observation(
                    category="error", content=f"Strategy {name} rejected: {error}",
                    relevance="high"
                )
                return {"action": "create_strategy", "result": f"validation_failed: {error}"}

            written = self.strategy_loader.write_strategy(name, code)
            if written:
                # Register in memory as probation
                self.memory.register_strategy(name, filename=f"{name}.py",
                                               metadata={"created_by": "llm", "reasoning": reasoning})
                self.memory.update_strategy(name, status="probation", weight=0.05)
                self.memory.add_observation(
                    category="strategy_created",
                    content=f"Created strategy {name}: {reasoning}",
                    relevance="high"
                )
                results["result"] = "created"
            else:
                results["result"] = "write_failed"

        elif action == "modify_strategy":
            name = decision.get("name", "")
            code = decision.get("code", "")
            reasoning = decision.get("reasoning", "")

            if not name or not code:
                return {"action": "modify_strategy", "result": "missing_name_or_code"}

            is_valid, error = self.strategy_loader.validate_code(code)
            if not is_valid:
                return {"action": "modify_strategy", "result": f"validation_failed: {error}"}

            written = self.strategy_loader.write_strategy(name, code)
            if written:
                # Force reload
                self.strategy_loader.reload(name)
                self.memory.add_observation(
                    category="strategy_modified",
                    content=f"Modified strategy {name}: {reasoning}",
                    relevance="high"
                )
                results["result"] = "modified"
            else:
                results["result"] = "write_failed"

        elif action == "adjust_weight":
            strategy = decision.get("strategy", "")
            new_status = decision.get("status", "")
            new_weight = decision.get("weight", None)
            reasoning = decision.get("reasoning", "")

            if not strategy:
                return {"action": "adjust_weight", "result": "missing_strategy"}

            if new_status:
                self.memory.set_strategy_status(strategy, new_status)
            if new_weight is not None:
                self.memory.update_strategy(strategy, weight=float(new_weight))

            self.memory.add_observation(
                category="weight_adjustment",
                content=f"Adjusted {strategy}: status={new_status}, weight={new_weight}. Reason: {reasoning}",
                relevance="medium"
            )
            results["result"] = "adjusted"

        elif action == "test_strategy":
            name = decision.get("name", "")
            symbol = decision.get("symbol", "BTC")

            module = self.strategy_loader.load(name)
            if module is None:
                return {"action": "test_strategy", "result": "strategy_not_found"}

            # Get historical candles for backtest
            candles = self._get_candles_for_backtest(symbol)
            if not candles:
                return {"action": "test_strategy", "result": "no_candle_data"}

            result = self.backtester.run(module, candles, f"{symbol}-USDT")
            results["backtest"] = result.to_dict()

            # If passed, activate; if failed, set dormant
            if result.passed:
                self.memory.set_strategy_status(name, "active")
                self.memory.add_lesson(
                    lesson=f"Strategy {name} passed backtest: WR={result.win_rate:.0%}, avg_pnl={result.avg_pnl:.4f}",
                    context=f"Backtest on {symbol}"
                )
            else:
                self.memory.set_strategy_status(name, "dormant")
                self.memory.add_observation(
                    category="backtest_failed",
                    content=f"Strategy {name} failed backtest: {result.total_trades} trades, WR={result.win_rate:.0%}",
                    relevance="medium"
                )

        # Record observations and lessons from any action
        observations = decision.get("observations", [])
        lessons = decision.get("lessons", [])
        if isinstance(observations, list):
            for obs in observations:
                self.memory.add_observation(category="brain", content=str(obs))
        if isinstance(lessons, list):
            for lesson in lessons:
                self.memory.add_lesson(lesson=str(lesson))

        return results

    def _discover_new_strategies(self):
        """Find strategy files on disk that aren't registered in memory yet."""
        disk_strategies = self.strategy_loader.discover()
        memory_strategies = {s["name"] for s in self.memory.get_all_strategies()}

        for name in disk_strategies:
            if name not in memory_strategies:
                meta = self.strategy_loader.get_metadata(name)
                if meta:
                    self.memory.register_strategy(
                        name=name,
                        filename=f"{name}.py",
                        metadata=meta,
                    )
                    # New strategies start on probation
                    self.memory.update_strategy(name, status="probation", weight=0.05)
                    logger.info("Discovered new strategy: %s", name)

    def _evaluate_strategies(self):
        """Evaluate strategy performance from trade history.

        Adjusts weights based on performance. Never eliminates — only dormancy.
        """
        strategies = self.memory.get_all_strategies()

        for strat in strategies:
            name = strat["name"]
            if strat["status"] == "dormant":
                # Check if it's time to wake up (at least 24h dormant)
                dormant_since = strat.get("dormant_since")
                if dormant_since:
                    dormant_time = datetime.fromisoformat(dormant_since)
                    hours_dormant = (datetime.now(timezone.utc) - dormant_time).total_seconds() / 3600
                    if hours_dormant < 24:
                        continue  # Not time yet
                # Wake up on probation
                self.memory.update_strategy(name, status="probation", weight=0.05)
                logger.info("Strategy %s waking up from dormancy (probation)", name)

            # Calculate recent performance
            trades = self.memory.get_trades_by_strategy(name, limit=20)
            if not trades:
                continue

            closed = [t for t in trades if t.get("status") == "closed" and t.get("pnl") is not None]
            if len(closed) < 3:
                continue  # Not enough data

            wins = sum(1 for t in closed if (t.get("pnl") or 0) > 0)
            win_rate = wins / len(closed) if closed else 0
            total_pnl = sum(t.get("pnl", 0) for t in closed)

            # Update strategy stats
            self.memory.update_strategy(
                name,
                total_trades=len(closed),
                winning_trades=wins,
                win_rate=win_rate,
                total_pnl=total_pnl,
                last_evaluated=datetime.now(timezone.utc).isoformat(),
            )

            # Adjust weight: winners get boosted, losers go dormant
            current_weight = strat.get("weight", 0.1)

            if win_rate > 0.55 and total_pnl > 0:
                # Winning strategy — boost weight
                new_weight = min(current_weight * 1.2, 0.4)  # Cap at 40%
                self.memory.update_strategy(name, weight=new_weight)
                if strat["status"] == "probation":
                    self.memory.set_strategy_status(name, "active")
                logger.info("Strategy %s boosted: weight %.2f → %.2f (WR=%.0f%%)",
                            name, current_weight, new_weight, win_rate * 100)

            elif win_rate < 0.25 and len(closed) >= 5:
                # Losing strategy — put to sleep (dormant), not killed
                self.memory.set_strategy_status(name, "dormant")
                self.memory.update_strategy(name, weight=0.01)
                self.memory.add_observation(
                    category="strategy_dormant",
                    content=f"Strategy {name} went dormant: WR={win_rate:.0%} over {len(closed)} trades",
                    relevance="medium"
                )
                logger.info("Strategy %s went dormant: WR=%.0f%% over %d trades",
                            name, win_rate * 100, len(closed))

    def _get_candles_for_backtest(self, symbol: str) -> List[Dict]:
        """Get historical candle data for backtesting."""
        # Try to get from the primary exchange
        exchange_name = list(self.exchanges.keys())[0]
        exchange = self.exchanges[exchange_name]
        formatted = exchange.format_symbol(symbol)

        try:
            candles = exchange.get_candles(formatted, interval="1h", limit=200)
            return candles or []
        except Exception as e:
            logger.error("Failed to get candles for backtest: %s", e)
            return []