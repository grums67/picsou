"""Main Picsou agent loop - autonomous LLM-driven crypto trading agent.

The agent uses an LLM brain (Mistral) to analyze market data, sentiment,
and portfolio state, and produce trading decisions with reasoning.
Falls back to simple EMA crossover signals if the LLM is unavailable.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .brain import PicsouBrain
from .config import PicsouConfig, get_config
from .exchanges.okx import OKXExchange
from .exchanges.kraken import KrakenExchange
from .exchanges.bitstamp import BitstampExchange
from .exchanges.base import BaseExchange
from .journal import DecisionJournal
from .learning import LearningEngine
from .portfolio import PortfolioManager
from .strategy_researcher import StrategyResearcher
from .self_reflect import SelfReflect

logger = logging.getLogger(__name__)


class PicsouAgent:
    """Main autonomous trading agent.

    Orchestrates market data fetching, LLM analysis,
    decision making, and portfolio management.
    """

    def __init__(self, config: Optional[PicsouConfig] = None) -> None:
        self.config = config or get_config()

        # Initialize components
        self.portfolio = PortfolioManager(
            data_path=self.config.paths.data,
            starting_capital=self.config.starting_capital,
        )
        self.journal = DecisionJournal(data_path=self.config.paths.data)
        self.learning = LearningEngine(
            data_path=self.config.paths.data,
            win_rate_threshold=self.config.learning.win_rate_threshold,
            min_trades=self.config.learning.min_trades,
            min_days=self.config.learning.min_days,
            elimination_win_rate=self.config.learning.elimination_win_rate,
            elimination_max_drawdown=self.config.learning.elimination_max_drawdown,
            min_exploration_trades=self.config.learning.min_exploration_trades,
        )

        # Initialize exchanges
        self.exchanges: Dict[str, BaseExchange] = {}
        for name, exc_cfg in self.config.exchanges.items():
            if name == "okx":
                self.exchanges[name] = OKXExchange(
                    rest_url=exc_cfg.rest_url,
                    fee_rate=exc_cfg.fee_rate,
                )
            elif name == "kraken":
                self.exchanges[name] = KrakenExchange(
                    rest_url=exc_cfg.rest_url,
                    fee_rate=exc_cfg.fee_rate,
                )
            elif name == "bitstamp":
                self.exchanges[name] = BitstampExchange(
                    rest_url=exc_cfg.rest_url,
                    fee_rate=exc_cfg.fee_rate,
                )

        # Initialize the LLM brain
        self.brain = PicsouBrain(
            llm_url=self.config.llm_url,
            llm_api_key=self.config.llm_api_key,
            llm_model=self.config.llm_model,
            llm_temperature=self.config.llm_temperature,
            llm_max_tokens=self.config.llm_max_tokens,
            fear_and_greed_enabled=self.config.fear_and_greed_enabled,
            news_enabled=self.config.news_enabled,
            config_path=self.config.llm_config_path,
        )

        # Initialize strategy researcher (web-sourced insights for LLM)
        self.strategy_researcher = StrategyResearcher(
            cache_path=str(self.config.paths.data / "research_cache.json"),
            cache_ttl=self.config.research_cache_ttl,
            max_sources=self.config.research_max_sources,
            enabled=self.config.research_enabled,
        )

        # Initialize self-reflection engine
        self.reflector = SelfReflect(
            data_path=self.config.paths.data,
            llm_url=self.config.llm_url,
            llm_api_key=self.config.llm_api_key,
            llm_model=self.config.llm_model,
            llm_config_path=self.config.llm_config_path,
        )
        self.reflection_counter = 0
        self._strategy_rules = {}

        # Symbol list
        self.symbols = self.config.symbols

        logger.info(
            "Picsou agent initialized: phase=%s, capital=%.2f, "
            "exchanges=%s, llm_model=%s, llm_url=%s",
            self.config.phase, self.config.starting_capital,
            list(self.exchanges.keys()),
            self.config.llm_model, self.config.llm_url,
        )

    def _get_exchange_symbol(self, exchange_name: str, base: str) -> str:
        """Get the exchange-formatted symbol for a base currency."""
        exc = self.exchanges.get(exchange_name)
        if exc:
            return exc.format_symbol(base)
        return f"{base}-USDT"

    def fetch_market_data(self) -> Dict[str, Dict[str, Any]]:
        """Fetch market data from all exchanges for all symbols.

        Returns:
            Dict mapping "exchange:symbol" to market data dicts.
        """
        market_data: Dict[str, Dict[str, Any]] = {}

        for exc_name, exchange in self.exchanges.items():
            for base in self.symbols:
                symbol = self._get_exchange_symbol(exc_name, base)
                key = f"{exc_name}:{base}"

                logger.debug("Fetching %s data from %s", base, exc_name)

                # Get ticker
                ticker = exchange.get_ticker(symbol)
                if not ticker:
                    logger.warning("No ticker data for %s on %s", symbol, exc_name)
                    continue

                # Get candles
                candles = exchange.get_candles(symbol, interval=self.config.candle_interval, limit=100)

                # Get order book
                order_book = exchange.get_order_book(symbol, depth=20)

                market_data[key] = {
                    "exchange": exc_name,
                    "symbol": symbol,
                    "base": base,
                    "ticker": ticker,
                    "candles": candles,
                    "order_book": order_book,
                }

                logger.info("Fetched %s from %s: last=%.2f, candles=%d",
                            base, exc_name, ticker.get("last", 0), len(candles))

        return market_data

    def fetch_sentiment(self) -> Dict[str, Any]:
        """Fetch market sentiment data (Fear & Greed Index, news).

        Returns:
            Dict with 'fear_and_greed' and 'headlines' keys.
        """
        from .brain import fetch_fear_and_greed, fetch_crypto_headlines

        sentiment: Dict[str, Any] = {}

        if self.config.fear_and_greed_enabled:
            sentiment["fear_and_greed"] = fetch_fear_and_greed()
        else:
            sentiment["fear_and_greed"] = {}

        if self.config.news_enabled:
            sentiment["headlines"] = fetch_crypto_headlines()
        else:
            sentiment["headlines"] = []

        return sentiment

    def ask_brain(self, market_data: Dict[str, Dict[str, Any]],
                  research_insights: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Ask the LLM brain for trading decisions based on current context.

        Returns:
            List of decision dicts from the LLM (or fallback).
        """
        decisions = self.brain.ask_brain(
            market_data=market_data,
            portfolio_mgr=self.portfolio,
            journal=self.journal,
            symbols=self.symbols,
            risk_config=self.config.risk,
            research_insights=research_insights,
        )
        return decisions

    def make_decisions(self, llm_decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute trading decisions from the LLM brain.

        Applies risk rules:
        - Max position size (20% of capital)
        - Max open positions (5)
        - Max drawdown (20%)

        Args:
            llm_decisions: List of decision dicts from PicsouBrain.

        Returns:
            List of executed decision dicts with results.
        """
        decisions: List[Dict[str, Any]] = []
        balance = self.portfolio.get_balance()
        open_positions = self.portfolio.get_position_count()

        # Check drawdown limit
        pnl = self.portfolio.get_pnl()
        drawdown_pct = 0.0
        if pnl["starting_capital"] > 0:
            drawdown_pct = max(0, -pnl["return_pct"] / 100.0)

        if drawdown_pct > self.config.risk.max_drawdown_pct:
            logger.warning("Max drawdown reached (%.1f%% > %.1f%%). Pausing trading.",
                           drawdown_pct * 100, self.config.risk.max_drawdown_pct * 100)
            return decisions

        for decision in llm_decisions:
            action = decision.get("action", "").lower()
            if action == "hold":
                logger.info("LLM decided HOLD for %s: %s",
                            decision.get("symbol", "?"), decision.get("reasoning", ""))
                # Log hold decisions too
                self.journal.log_decision(
                    exchange=decision.get("exchange", "unknown"),
                    symbol=decision.get("symbol", ""),
                    strategy=decision.get("strategy_type", "llm"),
                    action="hold",
                    reasoning=decision.get("reasoning", ""),
                    confidence=decision.get("confidence", 0.0),
                    llm_prompt=self.brain.last_prompt or "",
                    llm_response=self.brain.last_response or "",
                    llm_tokens=self.brain.last_tokens or {},
                )
                continue

            if action not in ("buy", "sell"):
                logger.warning("Unknown LLM action: %s", action)
                continue

            symbol_base = decision.get("symbol", "")
            exchange_name = decision.get("exchange", "okx")
            confidence = decision.get("confidence", 0.5)
            amount_pct = decision.get("amount_pct", 0.05)
            reasoning = decision.get("reasoning", "")
            strategy_type = decision.get("strategy_type", "llm_driven")

            # Apply self-reflection rules
            if hasattr(self, '_strategy_rules') and self._strategy_rules:
                rules = self._strategy_rules
                params = rules.get("parameter_changes", {})
                # Override position size if reflection says so
                if "preferred_position_pct" in params:
                    amount_pct = min(amount_pct, params["preferred_position_pct"])
                # Skip certain strategies if reflection penalizes them heavily
                adjustments = rules.get("strategy_adjustments", {})
                if strategy_type in adjustments:
                    adj = adjustments[strategy_type]
                    if adj.get("weight_penalty", 0) >= 0.5:
                        logger.info("Skipping %s: self-reflection penalized it", strategy_type)
                        continue

            # Validate exchange
            if exchange_name not in self.exchanges:
                exchange_name = list(self.exchanges.keys())[0]

            # Get the full symbol for the exchange
            symbol = self._get_exchange_symbol(exchange_name, symbol_base)

            # === BUY logic ===
            if action == "buy":
                # Check max open positions
                if open_positions >= self.config.risk.max_open_positions:
                    logger.info("Max open positions (%d) reached. Skipping %s buy.",
                                self.config.risk.max_open_positions, symbol_base)
                    continue

                # Position size: use amount_pct from LLM, capped by risk limits
                position_pct = min(amount_pct, self.config.risk.max_position_pct)
                position_size_eur = balance * position_pct

                if position_size_eur < 10:  # Minimum 10 EUR per trade
                    continue

                # Get current price
                ticker = self.exchanges[exchange_name].get_ticker(symbol)
                if not ticker or ticker.get("last", 0) == 0:
                    logger.warning("No price for %s on %s, skipping", symbol, exchange_name)
                    continue

                price = ticker["last"]
                amount = position_size_eur / price

                # Execute paper trade
                position = self.portfolio.open_position(
                    exchange=exchange_name,
                    symbol=symbol,
                    side="long",
                    amount=amount,
                    price=price,
                )

                executed_decision = {
                    "exchange": exchange_name,
                    "symbol": symbol,
                    "side": "buy",
                    "amount": round(amount, 8),
                    "price": price,
                    "size_eur": round(position_size_eur, 2),
                    "confidence": confidence,
                    "strategy_type": strategy_type,
                    "reasoning": reasoning,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                if position:
                    executed_decision["position_id"] = position.id
                    open_positions += 1
                    balance = self.portfolio.get_balance()

                # Log to journal
                self.journal.log_decision(
                    exchange=exchange_name,
                    symbol=symbol,
                    strategy=strategy_type,
                    action="buy",
                    reasoning=reasoning,
                    confidence=confidence,
                    amount=amount,
                    price=price,
                    llm_prompt=self.brain.last_prompt or "",
                    llm_response=self.brain.last_response or "",
                    llm_tokens=self.brain.last_tokens or {},
                )

                decisions.append(executed_decision)

            # === SELL logic ===
            elif action == "sell":
                # Find matching open positions by BASE currency across all exchanges.
                # LLM may say "sell ETH on okx" but position could be on bitstamp as "ETHusdt".
                # Match by normalizing to base symbol (ETH, SOL, BTC) so cross-exchange works.
                def _base_symbol(sym: str) -> str:
                    """Normalize exchange-specific symbol to base currency."""
                    import re
                    return re.sub(r'[-_/]?[Uu][Ss][Dd][Tt]$', '', sym)

                base = _base_symbol(symbol)
                matching_positions = [
                    p for p in self.portfolio.get_open_positions()
                    if _base_symbol(p.symbol) == base and p.side == "long"
                ]

                if not matching_positions:
                    logger.info("No matching long position to sell for %s (base=%s) on any exchange",
                                symbol, base)
                    continue

                # Close the first matching position using ITS exchange (not LLM's exchange)
                pos = matching_positions[0]
                sell_exchange = pos.exchange  # Use the exchange where position is actually held
                sell_symbol = pos.symbol      # Use the exchange-formatted symbol

                # Minimum hold time: 15 minutes (3 cycles). Rapid round-trips
                # lose money on fees — the LLM must let positions breathe.
                min_hold_seconds = 900  # 15 minutes
                try:
                    from datetime import datetime as _dt
                    open_dt = _dt.fromisoformat(pos.open_time)
                    close_dt = _dt.fromisoformat(datetime.now(timezone.utc).isoformat())
                    hold_duration = (close_dt - open_dt).total_seconds()
                    if hold_duration < min_hold_seconds:
                        remaining = int(min_hold_seconds - hold_duration) // 60
                        logger.warning(
                            "Rejecting sell for %s: held only %ds (min %ds, %d min remaining). "
                            "Rapid round-trips lose money on fees.",
                            sell_symbol, int(hold_duration), min_hold_seconds, remaining,
                        )
                        continue
                except Exception:
                    pass  # If timestamp parsing fails, allow the sell
                ticker = self.exchanges[sell_exchange].get_ticker(sell_symbol)
                if not ticker or ticker.get("last", 0) == 0:
                    logger.warning("No price for %s on %s, skipping sell",
                                   sell_symbol, sell_exchange)
                    continue

                price = ticker["last"]
                trade = self.portfolio.close_position(pos.id, price)

                executed_decision = {
                    "exchange": sell_exchange,
                    "symbol": sell_symbol,
                    "side": "sell",
                    "amount": pos.amount,
                    "price": price,
                    "size_eur": round(pos.amount * price, 2),
                    "confidence": confidence,
                    "strategy_type": strategy_type,
                    "reasoning": reasoning,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                if trade:
                    executed_decision["trade_pnl"] = round(trade.pnl, 2)
                    balance = self.portfolio.get_balance()

                # Log to journal (use actual exchange/symbol where position was held)
                self.journal.log_decision(
                    exchange=sell_exchange,
                    symbol=sell_symbol,
                    strategy=strategy_type,
                    action="sell",
                    reasoning=reasoning,
                    confidence=confidence,
                    amount=pos.amount,
                    price=price,
                    llm_prompt=self.brain.last_prompt or "",
                    llm_response=self.brain.last_response or "",
                    llm_tokens=self.brain.last_tokens or {},
                )

                decisions.append(executed_decision)

        return decisions

    def evaluate_learning(self) -> Optional[Dict[str, Any]]:
        """Run learning evaluation on LLM decision categories.

        Evaluates win rates by strategy_type (LLM-generated categories)
        using both CLOSED TRADES (realized PnL) and journal decision stats.
        This ensures learning can happen even with few closed trades.
        """
        # Count total data points: closed trades + journal entries
        total_closed = len(self.portfolio.trades)
        recent = self.journal.get_recent(limit=5000)
        total_journal_decisions = len(recent)

        # Use a lower threshold if we have journal data (decisions = learning signal)
        min_required = max(1, self.config.learning.min_trades)
        has_enough_data = (total_closed >= min_required or
                           total_journal_decisions >= min_required)

        if not has_enough_data:
            logger.info("Not enough data for learning evaluation "
                        "(closed_trades=%d, journal_decisions=%d, min=%d)",
                        total_closed, total_journal_decisions, min_required)
            return None

        # Build a lookup: close_time -> strategy_type from journal
        # Index both sell entries (closed trades) AND buy entries (open positions)
        strategy_lookup: Dict[str, str] = {}
        for entry in recent:
            ts = entry.get("timestamp", "")
            if ts and entry.get("action") in ("buy", "sell"):
                strategy_lookup[ts[:19]] = entry.get("strategy", "llm_driven")

        # 1. Closed trades with realized PnL (primary signal)
        trades_by_strategy: Dict[str, List[Dict[str, Any]]] = {}
        for trade in self.portfolio.trades:
            close_ts = trade.close_time[:19] if trade.close_time else ""
            strategy = strategy_lookup.get(close_ts, "llm_driven")
            if strategy not in trades_by_strategy:
                trades_by_strategy[strategy] = []
            trades_by_strategy[strategy].append({"pnl": trade.pnl})

        # 2. Open positions with unrealized PnL estimation (secondary signal)
        # This gives us more data for learning even without closed trades
        for pos in self.portfolio.get_open_positions():
            # Find matching journal buy entry for this position
            pos_strategy = "llm_driven"
            for entry in recent:
                if (entry.get("action") == "buy" and
                        entry.get("symbol", "").startswith(pos.symbol.replace("-", "").replace("USDT", "").upper()) and
                        entry.get("timestamp", "")):
                    pos_strategy = entry.get("strategy", "llm_driven")
                    break

            # Estimate unrealized PnL (negative = loss, includes fees)
            unrealized_pnl = -(pos.fee)  # At minimum, fees are a loss
            if pos_strategy not in trades_by_strategy:
                trades_by_strategy[pos_strategy] = []
            trades_by_strategy[pos_strategy].append({"pnl": unrealized_pnl})

        # Update learning with all available trade data
        for strategy_name, trade_list in trades_by_strategy.items():
            self.learning.update_from_trades(
                strategy_name=strategy_name,
                trades=trade_list,
            )

        # Also count hold decisions per strategy, but do NOT inflate trade counts
        # Holds are non-events — they should not count as real trades for win rate
        hold_by_strategy: Dict[str, int] = {}
        for entry in recent:
            if entry.get("action") == "hold":
                strategy = entry.get("strategy", "risk_management")
                hold_by_strategy[strategy] = hold_by_strategy.get(strategy, 0) + 1

        # Only update learning with actual (non-hold) trade data for strategies
        # that have real trades. Skip strategies that only have holds — they
        # don't have meaningful PnL data.
        for strategy_name, trade_list in trades_by_strategy.items():
            # Filter out zero-pnl entries that came from holds
            real_trades = [t for t in trade_list if t.get("pnl", 0) != 0.0 or t.get("is_hold", False) is not True]
            if real_trades:
                self.learning.update_from_trades(
                    strategy_name=strategy_name,
                    trades=real_trades,
                )

        result = self.learning.evaluate_strategies()
        logger.info("Learning evaluation: eliminated=%s, promoted=%s",
                     result.get("eliminated", []),
                     result.get("promoted", []))
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of current state for dashboard."""
        pnl = self.portfolio.get_pnl()
        stats = self.journal.get_stats()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": self.config.phase,
            "portfolio": {
                "balance": round(self.portfolio.get_balance(), 2),
                "real_capital": self.config.real_capital,
                "simulation_multiplier": self.config.starting_capital / self.config.real_capital,
                "open_positions": self.portfolio.get_position_count(),
                "pnl": pnl,
            },
            "journal_stats": stats,
            "learning": {
                "evaluation_count": self.learning.evaluation_count,
                "active_strategies": self.learning.get_active_strategies(),
                "strategy_weights": {
                    name: round(score.weight, 4)
                    for name, score in self.learning.scores.items()
                },
            },
            "brain": {
                "llm_model": self.config.llm_model,
                "llm_url": self.config.llm_url,
                "last_tokens": self.brain.last_tokens,
            },
            "exchanges": list(self.exchanges.keys()),
            "symbols": self.symbols,
            "exploration_phase": self.config.exploration_phase,
            "underexplored_strategies": self.learning.get_underexplored_strategies(),
        }

    def _generate_exploration_trades(self, market_data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate small exploratory trades for under-explored strategies.

        Uses simple EMA crossover signals as fallback to produce trades
        when the LLM refuses to trade. Positions are sized at exploration_position_pct
        (typically 2%) to limit risk during exploration.

        Returns:
            List of decision dicts for under-explored strategies.
        """
        underexplored = self.learning.get_underexplored_strategies()
        if not underexplored:
            return []

        # Pick the most under-explored strategy first
        # Sort by total_trades ascending (none = 0)
        def strategy_trade_count(name: str) -> int:
            if name in self.learning.scores:
                return self.learning.scores[name].total_trades
            return 0

        underexplored.sort(key=strategy_trade_count)
        target_strategy = underexplored[0]

        logger.info("Exploration: strategy '%s' has insufficient data, generating fallback trade",
                     target_strategy)

        # Use EMA fallback signal on available market data
        from .brain import fallback_ema_signal

        exploration_trades = []
        for key, md in market_data.items():
            candles = md.get("candles", [])
            symbol = md.get("base", "")
            exchange = md.get("exchange", "")

            sig = fallback_ema_signal(candles, symbol, exchange)
            if sig:
                # Override strategy_type to the under-explored strategy
                sig["strategy_type"] = target_strategy
                # Use small position size for exploration
                sig["amount_pct"] = self.config.exploration_position_pct
                sig["reasoning"] = (
                    f"EXPLORATION: Testing strategy '{target_strategy}' with EMA fallback. "
                    f"Original signal: {sig.get('reasoning', '')}"
                )
                exploration_trades.append(sig)
                break  # One trade per exploration cycle is enough

        return exploration_trades

    def run_once(self) -> Dict[str, Any]:
        """Run one iteration of the main agent loop.

        Returns summary of actions taken.
        """
        logger.info("=== Picsou agent loop started ===")

        # Load self-reflection rules (hot-reload from disk each cycle)
        self.reflector.load_rules()
        self._strategy_rules = self.reflector.current_rules

        # 1. Fetch market data
        logger.info("Fetching market data...")
        market_data = self.fetch_market_data()
        logger.info("Fetched data for %d market pairs", len(market_data))

        # 2. Fetch sentiment
        logger.info("Fetching sentiment data...")
        sentiment = self.fetch_sentiment()
        logger.info("Sentiment: F&G=%s, headlines=%d",
                     sentiment.get("fear_and_greed", {}).get("value", "N/A"),
                     len(sentiment.get("headlines", [])))

        # 3. Ask the LLM brain (with exploration context if needed)
        # Build learning context for the prompt
        learning_context = self.learning.get_learning_context()
        underexplored = learning_context.get("underexplored_strategies", [])
        in_exploration_phase = (
            self.config.exploration_phase and len(underexplored) > 0
        )

        # Inject exploration context into the brain
        self.brain.exploration_mode = in_exploration_phase
        self.brain.underexplored_strategies = underexplored
        self.brain.learning_context = learning_context

        # 2.5 Fetch strategy research insights (web-sourced)
        research_insights = None
        try:
            research_insights = self.strategy_researcher.fetch_strategy_insights(
                symbols=self.symbols
            )
            logger.info("Research insights: strategies=%s, signals=%s",
                         research_insights.get("trending_strategies", []),
                         research_insights.get("technical_signals", []))
        except Exception as e:
            logger.warning("Strategy research failed (non-critical): %s", e)

        logger.info("Asking LLM brain for decisions (exploration=%s, under-explored=%s)...",
                     in_exploration_phase, underexplored)
        llm_decisions = self.ask_brain(market_data, research_insights=research_insights)
        logger.info("LLM brain returned %d decisions", len(llm_decisions))

        # 3.5 Exploration fallback: if LLM returned no decisions or only holds
        # and we're in exploration phase, generate trades for under-explored strategies
        real_decisions = [d for d in llm_decisions if d.get("action", "").lower() != "hold"]
        if in_exploration_phase and len(real_decisions) == 0:
            logger.info("LLM returned no actionable decisions — generating exploration trades")
            exploration_trades = self._generate_exploration_trades(market_data)
            if exploration_trades:
                logger.info("Generated %d exploration trade(s) for under-explored strategies",
                            len(exploration_trades))
                llm_decisions.extend(exploration_trades)
            else:
                logger.info("No EMA fallback signal available for exploration")

        # 3.6 If STILL no decisions, log holds (but NOT as generic risk_management)
        # In exploration mode, we avoid flooding the journal with risk_management holds
        if not llm_decisions:
            if in_exploration_phase:
                # In exploration mode, don't log holds at all — they pollute the learning data
                logger.info("No decisions this cycle (exploration mode — skipping hold logging)")
            else:
                logger.info("LLM returned no decisions — logging hold for all symbols")
                for key, md in market_data.items():
                    self.journal.log_decision(
                        exchange=md.get("exchange", "unknown"),
                        symbol=md.get("symbol", ""),
                        strategy="risk_management",
                        action="hold",
                        reasoning="LLM returned no decisions (market conditions unclear or Extreme Fear)",
                        confidence=0.3,
                        llm_prompt=self.brain.last_prompt or "",
                        llm_response=self.brain.last_response or "",
                        llm_tokens=self.brain.last_tokens or {},
                    )

        # 4. Execute decisions (with risk management)
        logger.info("Making decisions...")
        decisions = self.make_decisions(llm_decisions)
        logger.info("Executed %d decisions", len(decisions))

        # 5. Learning evaluation (periodic)
        eval_result = self.evaluate_learning()

        # 5.5 Auto-disable exploration phase when all strategies have enough data
        if self.config.exploration_phase:
            target_strategies = ["contrarian", "breakout", "dca", "momentum", "mean_reversion"]
            still_underexplored = self.learning.get_underexplored_strategies(target_strategies)
            if not still_underexplored:
                logger.info("All target strategies have >= %d trades — disabling exploration phase",
                            self.config.learning.min_exploration_trades)
                self.config.exploration_phase = False

        # 5.6 Self-reflection every 48 cycles (~4 hours at 5-min intervals)
        # Only starts after enough trading data has been collected
        self.reflection_counter += 1
        reflection_result = None
        min_trades_for_reflection = 20  # Need real experience before reflecting
        total_closed_trades = len(self.portfolio.trades) if hasattr(self, 'portfolio') else 0
        ready_to_reflect = (
            self.reflection_counter >= 48
            and total_closed_trades >= min_trades_for_reflection
        )
        if ready_to_reflect:
            self.reflection_counter = 0
            try:
                reflection_result = self.reflector.reflect()
                if reflection_result:
                    # Hot-reload brain with new rules
                    self.brain.load_strategy_rules()
                    # Update local rules cache
                    self._strategy_rules = self.reflector.current_rules
                    logger.info("Self-reflection completed: %s changes (trades=%d)",
                                len(reflection_result.get("changes", [])),
                                total_closed_trades)
            except Exception as e:
                logger.error("Self-reflection failed: %s", e)
        elif self.reflection_counter >= 48 and total_closed_trades < min_trades_for_reflection:
            logger.info("Self-reflection skipped: only %d closed trades (need %d)",
                        total_closed_trades, min_trades_for_reflection)
            # Don't reset counter — keep counting until we have enough data

        # 6. Get summary
        summary = self.get_summary()
        summary["decisions"] = decisions
        summary["llm_decisions_count"] = len(llm_decisions)
        summary["sentiment"] = sentiment
        if eval_result:
            summary["learning_evaluation"] = eval_result
        if reflection_result:
            summary["self_reflection"] = reflection_result

        logger.info("=== Picsou agent loop completed: %d decisions ===",
                     len(decisions))

        # Persist brain status for dashboard to read
        try:
            brain_status = self.brain.get_config_status()
            brain_status_file = self.config.paths.data / "brain_status.json"
            with open(brain_status_file, "w", encoding="utf-8") as f:
                json.dump(brain_status, f, indent=2)
        except Exception as e:
            logger.warning("Failed to persist brain status: %s", e)

        return summary

    def run(self, iterations: int = 0) -> None:
        """Run the main agent loop continuously.

        Args:
            iterations: Number of iterations to run. 0 = infinite.
        """
        logger.info("Starting Picsou agent in %s mode (interval=%ds, llm=%s)",
                     self.config.phase, self.config.loop_interval,
                     self.config.llm_model)

        count = 0
        while True:
            try:
                self.run_once()
            except Exception as e:
                logger.error("Agent loop error: %s", e, exc_info=True)

            count += 1
            if iterations > 0 and count >= iterations:
                logger.info("Completed %d iterations, stopping", count)
                break

            logger.info("Sleeping %d seconds until next iteration...",
                        self.config.loop_interval)
            time.sleep(self.config.loop_interval)


def setup_logging(log_path: Path, level: int = logging.INFO) -> None:
    """Configure logging for the Picsou agent."""
    log_path.mkdir(parents=True, exist_ok=True)

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_path / "picsou.log"),
    ]

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def main() -> None:
    """Entry point for running Picsou agent."""
    config = get_config()
    setup_logging(config.paths.logs)

    agent = PicsouAgent(config)
    summary = agent.run_once()

    # Output summary
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()