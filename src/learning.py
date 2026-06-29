"""Learning engine for Picsou - evaluates LLM decision categories.

Instead of evaluating 4 fixed strategies, this module evaluates the
strategy_type categories that the LLM assigns to its own decisions
(e.g. "momentum", "contrarian", "dca", "risk_management", etc.).
"""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .backtest import Backtester

logger = logging.getLogger(__name__)


class StrategyScore:
    """Score for a single decision category (strategy_type)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.win_rate: float = 0.0
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.avg_profit: float = 0.0
        self.total_profit: float = 0.0
        self.max_drawdown: float = 0.0
        self.sharpe_ratio: float = 0.0
        self.returns: List[float] = []  # Per-trade returns
        self.weight: float = 1.0  # Selection weight
        self.active: bool = True  # Whether strategy type is still in play
        self.probation: bool = False  # On probation (trial period before full reactivation)
        self.probation_trades: int = 0  # Number of trades made during probation
        self._prev_total_trades: int = 0  # Internal: track total_trades delta for probation

    # Maximum position size as % of portfolio for strategies on probation
    PROBATION_MAX_POSITION_PCT: float = 0.15  # 15% vs 3% normal
    # Number of trades required before probation evaluation
    PROBATION_TRADE_LIMIT: int = 3
    # Win rate required to pass probation
    PROBATION_WIN_RATE_THRESHOLD: float = 0.40

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_profit": self.avg_profit,
            "total_profit": self.total_profit,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "weight": self.weight,
            "active": self.active,
            "probation": self.probation,
            "probation_trades": self.probation_trades,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyScore":
        s = cls(d["name"])
        s.win_rate = d.get("win_rate", 0.0)
        s.total_trades = d.get("total_trades", 0)
        s.winning_trades = d.get("winning_trades", 0)
        s.losing_trades = d.get("losing_trades", 0)
        s.avg_profit = d.get("avg_profit", 0.0)
        s.total_profit = d.get("total_profit", 0.0)
        s.max_drawdown = d.get("max_drawdown", 0.0)
        s.sharpe_ratio = d.get("sharpe_ratio", 0.0)
        s.weight = d.get("weight", 1.0)
        s.active = d.get("active", True)
        s.probation = d.get("probation", False)
        s.probation_trades = d.get("probation_trades", 0)
        return s


class LearningEngine:
    """Evaluates LLM decision categories and adjusts their weights.

    - Tracks performance per strategy_type (LLM-assigned categories)
    - Scores categories: win_rate, sharpe_ratio, max_drawdown, avg_profit
    - Eliminates categories with win_rate < 50% or max_drawdown > 30%
    - Reinforces winners by increasing their weight
    - Saves learning state to disk
    """

    def __init__(self, data_path: Path = Path("/root/PROJECTS/picsou/data"),
                 win_rate_threshold: float = 0.55,
                 min_trades: int = 3,
                 min_days: int = 1,
                 elimination_win_rate: float = 0.50,
                 elimination_max_drawdown: float = 0.30,
                 min_exploration_trades: int = 3,
                 max_weight_pct: float = 0.40,
                 starting_capital: float = 10000.0) -> None:
        self.data_path = data_path
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.file_path = data_path / "learning.json"
        self.win_rate_threshold = win_rate_threshold
        self.min_trades = min_trades
        self.min_days = min_days
        self.elimination_win_rate = elimination_win_rate
        self.elimination_max_drawdown = elimination_max_drawdown
        self.min_exploration_trades = min_exploration_trades
        self.max_weight_pct = max_weight_pct
        self.starting_capital = starting_capital

        # Decision category scores
        self.scores: Dict[str, StrategyScore] = {}
        self.evaluation_count: int = 0
        self.last_evaluation: Optional[str] = None

        self._load()

    def _load(self) -> None:
        """Load learning state from disk."""
        if self.file_path.exists():
            try:
                data = json.loads(self.file_path.read_text())
                self.evaluation_count = data.get("evaluation_count", 0)
                self.last_evaluation = data.get("last_evaluation")
                scores_data = data.get("scores", {})
                self.scores = {
                    name: StrategyScore.from_dict(sd)
                    for name, sd in scores_data.items()
                }
                logger.info("Loaded learning state: %d categories scored",
                            len(self.scores))
            except Exception as e:
                logger.error("Failed to load learning state: %s", e)

    def _save(self) -> None:
        """Save learning state to disk."""
        data = {
            "evaluation_count": self.evaluation_count,
            "last_evaluation": self.last_evaluation,
            "scores": {name: score.to_dict() for name, score in self.scores.items()},
        }
        self.file_path.write_text(json.dumps(data, indent=2))
        logger.debug("Learning state saved to %s", self.file_path)

    def update_from_trades(self, strategy_name: str,
                           trades: List[Dict[str, Any]]) -> None:
        """Update category score from trade results.

        Recalculates all statistics from the full trade list rather than
        accumulating, to avoid double-counting when evaluate_learning
        is called multiple times.

        Args:
            strategy_name: LLM-assigned strategy_type category.
            trades: List of trade dicts with 'pnl' key.
        """
        if strategy_name not in self.scores:
            self.scores[strategy_name] = StrategyScore(strategy_name)

        score = self.scores[strategy_name]
        returns: List[float] = []

        # Reset and recalculate from full trade list to avoid accumulation
        score.total_trades = 0
        score.winning_trades = 0
        score.losing_trades = 0
        score.total_profit = 0.0
        score.returns = []

        for trade in trades:
            pnl = trade.get("pnl", 0.0)
            returns.append(pnl)
            score.total_trades += 1
            score.total_profit += pnl
            if pnl > 0:
                score.winning_trades += 1
            else:
                score.losing_trades += 1

        if score.total_trades > 0:
            score.win_rate = score.winning_trades / score.total_trades
            score.avg_profit = score.total_profit / score.total_trades

        # Track probation trades: if this strategy is on probation, count this trade
        # Note: probation_trades is a cumulative counter that resets when probation starts.
        # Since update_from_trades is called with the full trade list each time,
        # we only increment probation_trades if the strategy is currently on probation
        # and this is a new batch of trades (detected by checking if total_trades increased).
        if score.probation:
            old_total = getattr(score, '_prev_total_trades', 0)
            new_trades = score.total_trades - old_total
            if new_trades > 0:
                score.probation_trades += new_trades
            score._prev_total_trades = score.total_trades

        # Calculate max drawdown from returns
        score.returns = returns
        score.max_drawdown = self._calculate_max_drawdown(score.returns, starting_capital=self.starting_capital)
        score.sharpe_ratio = self._calculate_sharpe_ratio(score.returns)

        logger.info("Updated %s: win_rate=%.2f, trades=%d, sharpe=%.2f, "
                     "max_dd=%.2f",
                     strategy_name, score.win_rate, score.total_trades,
                     score.sharpe_ratio, score.max_drawdown)

    @staticmethod
    def _calculate_max_drawdown(returns: List[float], starting_capital: float = 10000.0) -> float:
        """Calculate maximum drawdown from a list of PnL returns (in dollars).

        Returns the maximum drawdown as a positive decimal (e.g. 0.20 = 20%).
        Uses starting_capital to convert absolute PnL to percentage drawdown.
        """
        if not returns:
            return 0.0

        # Convert PnL returns to cumulative equity curve starting from starting_capital
        cumulative = [starting_capital]
        for r in returns:
            cumulative.append(cumulative[-1] + r)

        peak = cumulative[0]
        max_dd = 0.0

        for value in cumulative:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, drawdown)

        return max_dd

    @staticmethod
    def _calculate_sharpe_ratio(returns: List[float],
                                 risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio from returns.

        Assumes returns are daily for annualization (sqrt(365) for crypto).
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = [r - risk_free_rate for r in returns]
        mean_ret = sum(excess_returns) / len(excess_returns)
        variance = sum((r - mean_ret) ** 2 for r in excess_returns) / len(excess_returns)
        std_ret = math.sqrt(variance) if variance > 0 else 0.0

        if std_ret == 0:
            return 0.0

        # Annualize (crypto trades 365 days/year)
        sharpe = (mean_ret / std_ret) * math.sqrt(365)
        return sharpe

    def evaluate_strategies(self) -> Dict[str, Any]:
        """Evaluate all decision categories and adjust weights.

        - Eliminates categories with win_rate < threshold or max_drawdown > limit
        - Auto-eliminates strategies with 0% win rate after >=3 trades
        - Reinforces winners by increasing weight
        - Applies self-reflect reactivation rules (weight_boost + active=true)
        - Caps individual strategy weight at max_weight_pct (default 40%)
        - Returns evaluation summary.
        """
        self.evaluation_count += 1
        self.last_evaluation = datetime.now(timezone.utc).isoformat()

        eliminated: List[str] = []
        promoted: List[str] = []
        reactivated: List[str] = []

        for name, score in self.scores.items():
            if not score.active:
                continue

            # Need minimum trades to evaluate — skip strategies with insufficient data
            if score.total_trades < self.min_trades:
                logger.info("%s: insufficient trades (%d < %d), skipping evaluation",
                            name, score.total_trades, self.min_trades)
                continue

            # NEVER eliminate a strategy that hasn't been explored enough
            # (even if min_trades threshold is met, check exploration minimum)
            if score.total_trades < self.min_exploration_trades:
                logger.info("%s: under-explored (%d < %d min_exploration_trades), protecting from elimination",
                            name, score.total_trades, self.min_exploration_trades)
                continue

            # Auto-eliminate strategies with 0% win rate after >=3 trades
            if score.win_rate == 0.0 and score.total_trades >= 3:
                score.active = False
                score.weight = 0.0
                score.probation = False  # Eliminated, not probation
                score.probation_trades = 0
                eliminated.append(name)
                logger.info("%s: ELIMINATED (0%% win rate after %d trades)",
                            name, score.total_trades)
                continue

            # Elimination criteria
            if score.win_rate < self.elimination_win_rate:
                score.active = False
                score.weight = 0.0
                score.probation = False  # Mark for probation eligibility
                score.probation_trades = 0
                eliminated.append(name)
                logger.info("%s: ELIMINATED (win_rate=%.2f < %.2f)",
                            name, score.win_rate, self.elimination_win_rate)
                continue

            if score.max_drawdown > self.elimination_max_drawdown:
                score.active = False
                score.weight = 0.0
                score.probation = False
                score.probation_trades = 0
                eliminated.append(name)
                logger.info("%s: ELIMINATED (max_drawdown=%.2f > %.2f)",
                            name, score.max_drawdown, self.elimination_max_drawdown)
                continue

            # Promote winners
            if score.win_rate >= self.win_rate_threshold and score.sharpe_ratio > 0.5:
                # Increase weight proportionally to performance
                performance_score = score.win_rate * max(score.sharpe_ratio, 0.1)
                score.weight = min(performance_score * 2.0, 5.0)  # Cap at 5x
                promoted.append(name)
                logger.info("%s: PROMOTED (weight=%.2f, win_rate=%.2f, sharpe=%.2f)",
                            name, score.weight, score.win_rate, score.sharpe_ratio)

        # Apply self-reflect reactivation rules: if a rule says "Reactivate X"
        # with a weight_boost, validate via backtest first. If backtest passes,
        # put the strategy on probation (not direct reactivation). If backtest
        # fails, the strategy stays eliminated.
        try:
            rules_path = self.data_path / "strategy_rules.json"
            if rules_path.exists():
                rules_data = json.loads(rules_path.read_text())
                adjustments = rules_data.get("strategy_adjustments", {})
                for name, adj in adjustments.items():
                    boost = float(adj.get("weight_boost", 0))
                    if boost > 0 and name in self.scores:
                        score = self.scores[name]
                        if not score.active:
                            # Run backtest before reactivating
                            bt_result = self.backtest_strategy(name)
                            if bt_result.get("pass", False):
                                # Backtest passed — put on probation, not direct reactivation
                                score.active = True
                                score.probation = True
                                score.probation_trades = 0
                                score.weight = max(score.weight, boost)
                                reactivated.append(name)
                                logger.info(
                                    "%s: PROBATION ACTIVATION by self-reflect rule "
                                    "(backtest passed: WR=%.2f, avg_profit=%.6f, "
                                    "weight_boost=%.2f). On probation for %d trades.",
                                    name, bt_result.get("win_rate", 0),
                                    bt_result.get("avg_profit", 0), boost,
                                    StrategyScore.PROBATION_TRADE_LIMIT,
                                )
                            else:
                                # Backtest failed — strategy stays eliminated
                                logger.info(
                                    "%s: REACTIVATION BLOCKED — backtest failed "
                                    "(WR=%.2f < 0.30 or avg_profit <= 0). "
                                    "Strategy remains eliminated.",
                                    name, bt_result.get("win_rate", 0),
                                )
                        else:
                            # Already active, just apply boost (capped if on probation)
                            if score.probation:
                                # On probation: cap weight boost at a lower level
                                capped_boost = min(boost, 0.15)
                                score.weight = max(score.weight, score.weight + capped_boost)
                                logger.info(
                                    "%s: PROBATION BOOST (capped) by self-reflect rule "
                                    "(weight_boost=%.2f capped to %.2f, new_weight=%.2f)",
                                    name, boost, capped_boost, score.weight,
                                )
                            else:
                                score.weight = max(score.weight, score.weight + boost)
                                logger.info(
                                    "%s: BOOSTED by self-reflect rule (weight_boost=%.2f, new_weight=%.2f)",
                                    name, boost, score.weight,
                                )
        except Exception as e:
            logger.warning("Failed to apply self-reflect reactivation rules: %s", e)

        # Evaluate probation: check if any strategies on probation have completed
        # their trial period (3 trades) and promote or re-eliminate them
        probation_evaluated = self.evaluate_probation()
        if probation_evaluated:
            logger.info("Probation evaluated: %s", probation_evaluated)

        # Normalize weights so they sum to 1.0 among active categories
        total_weight = sum(s.weight for s in self.scores.values() if s.active)
        if total_weight > 0:
            for score in self.scores.values():
                if score.active:
                    score.weight = score.weight / total_weight

        # Cap individual strategy weight at max_weight_pct (e.g. 40%)
        # Redistribute excess weight proportionally among other active strategies
        max_weight = self.max_weight_pct
        for _ in range(5):  # Iterate to converge (rarely needs more than 2-3 rounds)
            overweight = [(name, s) for name, s in self.scores.items()
                          if s.active and s.weight > max_weight]
            if not overweight:
                break
            for name, s in overweight:
                excess = s.weight - max_weight
                s.weight = max_weight
                # Distribute excess to other active strategies proportionally
                others_total = sum(sc.weight for n, sc in self.scores.items()
                                   if sc.active and n != name)
                if others_total > 0:
                    for n2, sc in self.scores.items():
                        if sc.active and n2 != name:
                            sc.weight += excess * (sc.weight / others_total)
            # Renormalize after redistribution
            total_weight = sum(s.weight for s in self.scores.values() if s.active)
            if total_weight > 0:
                for score in self.scores.values():
                    if score.active:
                        score.weight = score.weight / total_weight

        self._save()

        return {
            "evaluation_count": self.evaluation_count,
            "eliminated": eliminated,
            "promoted": promoted,
            "reactivated": reactivated,
            "active_strategies": [
                name for name, score in self.scores.items() if score.active
            ],
            "strategy_weights": {
                name: round(score.weight, 4)
                for name, score in self.scores.items() if score.active
            },
        }

    def get_strategy_weight(self, strategy_name: str) -> float:
        """Get the current weight for a decision category."""
        if strategy_name in self.scores:
            return self.scores[strategy_name].weight
        return 1.0  # Default weight for new categories

    def get_active_strategies(self) -> List[str]:
        """Get list of active decision category names."""
        return [name for name, score in self.scores.items() if score.active]

    def get_underexplored_strategies(self, target_strategies: List[str] = None) -> List[str]:
        """Get strategies that are active but have fewer trades than min_exploration_trades.

        Args:
            target_strategies: List of strategy names to check. If None, checks all active strategies.

        Returns:
            List of strategy names that need more exploration trades.
        """
        # Default target strategies if none specified
        if target_strategies is None:
            target_strategies = ["contrarian", "breakout", "dca", "momentum",
                                 "mean_reversion", "llm_driven"]

        underexplored = []
        for name in target_strategies:
            if name not in self.scores:
                # Strategy has never been seen — definitely under-explored
                underexplored.append(name)
            else:
                score = self.scores[name]
                if score.active and score.total_trades < self.min_exploration_trades:
                    underexplored.append(name)
        return underexplored

    def get_learning_context(self) -> Dict[str, Any]:
        """Get current learning state context for the LLM prompt.

        Returns dict with strategy weights, win rates, and under-explored strategies.
        """
        strategy_info = {}
        for name, score in self.scores.items():
            strategy_info[name] = {
                "weight": round(score.weight, 4),
                "win_rate": round(score.win_rate, 4),
                "total_trades": score.total_trades,
                "active": score.active,
                "probation": score.probation,
                "probation_trades": score.probation_trades,
            }

        return {
            "strategy_scores": strategy_info,
            "active_strategies": self.get_active_strategies(),
            "underexplored_strategies": self.get_underexplored_strategies(),
        }

    def backtest_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """Run a backtest for a strategy using historical journal data.

        Validates that a strategy has win_rate > 30% and avg_profit > 0
        on historical data before (re)activating it.

        Args:
            strategy_name: The strategy to backtest.

        Returns:
            Dict with keys: pass, win_rate, avg_profit, trades_simulated,
            max_drawdown, sharpe_ratio.
        """
        backtester = Backtester(
            data_path=self.data_path,
            win_rate_threshold=0.30,
            avg_profit_threshold=0.0,
        )
        result = backtester.backtest_strategy(strategy_name)
        return result.to_dict()

    def evaluate_probation(self) -> List[str]:
        """Evaluate all strategies currently on probation.

        For each strategy on probation with >= 3 trades:
        - If win_rate >= 40% → promote (probation=False, keep active)
        - If win_rate < 40% → re-eliminate definitively (active=False, probation=False, weight=0)

        Returns:
            List of strategy names that were evaluated.
        """
        evaluated = []
        for name, score in list(self.scores.items()):
            if not score.probation:
                continue

            if score.probation_trades < StrategyScore.PROBATION_TRADE_LIMIT:
                logger.info(
                    "%s: on probation (%d/%d trades), waiting for more data",
                    name, score.probation_trades, StrategyScore.PROBATION_TRADE_LIMIT,
                )
                continue

            # Enough probation trades — evaluate
            if score.win_rate >= StrategyScore.PROBATION_WIN_RATE_THRESHOLD:
                # Promote: probation passed
                score.probation = False
                score.probation_trades = 0
                # Assign a reasonable starting weight
                score.weight = 1.0
                score.active = True
                logger.info(
                    "%s: PROBATION PASSED (win_rate=%.2f >= %.2f). Promoted to active.",
                    name, score.win_rate, StrategyScore.PROBATION_WIN_RATE_THRESHOLD,
                )
            else:
                # Re-eliminate: probation failed
                score.probation = False
                score.probation_trades = 0
                score.active = False
                score.weight = 0.0
                logger.info(
                    "%s: PROBATION FAILED (win_rate=%.2f < %.2f). Re-eliminated.",
                    name, score.win_rate, StrategyScore.PROBATION_WIN_RATE_THRESHOLD,
                )

            evaluated.append(name)

        if evaluated:
            # Renormalize weights after probation changes
            total_weight = sum(s.weight for s in self.scores.values() if s.active)
            if total_weight > 0:
                for score in self.scores.values():
                    if score.active:
                        score.weight = score.weight / total_weight

            # Cap individual strategy weight at max_weight_pct
            max_weight = self.max_weight_pct
            for _ in range(5):
                overweight = [(n, s) for n, s in self.scores.items()
                              if s.active and s.weight > max_weight]
                if not overweight:
                    break
                for n, s in overweight:
                    excess = s.weight - max_weight
                    s.weight = max_weight
                    others_total = sum(sc.weight for n2, sc in self.scores.items()
                                       if sc.active and n2 != n)
                    if others_total > 0:
                        for n2, sc in self.scores.items():
                            if sc.active and n2 != n:
                                sc.weight += excess * (sc.weight / others_total)
                total_weight = sum(s.weight for s in self.scores.values() if s.active)
                if total_weight > 0:
                    for score in self.scores.values():
                        if score.active:
                            score.weight = score.weight / total_weight

            self._save()

        return evaluated

    def get_position_size_pct(self, strategy_name: str) -> float:
        """Get the maximum position size percentage for a strategy.

        Strategies on probation are limited to 15% instead of the normal max.

        Args:
            strategy_name: The strategy name.

        Returns:
            Maximum position size as a decimal (e.g. 0.15 = 15%).
        """
        if strategy_name in self.scores:
            score = self.scores[strategy_name]
            if score.probation:
                return StrategyScore.PROBATION_MAX_POSITION_PCT
        # Normal max position from risk config (default 20%)
        return 0.20

    def run_backtest(self, strategy_signals: List[Dict[str, Any]],
                     initial_capital: float = 10000.0,
                     fee_rate: float = 0.001) -> Dict[str, Any]:
        """Run a simple backtest on historical signals.

        Args:
            strategy_signals: List of signal dicts with 'action', 'price', 'amount'.
            initial_capital: Starting capital for backtest.
            fee_rate: Trading fee rate.

        Returns:
            Backtest results dict.
        """
        capital = initial_capital
        position = 0.0  # Amount of asset held
        trades: List[Dict[str, Any]] = []
        equity_curve: List[float] = [initial_capital]

        for signal in strategy_signals:
            action = signal.get("action", "hold")
            price = signal.get("price", 0)
            amount = signal.get("amount", 0)

            if action == "buy" and price > 0 and amount > 0:
                cost = amount * price
                fee = cost * fee_rate
                if capital >= cost + fee:
                    capital -= cost + fee
                    position += amount
                    trades.append({
                        "action": "buy",
                        "price": price,
                        "amount": amount,
                        "fee": fee,
                        "pnl": 0,
                    })

            elif action == "sell" and price > 0 and position > 0:
                revenue = position * price
                fee = revenue * fee_rate
                capital += revenue - fee
                pnl = (price - trades[-1]["price"]) * position if trades else 0
                trades.append({
                    "action": "sell",
                    "price": price,
                    "amount": position,
                    "fee": fee,
                    "pnl": pnl,
                })
                position = 0

            # Track equity
            equity = capital + position * price
            equity_curve.append(equity)

        final_equity = capital + position * (strategy_signals[-1]["price"] if strategy_signals else 0)
        total_pnl = final_equity - initial_capital
        return_pct = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0

        winning_trades = sum(1 for t in trades if t.get("pnl", 0) > 0)
        total_trades = len([t for t in trades if t["action"] == "sell"])

        return {
            "initial_capital": initial_capital,
            "final_equity": round(final_equity, 2),
            "total_pnl": round(total_pnl, 2),
            "return_pct": round(return_pct, 2),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0.0,
            "equity_curve": equity_curve,
        }