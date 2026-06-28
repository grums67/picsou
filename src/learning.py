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
                 min_trades: int = 50,
                 min_days: int = 14,
                 elimination_win_rate: float = 0.50,
                 elimination_max_drawdown: float = 0.30) -> None:
        self.data_path = data_path
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.file_path = data_path / "learning.json"
        self.win_rate_threshold = win_rate_threshold
        self.min_trades = min_trades
        self.min_days = min_days
        self.elimination_win_rate = elimination_win_rate
        self.elimination_max_drawdown = elimination_max_drawdown

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

        Args:
            strategy_name: LLM-assigned strategy_type category.
            trades: List of trade dicts with 'pnl' key.
        """
        if strategy_name not in self.scores:
            self.scores[strategy_name] = StrategyScore(strategy_name)

        score = self.scores[strategy_name]
        returns: List[float] = []

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

        # Calculate max drawdown from returns
        score.returns.extend(returns)
        score.max_drawdown = self._calculate_max_drawdown(score.returns)

        # Calculate Sharpe ratio
        score.sharpe_ratio = self._calculate_sharpe_ratio(score.returns)

        logger.info("Updated %s: win_rate=%.2f, trades=%d, sharpe=%.2f, "
                     "max_dd=%.2f",
                     strategy_name, score.win_rate, score.total_trades,
                     score.sharpe_ratio, score.max_drawdown)

    @staticmethod
    def _calculate_max_drawdown(returns: List[float]) -> float:
        """Calculate maximum drawdown from a list of returns.

        Returns the maximum drawdown as a positive decimal (e.g. 0.20 = 20%).
        """
        if not returns:
            return 0.0

        # Convert PnL returns to cumulative equity curve
        cumulative = [1.0]  # Start with 1.0
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
        - Reinforces winners by increasing weight
        - Returns evaluation summary.
        """
        self.evaluation_count += 1
        self.last_evaluation = datetime.now(timezone.utc).isoformat()

        eliminated: List[str] = []
        promoted: List[str] = []

        for name, score in self.scores.items():
            if not score.active:
                continue

            # Need minimum trades to evaluate
            if score.total_trades < self.min_trades:
                logger.info("%s: insufficient trades (%d < %d), skipping evaluation",
                            name, score.total_trades, self.min_trades)
                continue

            # Elimination criteria
            if score.win_rate < self.elimination_win_rate:
                score.active = False
                score.weight = 0.0
                eliminated.append(name)
                logger.info("%s: ELIMINATED (win_rate=%.2f < %.2f)",
                            name, score.win_rate, self.elimination_win_rate)
                continue

            if score.max_drawdown > self.elimination_max_drawdown:
                score.active = False
                score.weight = 0.0
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

        # Normalize weights so they sum to 1.0 among active categories
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