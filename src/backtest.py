"""Backtesting module for Picsou trading agent.

Reads historical trade data from journal.jsonl and simulates past trades
per strategy to compute performance metrics: win_rate, avg_profit,
max_drawdown, sharpe_ratio.

Used by the learning engine to validate strategies before (re)activation.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BacktestResult:
    """Result of a backtest run for a single strategy."""

    def __init__(
        self,
        strategy_name: str,
        win_rate: float = 0.0,
        avg_profit: float = 0.0,
        max_drawdown: float = 0.0,
        sharpe_ratio: float = 0.0,
        total_trades: int = 0,
        winning_trades: int = 0,
        losing_trades: int = 0,
        total_profit: float = 0.0,
        passes: bool = False,
    ) -> None:
        self.strategy_name = strategy_name
        self.win_rate = win_rate
        self.avg_profit = avg_profit
        self.max_drawdown = max_drawdown
        self.sharpe_ratio = sharpe_ratio
        self.total_trades = total_trades
        self.winning_trades = winning_trades
        self.losing_trades = losing_trades
        self.total_profit = total_profit
        self.passes = passes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "pass": self.passes,
            "win_rate": round(self.win_rate, 4),
            "avg_profit": round(self.avg_profit, 6),
            "max_drawdown": round(self.max_drawdown, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "trades_simulated": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_profit": round(self.total_profit, 4),
        }


class Backtester:
    """Backtests strategies against historical journal data.

    Reads journal.jsonl entries, groups them by strategy, pairs buy/sell
    trades, and computes performance metrics.
    """

    def __init__(
        self,
        data_path: Path = Path("/root/PROJECTS/picsou/data"),
        win_rate_threshold: float = 0.30,
        avg_profit_threshold: float = 0.0,
    ) -> None:
        self.data_path = data_path
        self.journal_path = data_path / "journal.jsonl"
        self.win_rate_threshold = win_rate_threshold
        self.avg_profit_threshold = avg_profit_threshold

    def _load_journal(self, limit: int = 0) -> List[Dict[str, Any]]:
        """Load journal entries from journal.jsonl.

        Args:
            limit: Maximum number of entries to return. 0 = all.
        """
        if not self.journal_path.exists():
            logger.warning("Journal file not found: %s", self.journal_path)
            return []

        entries = []
        try:
            with open(self.journal_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error("Failed to read journal: %s", e)
            return []

        if limit > 0:
            return entries[-limit:]
        return entries

    def _pair_trades_by_strategy(
        self, entries: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Pair buy/sell entries to compute realized PnL per strategy.

        For each strategy, we track open positions and match sells to buys.
        Returns a dict of strategy_name -> list of trade dicts with pnl.

        Trade dict format:
        {
            "timestamp": str,
            "symbol": str,
            "action": "buy" | "sell",
            "entry_price": float,
            "close_price": float,
            "pnl": float,  # realized PnL for completed trades
            "strategy": str,
        }
        """
        # Group entries by strategy
        strategy_entries: Dict[str, List[Dict[str, Any]]] = {}
        for entry in entries:
            strategy = entry.get("strategy", "unknown")
            action = entry.get("action", "")
            if action not in ("buy", "sell"):
                continue
            if strategy not in strategy_entries:
                strategy_entries[strategy] = []
            strategy_entries[strategy].append(entry)

        # For each strategy, pair buy/sell trades
        strategy_trades: Dict[str, List[Dict[str, Any]]] = {}

        for strategy, strat_entries in strategy_entries.items():
            # Sort by timestamp
            strat_entries.sort(key=lambda e: e.get("timestamp", ""))

            # Track open positions: list of (price, amount)
            open_positions: List[Dict[str, Any]] = []
            completed_trades: List[Dict[str, Any]] = []

            for entry in strat_entries:
                action = entry.get("action", "")
                price = entry.get("price")
                amount = entry.get("amount")

                # Skip entries without price
                if price is None:
                    continue

                try:
                    price = float(price)
                except (TypeError, ValueError):
                    continue

                if action == "buy":
                    open_positions.append({
                        "price": price,
                        "amount": float(amount) if amount else 0,
                        "timestamp": entry.get("timestamp", ""),
                        "symbol": entry.get("symbol", ""),
                    })
                elif action == "sell" and open_positions:
                    # Match with oldest open position (FIFO)
                    pos = open_positions.pop(0)
                    entry_price = pos["price"]
                    close_price = price
                    pnl = close_price - entry_price
                    # Scale by amount if available
                    if pos.get("amount") and pos["amount"] > 0:
                        pnl = (close_price - entry_price) * pos["amount"]

                    completed_trades.append({
                        "timestamp": entry.get("timestamp", ""),
                        "symbol": entry.get("symbol", pos.get("symbol", "")),
                        "action": "sell",
                        "entry_price": entry_price,
                        "close_price": close_price,
                        "pnl": pnl,
                        "strategy": strategy,
                    })

            # Also include unmatched buys as open positions (pnl = 0, they're unrealized)
            for pos in open_positions:
                completed_trades.append({
                    "timestamp": pos.get("timestamp", ""),
                    "symbol": pos.get("symbol", ""),
                    "action": "buy",
                    "entry_price": pos["price"],
                    "close_price": 0,
                    "pnl": 0,  # Unrealized, no PnL yet
                    "strategy": strategy,
                })

            strategy_trades[strategy] = completed_trades

        return strategy_trades

    def backtest_strategy(self, strategy_name: str) -> BacktestResult:
        """Run a backtest for a specific strategy on historical data.

        Reads journal.jsonl, filters trades for the given strategy,
        pairs buy/sell, and computes performance metrics.

        Args:
            strategy_name: The strategy to backtest.

        Returns:
            BacktestResult with metrics and pass/fail status.
        """
        entries = self._load_journal()

        if not entries:
            logger.info("No journal entries for backtest of %s", strategy_name)
            return BacktestResult(
                strategy_name=strategy_name,
                passes=False,
            )

        # Get all paired trades
        all_trades = self._pair_trades_by_strategy(entries)
        strategy_trades = all_trades.get(strategy_name, [])

        if not strategy_trades:
            logger.info("No trades found for strategy %s in backtest", strategy_name)
            return BacktestResult(
                strategy_name=strategy_name,
                passes=False,
            )

        # Only count completed trades (those with a close_price > 0)
        completed = [t for t in strategy_trades if t.get("close_price", 0) > 0]

        if not completed:
            logger.info("No completed trades for strategy %s in backtest", strategy_name)
            return BacktestResult(
                strategy_name=strategy_name,
                total_trades=len(strategy_trades),
                passes=False,
            )

        # Compute metrics
        pnls = [t["pnl"] for t in completed]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p <= 0]

        total_trades = len(completed)
        win_rate = len(winning) / total_trades if total_trades > 0 else 0.0
        avg_profit = sum(pnls) / total_trades if total_trades > 0 else 0.0
        total_profit = sum(pnls)

        # Max drawdown from cumulative PnL
        max_drawdown = self._calculate_max_drawdown(pnls)

        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio(pnls)

        # Determine pass/fail
        passes = win_rate > self.win_rate_threshold and avg_profit > self.avg_profit_threshold

        result = BacktestResult(
            strategy_name=strategy_name,
            win_rate=win_rate,
            avg_profit=avg_profit,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            total_trades=total_trades,
            winning_trades=len(winning),
            losing_trades=len(losing),
            total_profit=total_profit,
            passes=passes,
        )

        logger.info(
            "Backtest %s: win_rate=%.2f, avg_profit=%.6f, trades=%d, pass=%s",
            strategy_name, win_rate, avg_profit, total_trades, passes,
        )

        return result

    def backtest_all_strategies(self) -> Dict[str, BacktestResult]:
        """Run backtest for all strategies found in the journal.

        Returns:
            Dict mapping strategy name to BacktestResult.
        """
        entries = self._load_journal()
        if not entries:
            return {}

        all_trades = self._pair_trades_by_strategy(entries)
        results = {}
        for strategy_name in all_trades:
            results[strategy_name] = self.backtest_strategy(strategy_name)

        return results

    @staticmethod
    def _calculate_max_drawdown(pnls: List[float]) -> float:
        """Calculate maximum drawdown from a list of trade PnLs.

        Returns the maximum drawdown as a positive decimal (e.g. 0.20 = 20%).
        """
        if not pnls:
            return 0.0

        # Build cumulative equity curve starting from 1.0
        cumulative = [1.0]
        for pnl in pnls:
            cumulative.append(cumulative[-1] + pnl)

        peak = cumulative[0]
        max_dd = 0.0

        for value in cumulative:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, drawdown)

        return max_dd

    @staticmethod
    def _calculate_sharpe_ratio(pnls: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio from trade PnLs."""
        if len(pnls) < 2:
            return 0.0

        excess_returns = [p - risk_free_rate for p in pnls]
        mean_ret = sum(excess_returns) / len(excess_returns)
        variance = sum((r - mean_ret) ** 2 for r in excess_returns) / len(excess_returns)
        std_ret = math.sqrt(variance) if variance > 0 else 0.0

        if std_ret == 0:
            return 0.0

        # Annualize (crypto = 365 days/year)
        sharpe = (mean_ret / std_ret) * math.sqrt(365)
        return sharpe