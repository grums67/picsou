"""Picsou v4 — Backtest engine (immutable, part of the socle).

The LLM can write strategy code and request a backtest.
This engine replays historical candles through a strategy's signal()
and returns performance metrics. The LLM CANNOT modify this code.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BacktestResult:
    """Result of a backtest run."""
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        self.avg_pnl = 0.0
        self.sharpe_ratio = 0.0
        self.trades: List[Dict] = []
        self.passed = False

    def to_dict(self) -> Dict:
        return {
            "strategy_name": self.strategy_name,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": round(self.total_pnl, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 4),
            "avg_pnl": round(self.avg_pnl, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "passed": self.passed,
            "trades": self.trades,
        }


class Backtester:
    """Simple backtest engine that replays candles through a strategy."""

    def __init__(self, starting_capital: float = 10000.0, fee_rate: float = 0.001):
        self.starting_capital = starting_capital
        self.fee_rate = fee_rate

    def run(self, strategy_module, candles: List[Dict],
            symbol: str = "BTC-USDT") -> BacktestResult:
        """Run a backtest on historical candle data.

        Args:
            strategy_module: A loaded strategy module with signal() and metadata()
            candles: List of candle dicts with keys: open, high, low, close, volume, timestamp
            symbol: Symbol for context

        Returns:
            BacktestResult with performance metrics
        """
        result = BacktestResult(
            strategy_name=strategy_module.metadata().get("name", "unknown")
        )

        if len(candles) < 20:
            logger.warning("Not enough candles for backtest: %d", len(candles))
            return result

        balance = self.starting_capital
        peak = self.starting_capital
        max_dd = 0.0
        position = None  # {amount, entry_price, entry_idx}

        # Slide a window of candles for context
        window_size = 20

        for i in range(window_size, len(candles)):
            # Build market_data for this point in time
            historical_candles = candles[:i]
            current_price = candles[i]["close"]

            market_data = {
                symbol: {
                    "candles": historical_candles[-window_size:],
                    "ticker": {"last": current_price, "volume_24h": candles[i].get("volume", 0)},
                    "current_price": current_price,
                }
            }

            portfolio_state = {
                "balance": balance,
                "position": position,
                "pnl_pct": ((balance - self.starting_capital) / self.starting_capital) * 100,
            }

            # Get signal from strategy
            try:
                signal = strategy_module.signal(market_data, portfolio_state, {})
            except Exception as e:
                logger.warning("Strategy signal() error at candle %d: %s", i, e)
                continue

            if not isinstance(signal, dict) or "action" not in signal:
                continue

            action = signal["action"].lower()
            confidence = signal.get("confidence", 0.5)
            size_pct = signal.get("size_pct", 0.05)

            # Execute signal
            if action == "buy" and position is None:
                # Open position
                amount_usd = min(balance * size_pct, balance * 0.2)  # Cap at 20%
                amount = amount_usd / current_price
                fee = amount_usd * self.fee_rate
                cost = amount_usd + fee

                if cost <= balance:
                    balance -= cost
                    position = {
                        "amount": amount,
                        "entry_price": current_price,
                        "entry_idx": i,
                        "fee": fee,
                    }

            elif action == "sell" and position is not None:
                # Close position
                close_value = position["amount"] * current_price
                close_fee = close_value * self.fee_rate
                pnl = close_value - (position["amount"] * position["entry_price"]) - position["fee"] - close_fee
                balance += close_value - close_fee

                result.trades.append({
                    "entry_idx": position["entry_idx"],
                    "exit_idx": i,
                    "pnl": round(pnl, 4),
                    "pnl_pct": round(pnl / (position["amount"] * position["entry_price"]) * 100, 2),
                    "entry_price": position["entry_price"],
                    "exit_price": current_price,
                })

                result.total_trades += 1
                if pnl > 0:
                    result.winning_trades += 1
                else:
                    result.losing_trades += 1
                result.total_pnl += pnl

                position = None

            # Track drawdown
            equity = balance
            if position:
                equity += position["amount"] * current_price
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # If still in position at end, close it at last price
        if position:
            last_price = candles[-1]["close"]
            close_value = position["amount"] * last_price
            close_fee = close_value * self.fee_rate
            pnl = close_value - (position["amount"] * position["entry_price"]) - position["fee"] - close_fee
            balance += close_value - close_fee

            result.total_trades += 1
            if pnl > 0:
                result.winning_trades += 1
            result.total_pnl += pnl

        # Calculate final metrics
        result.max_drawdown = max_dd
        result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0
        result.avg_pnl = result.total_pnl / result.total_trades if result.total_trades > 0 else 0
        result.sharpe_ratio = self._calc_sharpe(result.trades)

        # Pass threshold: at least 5 trades, positive avg PnL, win rate > 30%
        result.passed = (
            result.total_trades >= 5
            and result.avg_pnl > 0
            and result.win_rate > 0.30
        )

        return result

    def _calc_sharpe(self, trades: List[Dict]) -> float:
        """Calculate Sharpe ratio from trade PnLs."""
        if len(trades) < 2:
            return 0.0

        pnls = [t["pnl"] for t in trades]
        avg = sum(pnls) / len(pnls)
        variance = sum((p - avg) ** 2 for p in pnls) / len(pnls)
        std = variance ** 0.5 if variance > 0 else 0.001
        return avg / std if std > 0 else 0.0