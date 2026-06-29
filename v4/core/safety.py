"""Picsou v4 — Safety module. Hard limits that the agent CANNOT override.

These are circuit breakers. They run BEFORE any trade is executed.
No LLM, no strategy, no parameter change can bypass these.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SafetyCheck:
    """Result of a safety check."""
    allowed: bool
    reason: str = ""


class Safety:
    """Hard safety limits enforced on every trade."""

    def __init__(self, config):
        self.max_position_pct = config.safety.max_position_pct
        self.max_open_positions = config.safety.max_open_positions
        self.max_drawdown_pct = config.safety.max_drawdown_pct
        self.min_position_usd = config.safety.min_position_usd
        self.max_strategies_active = config.safety.max_strategies_active

    def check_trade(self, side: str, amount_usd: float, balance: float,
                    open_positions: int, starting_capital: float,
                    current_pnl_pct: float) -> SafetyCheck:
        """Check if a trade is allowed by hard safety rules.

        Returns SafetyCheck(allowed=True) if the trade can proceed,
        or SafetyCheck(allowed=False, reason="...") if blocked.
        """
        # Circuit breaker: max drawdown
        drawdown = max(0, -current_pnl_pct)
        if drawdown >= self.max_drawdown_pct:
            logger.warning("SAFETY: Circuit breaker triggered — drawdown %.1f%% >= %.1f%%",
                           drawdown * 100, self.max_drawdown_pct * 100)
            return SafetyCheck(False, f"Drawdown {drawdown*100:.1f}% exceeds limit {self.max_drawdown_pct*100:.1f}%")

        # Max open positions
        if side.lower() == "buy" and open_positions >= self.max_open_positions:
            return SafetyCheck(False, f"Max open positions reached ({open_positions}/{self.max_open_positions})")

        # Min position size
        if amount_usd < self.min_position_usd:
            return SafetyCheck(False, f"Position size ${amount_usd:.2f} below minimum ${self.min_position_usd}")

        # Max position size (% of capital)
        max_usd = balance * self.max_position_pct
        if amount_usd > max_usd:
            logger.warning("SAFETY: Reducing position from $%.2f to $%.2f (max %.0f%% of balance)",
                            amount_usd, max_usd, self.max_position_pct * 100)
            # Don't block — just cap it. The caller should adjust.
            # Still allow the trade but flag it.
            pass  # We'll cap in the executor

        return SafetyCheck(True)

    def check_strategy_count(self, active_count: int) -> SafetyCheck:
        """Check if we can activate another strategy."""
        if active_count >= self.max_strategies_active:
            return SafetyCheck(False, f"Max {self.max_strategies_active} active strategies reached")
        return SafetyCheck(True)

    def cap_position_size(self, amount_usd: float, balance: float) -> float:
        """Cap position size to max allowed % of balance."""
        max_usd = balance * self.max_position_pct
        if amount_usd > max_usd:
            logger.info("Capping position: $%.2f → $%.2f", amount_usd, max_usd)
            return max_usd
        return amount_usd