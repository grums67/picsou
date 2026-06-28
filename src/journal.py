"""Decision journal for Picsou trading agent.

Records and queries all trading decisions, including LLM prompts,
responses, and token usage for analysis and debugging.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DecisionJournal:
    """Records and queries all trading decisions.

    Uses JSON Lines format for append-only, resilient storage.
    """

    def __init__(self, data_path: Path = Path("/root/PROJECTS/picsou/data")) -> None:
        self.data_path = data_path
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.file_path = data_path / "journal.jsonl"

    def log_decision(
        self,
        exchange: str,
        symbol: str,
        strategy: str,
        action: str,
        reasoning: str,
        confidence: float,
        outcome: Optional[str] = None,
        amount: Optional[float] = None,
        price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        llm_prompt: Optional[str] = None,
        llm_response: Optional[str] = None,
        llm_tokens: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """Record a trading decision.

        Args:
            exchange: Exchange name.
            symbol: Trading pair.
            strategy: Strategy type (LLM-assigned category).
            action: "buy", "sell", or "hold".
            reasoning: Human-readable explanation from the LLM.
            confidence: 0.0 to 1.0 confidence score.
            outcome: Optional result tracking.
            amount: Trade amount if applicable.
            price: Trade price if applicable.
            metadata: Additional data.
            llm_prompt: The full prompt sent to the LLM.
            llm_response: The raw LLM response.
            llm_tokens: Token usage dict (prompt_tokens, completion_tokens, total_tokens).

        Returns:
            The logged decision dict.
        """
        decision = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "exchange": exchange,
            "symbol": symbol,
            "strategy": strategy,
            "action": action,
            "reasoning": reasoning,
            "confidence": round(confidence, 4),
            "outcome": outcome,
            "amount": amount,
            "price": price,
            "metadata": metadata or {},
            # LLM-specific fields
            "llm_prompt": llm_prompt,
            "llm_response": llm_response,
            "llm_tokens": llm_tokens,
        }
        try:
            with open(self.file_path, "a") as f:
                f.write(json.dumps(decision) + "\n")
            logger.debug("Logged decision: %s %s %s", action, symbol, strategy)
        except Exception as e:
            logger.error("Failed to log decision: %s", e)
        return decision

    def get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get the most recent decisions.

        Args:
            limit: Maximum number of decisions to return.

        Returns:
            List of decision dicts, newest first.
        """
        decisions: List[Dict[str, Any]] = []
        if not self.file_path.exists():
            return decisions
        try:
            with open(self.file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        decisions.append(json.loads(line))
        except Exception as e:
            logger.error("Failed to read journal: %s", e)
        return decisions[-limit:][::-1]  # Newest first

    def get_stats(self) -> Dict[str, Any]:
        """Compute statistics over all decisions.

        Returns dict with: total_decisions, actions_breakdown,
        avg_confidence, strategy_breakdown, exchange_breakdown.
        """
        if not self.file_path.exists():
            return {"total_decisions": 0}

        decisions: List[Dict[str, Any]] = []
        try:
            with open(self.file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        decisions.append(json.loads(line))
        except Exception as e:
            logger.error("Failed to read journal for stats: %s", e)
            return {"total_decisions": 0}

        total = len(decisions)
        actions: Dict[str, int] = {}
        strategies: Dict[str, int] = {}
        exchanges: Dict[str, int] = {}
        confidences: List[float] = []

        for d in decisions:
            action = d.get("action", "unknown")
            actions[action] = actions.get(action, 0) + 1

            strategy = d.get("strategy", "unknown")
            strategies[strategy] = strategies.get(strategy, 0) + 1

            exchange = d.get("exchange", "unknown")
            exchanges[exchange] = exchanges.get(exchange, 0) + 1

            conf = d.get("confidence", 0)
            if conf is not None:
                confidences.append(conf)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "total_decisions": total,
            "actions_breakdown": actions,
            "avg_confidence": round(avg_confidence, 4),
            "strategy_breakdown": strategies,
            "exchange_breakdown": exchanges,
        }

    def get_by_strategy(self, strategy: str,
                        limit: int = 50) -> List[Dict[str, Any]]:
        """Get decisions for a specific strategy type."""
        all_decisions = self.get_recent(limit=10000)
        return [d for d in all_decisions if d.get("strategy") == strategy][:limit]

    def get_by_symbol(self, symbol: str,
                      limit: int = 50) -> List[Dict[str, Any]]:
        """Get decisions for a specific symbol."""
        all_decisions = self.get_recent(limit=10000)
        return [d for d in all_decisions if d.get("symbol") == symbol][:limit]

    def get_llm_usage_stats(self) -> Dict[str, Any]:
        """Get aggregated LLM token usage statistics."""
        if not self.file_path.exists():
            return {"total_prompt_tokens": 0, "total_completion_tokens": 0,
                    "total_tokens": 0, "calls_with_llm": 0}

        total_prompt = 0
        total_completion = 0
        total_tokens = 0
        calls_with_llm = 0

        try:
            with open(self.file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    tokens = d.get("llm_tokens")
                    if tokens and isinstance(tokens, dict):
                        total_prompt += tokens.get("prompt_tokens", 0)
                        total_completion += tokens.get("completion_tokens", 0)
                        total_tokens += tokens.get("total_tokens", 0)
                        calls_with_llm += 1
        except Exception as e:
            logger.error("Failed to compute LLM usage stats: %s", e)

        return {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "calls_with_llm": calls_with_llm,
        }