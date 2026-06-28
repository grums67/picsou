"""SelfReflect - Self-reflection and self-improvement loop for Picsou trading agent.

Periodically analyzes trading performance using the LLM to generate
actionable strategy rules, which are then injected into future prompts
and used to adjust position sizing and strategy selection.
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Weight bounds for strategy adjustments
MAX_WEIGHT_BOOST = 0.20
MAX_WEIGHT_PENALTY = 0.30


class SelfReflect:
    """Analyzes past performance and generates strategy improvement rules.

    Reads recent trades, learning scores, and portfolio PnL, then calls
    the LLM to produce actionable rules saved to strategy_rules.json.
    These rules are hot-reloaded by brain.py and picsou.py each cycle.
    """

    def __init__(
        self,
        data_path: Path = Path("/root/PROJECTS/picsou/data"),
        llm_url: str = "http://127.0.0.1:11434/v1",
        llm_api_key: str = "",
        llm_model: str = "kimi-k2.6:cloud",
        llm_config_path: str = "",
    ) -> None:
        self.data_path = Path(data_path)
        self.llm_url = llm_url.rstrip("/")
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.llm_config_path = llm_config_path
        self.rules_path = self.data_path / "strategy_rules.json"
        self.current_rules: Dict[str, Any] = {}
        self._load_rules_from_disk()

    # ── Disk I/O ────────────────────────────────────────────────────────

    def _load_rules_from_disk(self) -> None:
        """Load existing rules from strategy_rules.json if it exists."""
        if self.rules_path.exists():
            try:
                with open(self.rules_path, "r", encoding="utf-8") as f:
                    self.current_rules = json.load(f)
                logger.info("Loaded %d strategy rules from %s",
                            len(self.current_rules.get("rules", [])),
                            self.rules_path)
            except Exception as e:
                logger.warning("Failed to load strategy rules: %s", e)
                self.current_rules = {}
        else:
            self.current_rules = {}

    def load_rules(self) -> Dict[str, Any]:
        """Re-read strategy_rules.json from disk (hot-reload).

        Called at the start of each cycle so the running process picks up
        changes made by a reflect() call.
        """
        self._load_rules_from_disk()
        return self.current_rules

    # ── Data gathering ──────────────────────────────────────────────────

    def _read_recent_trades(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Read the last N entries from journal.jsonl."""
        journal_path = self.data_path / "journal.jsonl"
        if not journal_path.exists():
            return []
        try:
            entries = []
            with open(journal_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            # Return last N entries
            return entries[-limit:]
        except Exception as e:
            logger.warning("Failed to read journal: %s", e)
            return []

    def _read_learning_scores(self) -> Dict[str, Any]:
        """Read learning.json for strategy performance scores."""
        learning_path = self.data_path / "learning.json"
        if not learning_path.exists():
            return {}
        try:
            with open(learning_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("scores", {})
        except Exception as e:
            logger.warning("Failed to read learning scores: %s", e)
            return {}

    def _read_portfolio_pnl(self) -> Dict[str, Any]:
        """Read portfolio.json for current PnL info."""
        portfolio_path = self.data_path / "portfolio.json"
        if not portfolio_path.exists():
            return {}
        try:
            with open(portfolio_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            balance = data.get("balance", 0)
            starting = data.get("starting_capital", 0)
            trades = data.get("trades", [])
            positions = data.get("positions", {})

            # Calculate PnL
            total_pnl = sum(t.get("pnl", 0) for t in trades) if trades else 0
            num_trades = len(trades)
            winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
            losing_trades = [t for t in trades if t.get("pnl", 0) < 0]
            win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0

            return {
                "balance": round(balance, 2),
                "starting_capital": starting,
                "total_pnl": round(total_pnl, 2),
                "return_pct": round((balance - starting) / starting * 100, 2) if starting > 0 else 0,
                "total_trades": num_trades,
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": round(win_rate, 4),
                "open_positions": len(positions),
            }
        except Exception as e:
            logger.warning("Failed to read portfolio: %s", e)
            return {}

    def _read_llm_config(self) -> Dict[str, Any]:
        """Read llm_config.json for current model settings."""
        config_path = Path(self.llm_config_path) if self.llm_config_path else self.data_path / "llm_config.json"
        if not config_path.exists():
            return {}
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to read LLM config: %s", e)
            return {}

    # ── LLM call ────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM with a reflection prompt and return the response text.

        Uses the same OpenAI-compatible API as brain.py.
        """
        # Hot-reload config to pick up model changes
        config = self._read_llm_config()
        model = config.get("llm_model", self.llm_model)
        url = config.get("llm_url", self.llm_url).rstrip("/")
        # Temperature for reflection can be slightly higher for creativity
        temperature = 0.6
        max_tokens = config.get("llm_max_tokens", 4096)

        api_url = f"{url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_api_key}",
        }
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a trading strategy analyst AI. You analyze past trading "
                        "performance data and produce structured, actionable rules to improve "
                        "future trading decisions. You always respond in valid JSON format. "
                        "Do NOT wrap your response in markdown code fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()

            # Extract response content (support reasoning models)
            msg = data.get("choices", [{}])[0].get("message", {})
            content = msg.get("content", "") or msg.get("reasoning", "") or ""

            if content:
                logger.info("Self-reflection LLM call succeeded (%d chars)", len(content))
            return content

        except requests.exceptions.Timeout:
            logger.error("Self-reflection LLM call timed out")
            return None
        except requests.exceptions.ConnectionError:
            logger.error("Self-reflection LLM connection error")
            return None
        except Exception as e:
            logger.error("Self-reflection LLM call failed: %s", e)
            return None

    # ── Prompt building ─────────────────────────────────────────────────

    def _build_reflection_prompt(
        self,
        trades: List[Dict[str, Any]],
        scores: Dict[str, Any],
        pnl_info: Dict[str, Any],
        current_rules: Dict[str, Any],
    ) -> str:
        """Build the reflection prompt for the LLM."""
        # Summarize recent trades
        trade_summary_lines = []
        for t in trades[-30:]:  # Last 30 for prompt brevity
            action = t.get("action", "?")
            symbol = t.get("symbol", "?")
            strategy = t.get("strategy", "?")
            confidence = t.get("confidence", 0)
            reasoning = (t.get("reasoning", "") or "")[:100]
            trade_summary_lines.append(
                f"  - {action} {symbol} [{strategy}] conf={confidence:.2f}: {reasoning}"
            )
        trade_summary = "\n".join(trade_summary_lines) if trade_summary_lines else "  (no recent trades)"

        # Summarize strategy scores
        score_lines = []
        for name, info in scores.items():
            if isinstance(info, dict):
                win_rate = info.get("win_rate", 0)
                total = info.get("total_trades", 0)
                weight = info.get("weight", 0)
                active = info.get("active", False)
                avg_profit = info.get("avg_profit", 0)
                status = "ACTIVE" if active else "ELIMINATED"
                score_lines.append(
                    f"  - {name}: win_rate={win_rate:.0%}, trades={total}, "
                    f"weight={weight:.2f}, avg_profit={avg_profit:.4f} [{status}]"
                )
        score_summary = "\n".join(score_lines) if score_lines else "  (no strategy scores)"

        # Summarize PnL
        pnl_summary = (
            f"  Balance: {pnl_info.get('balance', 'N/A')} EUR\n"
            f"  Total PnL: {pnl_info.get('total_pnl', 'N/A')} EUR "
            f"({pnl_info.get('return_pct', 'N/A')}%)\n"
            f"  Win rate: {pnl_info.get('win_rate', 0):.0%} "
            f"({pnl_info.get('winning_trades', 0)}W / {pnl_info.get('losing_trades', 0)}L)\n"
            f"  Open positions: {pnl_info.get('open_positions', 0)}"
        ) if pnl_info else "  (no PnL data)"

        # Current rules
        current_rules_str = ""
        if current_rules.get("rules"):
            current_rules_str = "Current rules:\n" + "\n".join(
                f"  - {r}" for r in current_rules["rules"]
            )
        else:
            current_rules_str = "No current rules (first reflection)."

        return f"""Analyze the Picsou crypto trading agent's recent performance and generate actionable self-improvement rules.

RECENT TRADES (last 30 of {len(trades)} total):
{trade_summary}

STRATEGY PERFORMANCE SCORES:
{score_summary}

PORTFOLIO PnL:
{pnl_summary}

CURRENT SELF-IMPROVEMENT RULES:
{current_rules_str}

Based on this data, analyze:
1. Which strategies are working and WHY (look at win_rate, avg_profit)
2. Which strategies are failing and WHY
3. What market conditions (e.g., extreme Fear & Greed) lead to losses
4. Specific parameter changes that could help (position sizes, risk thresholds, strategy focus)
5. Whether to enable/disable certain strategies (set weight_penalty >= 0.5 to effectively disable)

Respond with ONLY a JSON object in this exact format:
{{
  "rules": [
    "Rule 1: specific actionable rule",
    "Rule 2: another rule",
    "Rule 3: another rule"
  ],
  "strategy_adjustments": {{
    "strategy_name": {{"weight_boost": 0.0 to 0.20, "weight_penalty": 0.0 to 0.30, "reason": "why"}},
    "another_strategy": {{"weight_boost": 0.0, "weight_penalty": 0.0 to 0.30, "reason": "why"}}
  }},
  "parameter_changes": {{
    "preferred_position_pct": 0.03 to 0.20,
    "avoid_high_fear": true or false,
    "avoid_extreme_greed": true or false
  }},
  "analysis": "Brief summary of key findings"
}}

IMPORTANT constraints:
- weight_boost max is 0.20, weight_penalty max is 0.30
- Only include strategies that actually have data (listed in scores above)
- Set weight_penalty >= 0.5 to effectively disable a strategy
- preferred_position_pct should be between 0.03 (3%) and 0.20 (20%)
- Be specific and data-driven, not generic
- Rules should be actionable, not vague"""

    # ── Response parsing ────────────────────────────────────────────────

    def _parse_llm_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse the LLM's JSON response into structured rules."""
        if not response_text:
            return None

        # Strip markdown code fences if present
        raw = response_text.strip()
        if raw.startswith("```"):
            first_newline = raw.find("\n")
            if first_newline != -1:
                raw = raw[first_newline + 1:]
            if raw.endswith("```"):
                raw = raw[:-3].rstrip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse self-reflection LLM response: %s", e)
                    return None
            else:
                logger.error("No JSON found in self-reflection LLM response")
                return None

        # Validate and sanitize the parsed response
        rules = parsed.get("rules", [])
        if not isinstance(rules, list):
            rules = []
        # Ensure all rules are strings
        rules = [str(r) for r in rules if r]

        strategy_adjustments = parsed.get("strategy_adjustments", {})
        if not isinstance(strategy_adjustments, dict):
            strategy_adjustments = {}

        # Clamp weight values to bounds
        for name, adj in strategy_adjustments.items():
            if isinstance(adj, dict):
                adj["weight_boost"] = max(0.0, min(MAX_WEIGHT_BOOST, float(adj.get("weight_boost", 0))))
                adj["weight_penalty"] = max(0.0, min(MAX_WEIGHT_PENALTY, float(adj.get("weight_penalty", 0))))
                adj["reason"] = str(adj.get("reason", ""))

        parameter_changes = parsed.get("parameter_changes", {})
        if not isinstance(parameter_changes, dict):
            parameter_changes = {}

        # Validate parameter_changes
        if "preferred_position_pct" in parameter_changes:
            parameter_changes["preferred_position_pct"] = max(
                0.03, min(0.20, float(parameter_changes["preferred_position_pct"]))
            )
        if "avoid_high_fear" in parameter_changes:
            parameter_changes["avoid_high_fear"] = bool(parameter_changes["avoid_high_fear"])
        if "avoid_extreme_greed" in parameter_changes:
            parameter_changes["avoid_extreme_greed"] = bool(parameter_changes["avoid_extreme_greed"])

        analysis = parsed.get("analysis", "")
        if not isinstance(analysis, str):
            analysis = str(analysis)

        return {
            "rules": rules,
            "strategy_adjustments": strategy_adjustments,
            "parameter_changes": parameter_changes,
            "analysis": analysis,
        }

    # ── Main reflect method ─────────────────────────────────────────────

    def reflect(self) -> Optional[Dict[str, Any]]:
        """Run a self-reflection cycle.

        Reads performance data, calls the LLM for analysis, parses the
        response into actionable rules, and saves them to strategy_rules.json.

        Returns:
            A summary dict of what changed, or None on failure.
        """
        logger.info("Starting self-reflection cycle...")

        # 1. Gather data
        trades = self._read_recent_trades(limit=200)
        scores = self._read_learning_scores()
        pnl_info = self._read_portfolio_pnl()

        if not trades and not scores:
            logger.warning("No trades or scores data available for self-reflection")
            return None

        # 2. Build prompt
        prompt = self._build_reflection_prompt(trades, scores, pnl_info, self.current_rules)

        # 3. Call LLM
        response_text = self._call_llm(prompt)
        if not response_text:
            logger.error("Self-reflection: LLM returned no response")
            return None

        # 4. Parse response
        parsed = self._parse_llm_response(response_text)
        if not parsed:
            logger.error("Self-reflection: failed to parse LLM response")
            return None

        # 5. Build the rules dict
        now = datetime.now(timezone.utc).isoformat()
        prev_count = self.current_rules.get("reflection_count", 0)

        new_rules = {
            "last_reflection": now,
            "reflection_count": prev_count + 1,
            "rules": parsed["rules"],
            "strategy_adjustments": parsed["strategy_adjustments"],
            "parameter_changes": parsed["parameter_changes"],
            "llm_analysis": parsed["analysis"],
        }

        # 6. Compute changes (for summary)
        changes = []
        old_rules = self.current_rules.get("rules", [])
        new_rule_set = set(parsed["rules"])
        old_rule_set = set(old_rules)
        added = new_rule_set - old_rule_set
        removed = old_rule_set - new_rule_set
        if added:
            changes.append(f"rules_added={len(added)}")
        if removed:
            changes.append(f"rules_removed={len(removed)}")

        old_adjustments = self.current_rules.get("strategy_adjustments", {})
        for name, adj in parsed["strategy_adjustments"].items():
            if name not in old_adjustments:
                changes.append(f"{name}_adjustment_new")
            elif old_adjustments[name] != adj:
                changes.append(f"{name}_adjustment_updated")

        old_params = self.current_rules.get("parameter_changes", {})
        for key, val in parsed["parameter_changes"].items():
            if key not in old_params or old_params[key] != val:
                changes.append(f"param_{key}_changed")

        # 7. Save to disk
        try:
            self.data_path.mkdir(parents=True, exist_ok=True)
            with open(self.rules_path, "w", encoding="utf-8") as f:
                json.dump(new_rules, f, indent=2)
            logger.info("Self-reflection rules saved to %s", self.rules_path)
        except Exception as e:
            logger.error("Failed to save strategy rules: %s", e)
            return None

        # 8. Update in-memory rules
        self.current_rules = new_rules

        # 9. Return summary
        summary = {
            "changes": changes,
            "rules_count": len(parsed["rules"]),
            "strategy_adjustments": list(parsed["strategy_adjustments"].keys()),
            "parameter_changes": list(parsed["parameter_changes"].keys()),
            "analysis_preview": parsed["analysis"][:200] if parsed["analysis"] else "",
        }
        logger.info("Self-reflection completed: %s changes, %d rules",
                     len(changes), len(parsed["rules"]))
        return summary