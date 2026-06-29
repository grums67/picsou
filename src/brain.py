"""PicsouBrain - LLM-driven decision engine for Picsou trading agent.

Uses Mistral AI API (OpenAI-compatible) to analyze market data, sentiment,
and portfolio state to produce structured trading decisions with reasoning.
Falls back to simple EMA crossover signals if LLM is unavailable.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.technical_analysis import generate_technical_summary

logger = logging.getLogger(__name__)


# ── Fallback technical indicators ──────────────────────────────────────────

def calculate_ema(prices: List[float], period: int) -> List[float]:
    """Calculate Exponential Moving Average."""
    if not prices or len(prices) < period:
        return [0.0] * len(prices)
    ema = [0.0] * len(prices)
    multiplier = 2.0 / (period + 1)
    sma = sum(prices[:period]) / period
    ema[period - 1] = sma
    for i in range(period, len(prices)):
        ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1]
    return ema


def fallback_ema_signal(candles: List[Dict[str, Any]],
                       symbol: str,
                       exchange: str) -> Optional[Dict[str, Any]]:
    """Simple EMA crossover fallback when LLM is unavailable.

    Returns a single decision dict or None.
    """
    if not candles or len(candles) < 30:
        return None

    closes = [c["close"] for c in candles]
    ema9 = calculate_ema(closes, 9)
    ema21 = calculate_ema(closes, 21)

    n = len(closes)
    curr_fast = ema9[n - 1]
    curr_slow = ema21[n - 1]
    prev_fast = ema9[n - 2]
    prev_slow = ema21[n - 2]

    if curr_fast == 0.0 or curr_slow == 0.0:
        return None

    curr_price = closes[-1]
    separation = abs(curr_fast - curr_slow) / curr_price if curr_price > 0 else 0

    # Bullish crossover
    if prev_fast <= prev_slow and curr_fast > curr_slow:
        return {
            "action": "buy",
            "symbol": symbol,
            "exchange": exchange,
            "amount": 0.0,  # Will be sized by make_decisions
            "confidence": min(0.4 + separation * 50, 0.85),
            "reasoning": f"FALLBACK EMA9/EMA21 bullish crossover (sep={separation:.4f})",
            "strategy_type": "momentum_fallback",
        }
    # Bearish crossover
    if prev_fast >= prev_slow and curr_fast < curr_slow:
        return {
            "action": "sell",
            "symbol": symbol,
            "exchange": exchange,
            "amount": 0.0,
            "confidence": min(0.4 + separation * 50, 0.85),
            "reasoning": f"FALLBACK EMA9/EMA21 bearish crossover (sep={separation:.4f})",
            "strategy_type": "momentum_fallback",
        }

    return None


# ── Sentiment fetchers ────────────────────────────────────────────────────

def fetch_fear_and_greed() -> Dict[str, Any]:
    """Fetch the Crypto Fear & Greed Index from alternative.me.

    Returns dict with 'value' (0-100), 'classification', or empty dict on error.
    """
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=3",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        entries = data.get("data", [])
        if entries:
            latest = entries[0]
            result = {
                "value": int(latest.get("value", 50)),
                "classification": latest.get("value_classification", "Neutral"),
            }
            # Include recent trend
            if len(entries) > 1:
                result["yesterday"] = int(entries[1].get("value", 50))
            logger.info("Fear & Greed: %d (%s)", result["value"], result["classification"])
            return result
    except Exception as e:
        logger.warning("Failed to fetch Fear & Greed Index: %s", e)
    return {}


def fetch_crypto_headlines() -> List[str]:
    """Fetch recent crypto news headlines.

    Tries CoinGecko status updates first, then CryptoCompare as fallback.
    Returns list of headline strings (up to 10).
    """
    # Source 1: CoinGecko trending (no API key needed)
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/search/trending",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        coins = data.get("coins", [])
        headlines = []
        for c in coins:
            item = c.get("item", {})
            name = item.get("name", "")
            symbol = item.get("symbol", "")
            market_cap = item.get("data", {}).get("market_cap", "N/A")
            price_btc = item.get("data", {}).get("price_btc", "N/A")
            if name:
                headlines.append(
                    f"Trending: {name} ({symbol}) - "
                    f"market_cap={market_cap}, price_btc={price_btc}"
                )
        if headlines:
            logger.info("Fetched %d trending items from CoinGecko", len(headlines))
            return headlines[:10]
    except Exception as e:
        logger.debug("CoinGecko trending fetch failed: %s", e)

    # Source 2: CryptoCompare (may need API key)
    try:
        resp = requests.get(
            "https://min-api.cryptocompare.com/data/v2/news/?lang=EN",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("Data", [])
        headlines = [a.get("title", "") for a in articles[:10] if a.get("title")]
        if headlines:
            logger.info("Fetched %d crypto headlines from CryptoCompare", len(headlines))
            return headlines
    except Exception as e:
        logger.debug("CryptoCompare news fetch failed: %s", e)

    return []


# ── PicsouBrain ───────────────────────────────────────────────────────────

class PicsouBrain:
    """LLM-powered brain that analyzes market + sentiment and produces decisions.

    Uses Mistral AI API (OpenAI-compatible chat completions endpoint).
    Falls back to EMA crossover signals if LLM is unavailable.
    Supports hot-reload of model config from llm_config.json each cycle.
    """

    # Available models (Ollama cloud)
    AVAILABLE_MODELS = [
        {"id": "kimi-k2.6:cloud", "name": "Kimi K2.6", "provider": "ollama"},
        {"id": "kimi-k2.5:cloud", "name": "Kimi K2.5", "provider": "ollama"},
        {"id": "deepseek-v4-flash:cloud", "name": "DeepSeek V4 Flash", "provider": "ollama"},
        {"id": "qwen3.5:cloud", "name": "Qwen 3.5", "provider": "ollama"},
        {"id": "minimax-m2.7:cloud", "name": "MiniMax M2.7", "provider": "ollama"},
        {"id": "glm-5.1:cloud", "name": "GLM 5.1", "provider": "ollama"},
    ]

    # Total tokens used since instantiation (for dashboard tracking)
    total_tokens_used: int = 0

    def __init__(self, llm_url: str = "http://127.0.0.1:11434/v1",
                 llm_api_key: str = "",
                 llm_model: str = "kimi-k2.6:cloud",
                 llm_temperature: float = 0.3,
                 llm_max_tokens: int = 4096,
                 fear_and_greed_enabled: bool = True,
                 news_enabled: bool = True,
                 config_path: str = "") -> None:
        self.llm_url = llm_url.rstrip("/")
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.fear_and_greed_enabled = fear_and_greed_enabled
        self.news_enabled = news_enabled
        self.config_path = config_path

        # Track last prompt/response for journaling
        self.last_prompt: Optional[str] = None
        self.last_response: Optional[str] = None
        self.last_tokens: Optional[Dict[str, int]] = None
        self.connected: bool = False  # LLM connection status
        self.total_tokens_used = 0

        # Exploration mode context (set by PicsouAgent before each cycle)
        self.exploration_mode: bool = False
        self.underexplored_strategies: List[str] = []
        self.learning_context: Optional[Dict[str, Any]] = None
        # Self-reflection strategy rules (hot-reloaded from strategy_rules.json)
        self.strategy_rules: Dict[str, Any] = {}

        # Trading knowledge base (loaded from YAML, injected into every prompt)
        self.trading_knowledge: str = ""
        self._load_trading_knowledge()

        # Cache for last computed technical indicators (reused by market awareness)
        self._last_tech_indicators: Dict[str, Any] = {}

        # Write initial config file if it doesn't exist
        self._write_default_config()

    # ── Hot-reload config management ──────────────────────────────────────

    def _load_trading_knowledge(self) -> None:
        """Load trading knowledge base from YAML file for prompt injection."""
        knowledge_path = Path(self.config_path).parent / "trading_knowledge.yaml" if self.config_path else Path("/root/PROJECTS/picsou/data/trading_knowledge.yaml")
        if not knowledge_path.exists():
            logger.warning("Trading knowledge file not found: %s", knowledge_path)
            return
        try:
            import yaml
            with open(knowledge_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            # Build a compact text representation for the prompt
            lines = ["TRADING KNOWLEDGE BASE:"]
            for section, content in data.items():
                if isinstance(content, dict):
                    lines.append(f"\n{section.upper().replace('_', ' ')}:")
                    for name, details in content.items():
                        if isinstance(details, dict):
                            parts = [f"  {name}:"]
                            for k, v in details.items():
                                parts.append(f"    {k}: {v}")
                            lines.append("\n".join(parts))
                        else:
                            lines.append(f"  {name}: {details}")
                else:
                    lines.append(f"{section}: {content}")
            self.trading_knowledge = "\n".join(lines)
            logger.info("Loaded trading knowledge base (%d chars)", len(self.trading_knowledge))
        except ImportError:
            # yaml not available, try reading as plain text
            try:
                self.trading_knowledge = knowledge_path.read_text(encoding="utf-8")
                logger.info("Loaded trading knowledge as raw text (%d chars)", len(self.trading_knowledge))
            except Exception as e:
                logger.warning("Failed to load trading knowledge: %s", e)
        except Exception as e:
            logger.warning("Failed to load trading knowledge YAML: %s", e)

    def _write_default_config(self) -> None:
        """Write default llm_config.json if it doesn't exist."""
        if not self.config_path:
            return
        config_file = Path(self.config_path)
        if not config_file.exists():
            config_file.parent.mkdir(parents=True, exist_ok=True)
            config_data = {
                "llm_model": self.llm_model,
                "llm_url": self.llm_url,
                "llm_temperature": self.llm_temperature,
                "llm_max_tokens": self.llm_max_tokens,
                "available_models": self.AVAILABLE_MODELS,
            }
            try:
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, indent=2)
                logger.info("Created default LLM config at %s", config_file)
            except Exception as e:
                logger.warning("Failed to write default LLM config: %s", e)

    def reload_config(self) -> bool:
        """Hot-reload LLM config from llm_config.json.

        Called at the start of each cycle so changes take effect without restart.
        Returns True if config was reloaded (changed), False otherwise.
        """
        if not self.config_path:
            return False
        config_file = Path(self.config_path)
        if not config_file.exists():
            return False
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            changed = False
            new_model = data.get("llm_model", self.llm_model)
            new_url = data.get("llm_url", self.llm_url)
            new_temp = data.get("llm_temperature", self.llm_temperature)
            new_max_tokens = data.get("llm_max_tokens", self.llm_max_tokens)
            # Validate model is in available list
            valid_ids = [m["id"] if isinstance(m, dict) else m for m in self.AVAILABLE_MODELS]
            if new_model != self.llm_model:
                if new_model not in valid_ids:
                    logger.warning("Invalid model in config: %s, keeping %s", new_model, self.llm_model)
                else:
                    logger.info("LLM model changed: %s -> %s", self.llm_model, new_model)
                    self.llm_model = new_model
                    changed = True
            if new_url != self.llm_url:
                logger.info("LLM URL changed: %s -> %s", self.llm_url, new_url)
                self.llm_url = new_url.rstrip("/")
                changed = True
            if new_temp != self.llm_temperature:
                self.llm_temperature = float(new_temp)
                changed = True
            if new_max_tokens != self.llm_max_tokens:
                self.llm_max_tokens = int(new_max_tokens)
                changed = True
            if changed:
                logger.info("LLM config reloaded: model=%s, url=%s", self.llm_model, self.llm_url)
            return changed
        except Exception as e:
            logger.warning("Failed to reload LLM config: %s", e)
            return False

    def get_config_status(self) -> Dict[str, Any]:
        """Return current LLM config and status for the dashboard."""
        return {
            "llm_model": self.llm_model,
            "llm_url": self.llm_url,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "connected": self.connected,
            "total_tokens_used": self.total_tokens_used,
            "available_models": self.AVAILABLE_MODELS,
        }

    def load_strategy_rules(self) -> Dict[str, Any]:
        """Load self-reflection strategy rules from strategy_rules.json.

        Hot-reloads the rules from disk each cycle so the running process
        picks up changes made by SelfReflect.reflect().

        Returns:
            The loaded rules dict, or empty dict if file doesn't exist.
        """
        rules_path = Path(self.config_path).parent / "strategy_rules.json" if self.config_path else Path("/root/PROJECTS/picsou/data/strategy_rules.json")
        if not rules_path.exists():
            self.strategy_rules = {}
            return self.strategy_rules
        try:
            with open(rules_path, "r", encoding="utf-8") as f:
                self.strategy_rules = json.load(f)
            logger.info("Loaded %d strategy rules from %s",
                        len(self.strategy_rules.get("rules", [])),
                        rules_path)
            return self.strategy_rules
        except Exception as e:
            logger.warning("Failed to load strategy rules: %s", e)
            self.strategy_rules = {}
            return self.strategy_rules

    def _build_strategy_rules_section(self) -> str:
        """Build the strategy rules injection section for the LLM prompt."""
        if not self.strategy_rules or not self.strategy_rules.get("rules"):
            return ""

        rules = self.strategy_rules.get("rules", [])
        adjustments = self.strategy_rules.get("strategy_adjustments", {})
        params = self.strategy_rules.get("parameter_changes", {})

        lines = ["SELF-IMPROVEMENT RULES (learned from past performance):"]
        for rule in rules:
            lines.append(f"- {rule}")

        # Strategy adjustments summary
        if adjustments:
            adj_parts = []
            for name, adj in adjustments.items():
                boost = adj.get("weight_boost", 0)
                penalty = adj.get("weight_penalty", 0)
                if boost:
                    adj_parts.append(f"{name} +{boost:.0%}")
                if penalty:
                    adj_parts.append(f"{name} -{penalty:.0%}")
            if adj_parts:
                lines.append(f"Strategy adjustments: {', '.join(adj_parts)}")

        # Risk parameters summary
        if params:
            param_parts = []
            if "preferred_position_pct" in params:
                param_parts.append(f"position size {params['preferred_position_pct']:.0%}")
            if params.get("avoid_high_fear"):
                param_parts.append("avoid high fear markets (F&G < 25)")
            if params.get("avoid_extreme_greed"):
                param_parts.append("avoid extreme greed markets (F&G > 75)")
            if param_parts:
                lines.append(f"Risk parameters: {', '.join(param_parts)}")

        return "\n".join(lines) + "\n"

    # ── Market data summarizer ────────────────────────────────────────

    def _summarize_market(self, market_data: Dict[str, Dict[str, Any]]) -> str:
        """Build a concise text summary of current market conditions."""
        lines = []
        for key, data in market_data.items():
            ticker = data.get("ticker", {})
            candles = data.get("candles", [])
            ob = data.get("order_book", {})

            last_price = ticker.get("last", 0)
            vol_24h = ticker.get("volume", 0)

            # Recent price trend from last 5 candles
            if len(candles) >= 5:
                recent = candles[-5:]
                prices = [c["close"] for c in recent]
                trend_pct = (prices[-1] - prices[0]) / prices[0] * 100 if prices[0] > 0 else 0
                high_5 = max(c["high"] for c in recent)
                low_5 = min(c["low"] for c in recent)
                trend_str = f"5-candle trend: {trend_pct:+.2f}%, range: {low_5:.2f}-{high_5:.2f}"
            else:
                trend_str = "insufficient candle data"

            # Order book imbalance
            bid_total = sum(b[1] for b in ob.get("bids", [])[:10]) if ob else 0
            ask_total = sum(a[1] for a in ob.get("asks", [])[:10]) if ob else 0
            ob_str = f"ob_imbalance: {bid_total:.1f}/{ask_total:.1f} (bid/ask)" if bid_total or ask_total else "no orderbook"

            # Volume from last candle
            last_vol = candles[-1].get("volume", 0) if candles else 0

            lines.append(
                f"  {key}: price={last_price:.2f}, vol_24h={vol_24h:.0f}, "
                f"last_vol={last_vol:.1f}, {trend_str}, {ob_str}"
            )

        return "\n".join(lines) if lines else "  No market data available"

    def _summarize_portfolio(self, portfolio_mgr: Any) -> str:
        """Build a text summary of current portfolio state."""
        try:
            balance = portfolio_mgr.get_balance()
            pnl = portfolio_mgr.get_pnl()
            open_positions = portfolio_mgr.get_open_positions()
            pos_lines = []
            for p in open_positions:
                pos_lines.append(
                    f"    {p.side} {p.amount:.6f} {p.symbol} @ {p.entry_price:.2f} "
                    f"(entry {p.open_time[:16]})"
                )
            pos_str = "\n".join(pos_lines) if pos_lines else "    (no open positions)"

            return (
                f"  Balance: {balance:.2f} EUR\n"
                f"  PnL: {pnl['total_pnl']:.2f} EUR ({pnl['return_pct']:.2f}%)\n"
                f"  Win rate: {pnl['win_rate']:.2%} ({pnl['total_trades']} trades)\n"
                f"  Open positions ({len(open_positions)}):\n{pos_str}"
            )
        except Exception as e:
            return f"  Portfolio summary error: {e}"

    def _summarize_recent_decisions(self, journal: Any, limit: int = 5) -> str:
        """Summarize the last N decisions for context."""
        try:
            recent = journal.get_recent(limit=limit)
            if not recent:
                return "  (no recent decisions)"
            lines = []
            for d in recent:
                lines.append(
                    f"    {d.get('timestamp', '?')[:16]}: "
                    f"{d.get('action', '?')} {d.get('symbol', '?')} "
                    f"conf={d.get('confidence', 0):.2f} "
                    f"strat={d.get('strategy', '?')} "
                    f"reason={d.get('reasoning', '?')[:80]}"
                )
            return "\n".join(lines)
        except Exception:
            return "  (no recent decisions)"

    def _summarize_sentiment(self, fng: Dict[str, Any],
                             headlines: List[str]) -> str:
        """Build sentiment context string."""
        parts = []
        if fng:
            fng_val = fng.get("value", "?")
            fng_cls = fng.get("classification", "?")
            yesterday = fng.get("yesterday", "")
            change = ""
            if yesterday and isinstance(fng_val, int):
                change = f" (vs yesterday: {yesterday}, delta={fng_val - yesterday:+d})"
            parts.append(f"  Fear & Greed Index: {fng_val}/100 ({fng_cls}){change}")
        else:
            parts.append("  Fear & Greed Index: unavailable")

        if headlines:
            parts.append("  Recent headlines:")
            for h in headlines[:5]:
                parts.append(f"    - {h[:120]}")
        else:
            parts.append("  Recent headlines: unavailable")

        return "\n".join(parts)

    def _summarize_research(self, research_insights: Optional[Dict[str, Any]]) -> str:
        """Build a research insights section for the LLM prompt.

        Injects web-sourced strategy insights, technical signals, and
        risk factors as supplementary context for decision-making.
        """
        if not research_insights:
            return ""

        lines = ["\nRESEARCH INSIGHTS (web-sourced, for context only):"]

        narrative = research_insights.get("market_narrative", "")
        if narrative:
            lines.append(f"  Market narrative: {narrative}")

        trending = research_insights.get("trending_strategies", [])
        if trending:
            lines.append(f"  Trending strategies: {', '.join(trending)}")

        signals = research_insights.get("technical_signals", [])
        if signals:
            lines.append(f"  Technical signals mentioned online: {', '.join(signals)}")

        risks = research_insights.get("risk_factors", [])
        if risks:
            for r in risks[:3]:
                lines.append(f"  Risk: {r}")

        lines.append(
            "  Consider these research-backed strategies alongside your own analysis."
        )

        return "\n".join(lines)

    def _summarize_market_context(self, market_context: Optional[Dict[str, Any]]) -> str:
        """Format the market_context from MarketAwareness for LLM prompt injection.

        Produces a clear, structured summary of market regime, macro events,
        sentiment dynamics, macro indicators, LLM macro assessment, correlation
        rules, and behavioral recommendations.
        """
        if not market_context:
            return ""

        lines = ["\n═══ MARKET AWARENESS (macro perception) ═══"]

        # ── Market Regime ──
        regime = market_context.get("market_regime", {})
        if regime:
            regime_name = regime.get("regime", "unknown").upper()
            confidence = regime.get("confidence", 0)
            vol_level = regime.get("volatility_level", "unknown")
            trend_strength = regime.get("trend_strength", 0)
            description = regime.get("description", "")
            lines.append(f"MARKET REGIME: {regime_name} (confidence: {confidence:.0%})")
            lines.append(f"  Volatility: {vol_level} | Trend strength: {trend_strength:.0%}")
            lines.append(f"  {description}")

        # ── Macro Indicators (real data: DXY, yields, VIX, BTC.D, total mcap) ──
        indicators = market_context.get("macro_indicators", {})
        if indicators:
            lines.append("MACRO INDICATORS:")
            dxy = indicators.get("dxy")
            if dxy:
                lines.append(f"  💵 DXY (Dollar Index): {dxy.get('value', 'N/A')} ({dxy.get('zone', 'unknown')})")
                if dxy.get("trend"):
                    lines.append(f"     Trend: {dxy['trend']}")
            yields = indicators.get("us_10y_yield")
            if yields:
                lines.append(f"  📊 US 10Y Yield: {yields.get('value', 'N/A')}% ({yields.get('zone', 'unknown')})")
                if yields.get("trend"):
                    lines.append(f"     Trend: {yields['trend']}")
            vix = indicators.get("vix")
            if vix:
                lines.append(f"  😨 VIX: {vix.get('value', 'N/A')} ({vix.get('zone', 'unknown')})")
                if vix.get("trend"):
                    lines.append(f"     Trend: {vix['trend']}")
            btc_d = indicators.get("btc_dominance")
            if btc_d:
                lines.append(f"  ₿ BTC Dominance: {btc_d.get('value', 'N/A')}% ({btc_d.get('zone', 'unknown')})")
            total_mcap = indicators.get("total_market_cap")
            if total_mcap:
                lines.append(f"  🌐 Total Crypto Mcap: {total_mcap.get('description', 'N/A')}")
        else:
            lines.append("MACRO INDICATORS: Data unavailable (API may be down)")

        # ── Macro Assessment (LLM reasoning) ──
        macro_assessment = market_context.get("macro_assessment")
        if macro_assessment:
            lines.append("MACRO ASSESSMENT (LLM reasoning):")
            lines.append(f"  Regime confirmation: {macro_assessment.get('regime_confirmation', '?')}")
            lines.append(f"  Risk outlook: {macro_assessment.get('risk_outlook', '?')}")
            lines.append(f"  Time horizon: {macro_assessment.get('time_horizon', '?')}")
            lines.append(f"  Reasoning: {macro_assessment.get('reasoning', '')}")
            lines.append(f"  Advice: {macro_assessment.get('actionable_advice', '')}")

        # ── Sentiment Profile ──
        sentiment = market_context.get("sentiment_profile", {})
        if sentiment:
            cls = sentiment.get("classification", "unknown")
            trend = sentiment.get("trend", "unknown")
            score = sentiment.get("score", "?")
            delta = sentiment.get("trend_delta", 0)
            desc = sentiment.get("description", "")
            lines.append(f"SENTIMENT PROFILE: {cls} (score: {score}/100, trend: {trend}, Δ{delta:+d})")
            lines.append(f"  {desc}")

        # ── Macro Events ──
        events = market_context.get("macro_events", [])
        if events:
            lines.append(f"MACRO EVENTS ({len(events)} detected):")
            for event in events[:5]:  # Top 5 by magnitude
                impact = event.get("impact", "unknown")
                magnitude = event.get("magnitude", 1)
                keyword = event.get("keyword", "unknown")
                category = event.get("category", "unknown")
                impact_emoji = "🔴" if impact == "negative" else "🟢" if impact == "positive" else "🟡"
                lines.append(f"  {impact_emoji} {keyword}: {impact} (magnitude {magnitude}/5, {category})")
        else:
            lines.append("MACRO EVENTS: No significant macro events detected")

        # ── Recent News Headlines (from RSS feeds) ──
        # Inject raw headlines so the LLM can reason about them directly
        rss_headlines = market_context.get("rss_headlines", [])
        if rss_headlines:
            lines.append(f"RECENT NEWS HEADLINES ({len(rss_headlines)} headlines):")
            for h in rss_headlines[:10]:
                source = h.get("source", "unknown")
                title = h.get("title", "")
                if title:
                    lines.append(f"  - [{source}] {title}")
        else:
            lines.append("RECENT NEWS HEADLINES: No headlines available")

        # ── Deep Research Findings ──
        deep_research = market_context.get("deep_research", [])
        research_queries = market_context.get("research_queries", [])
        fg_trend = market_context.get("fg_trend", {})
        cg_context = market_context.get("coingecko_context", {})

        if fg_trend:
            lines.append("SENTIMENT TREND (30 days):")
            current_cls = fg_trend.get("current_classification", "")
            week_cls = fg_trend.get("week_ago_classification", "")
            month_cls = fg_trend.get("month_ago_classification", "")
            lines.append(f"  Current: {current_cls} ({fg_trend.get('current', '?')})")
            lines.append(f"  Week ago: {week_cls} ({fg_trend.get('week_ago', '?')}) → {fg_trend.get('change_week', 0):+d} pts ({fg_trend.get('trend', 'unknown')})")
            lines.append(f"  Month ago: {month_cls} ({fg_trend.get('month_ago', '?')}) → {fg_trend.get('change_month', 0):+d} pts")

        if cg_context:
            cg_global = cg_context.get("global", {})
            cg_trending = cg_context.get("trending", [])
            if cg_global:
                lines.append("MARKET CONTEXT (CoinGecko):")
                btc_dom = cg_global.get("btc_dominance", "N/A")
                btc_note = ""
                if isinstance(btc_dom, (int, float)):
                    if btc_dom > 55:
                        btc_note = " (BTC dominant, alt season not yet)"
                    elif btc_dom < 42:
                        btc_note = " (alt season in progress)"
                lines.append(f"  BTC Dominance: {btc_dom}%{btc_note}")
                lines.append(f"  ETH Dominance: {cg_global.get('eth_dominance', 'N/A')}%")
                mcap = cg_global.get("total_market_cap_usd", 0)
                if isinstance(mcap, (int, float)) and mcap > 0:
                    lines.append(f"  Total Market Cap: ${mcap / 1e9:.1f}B")
                chg = cg_global.get("market_cap_change_24h_pct", "N/A")
                if isinstance(chg, (int, float)):
                    lines.append(f"  24h Change: {chg:+.1f}%")
                else:
                    lines.append(f"  24h Change: {chg}")
            if cg_trending:
                trending_names = [f"{c['name']} ({c['symbol']})" for c in cg_trending[:5] if c.get("name")]
                meme_keywords = ["pepe", "bonk", "wif", "doge", "shib", "floki", "meme"]
                trending_lower = " ".join(trending_names).lower()
                meme_trending = any(kw in trending_lower for kw in meme_keywords)
                meme_note = " (⚠️ meme coins trending = risk-on speculation)" if meme_trending else ""
                lines.append(f"  Trending: [{', '.join(trending_names)}]{meme_note}")

        if research_queries:
            lines.append("RESEARCH QUERIES GENERATED:")
            for q in research_queries:
                lines.append(f"  - \"{q}\"")

        if deep_research:
            lines.append("DEEP RESEARCH FINDINGS:")
            for r in deep_research:
                source = r.get("source", "unknown")
                topic = r.get("topic", "")
                summary = r.get("summary", "")[:200]
                lines.append(f"  - [{source}] {topic}: {summary}")

        # ── Macro Correlation Rules (from knowledge base) ──
        # Inject the rules from the macro knowledge loaded by market_awareness
        lines.append("MACRO CORRELATION RULES:")
        lines.append("  • DXY > 103 + VIX > 25 → strong crypto sell-off (risk-off)")
        lines.append("  • DXY < 95 + VIX < 15 → strong crypto rally (risk-on)")
        lines.append("  • 10Y Yield > 4.5% → capital flows to bonds, pressuring BTC")
        lines.append("  • BTC.D < 42% → altcoin season, SOL/ETH outperform")
        lines.append("  • BTC.D > 58% → alt season ending, prefer BTC or cash")
        lines.append("  • VIX > 35 → liquidity crisis likely, minimize positions")
        lines.append("  • Recent Fed cut → bullish momentum, increase positions")
        lines.append("  • Recent Fed hike → bearish, reduce positions")

        # ── Behavioral Recommendations ──
        recs = market_context.get("behavioral_recommendations", [])
        if recs:
            lines.append("BEHAVIORAL ADAPTATIONS:")
            for rec in recs:
                priority = rec.get("priority", "medium")
                category = rec.get("category", "general")
                recommendation = rec.get("recommendation", "")
                priority_marker = "⚠️" if priority == "critical" else "⚡" if priority == "high" else "💡"
                lines.append(f"  {priority_marker} [{category.upper()}] {recommendation}")

        lines.append("════════════════════════════════════════════")

        return "\n".join(lines)

    # ── Prompt builder ─────────────────────────────────────────────────

    def _build_prompt(self, market_data: Dict[str, Dict[str, Any]],
                      portfolio_mgr: Any,
                      journal: Any,
                      sentiment_fng: Dict[str, Any],
                      sentiment_headlines: List[str],
                      symbols: List[str],
                      risk_config: Any,
                      research_insights: Optional[Dict[str, Any]] = None,
                      tech_summary: str = "",
                      market_context: Optional[Dict[str, Any]] = None) -> str:
        """Build the full LLM prompt with all context."""
        # Build learning context section
        learning_section = ""
        if self.learning_context:
            scores = self.learning_context.get("strategy_scores", {})
            active = self.learning_context.get("active_strategies", [])
            underexplored = self.learning_context.get("underexplored_strategies", [])

            if scores:
                learning_lines = ["CURRENT STRATEGY PERFORMANCE:"]
                for name, info in scores.items():
                    status = "ACTIVE" if info["active"] else "ELIMINATED"
                    learning_lines.append(
                        f"  - {name}: win_rate={info['win_rate']:.0%}, "
                        f"trades={info['total_trades']}, weight={info['weight']:.2f} [{status}]"
                    )
                learning_section = "\n".join(learning_lines) + "\n"

            if active:
                learning_section += f"\nActive strategies: {', '.join(active)}\n"

        # Build self-improvement rules section (from reflection)
        strategy_rules_section = self._build_strategy_rules_section()

        # Build exploration directive
        exploration_directive = ""
        if self.exploration_mode and self.underexplored_strategies:
            strategies_str = ", ".join(self.underexplored_strategies)
            exploration_directive = f"""
⚠️ EXPLORATION PHASE — MANDATORY TRADING DIRECTIVE ⚠️
You are in LEARNING/EXPLORATION mode. The following strategies have insufficient trade data
and MUST be tested: [{strategies_str}]

You MUST place at least one small trade (amount_pct: 0.02-0.05) for EACH under-explored strategy.
This is critical for learning — without trade data, we cannot evaluate strategy performance.

DO NOT return an empty array []. Even in uncertain market conditions, small exploratory positions
are required. Use:
- "contrarian": Buy when fear is high (counter-trend)
- "breakout": Buy when price breaks above recent resistance
- "dca": Dollar-cost average into positions
- "mean_reversion": Trade against overextended moves
- "momentum": Follow the current trend direction

LOW Fear & Greed (< 30) is NOT a reason to hold in exploration mode — it's actually an opportunity
for contrarian/DCA strategies. Fear creates buying opportunities.
"""

        # Fear & Greed context adjustment for exploration
        fng_context = ""
        if sentiment_fng:
            fng_val = sentiment_fng.get("value", 50)
            fng_cls = sentiment_fng.get("classification", "Neutral")
            if self.exploration_mode:
                fng_context = f"""Fear & Greed Index: {fng_val}/100 ({fng_cls})
Note: In exploration mode, low Fear & Greed is an OPPORTUNITY for contrarian strategies, not a reason to hold."""
            else:
                fng_context = f"""Fear & Greed Index: {fng_val}/100 ({fng_cls})"""

        return f"""You are Picsou, an autonomous crypto trading agent. Analyze the current market conditions, sentiment, and portfolio state, then decide what actions to take.

{learning_section}
{strategy_rules_section}
{exploration_directive}
{self._summarize_market_context(market_context)}
CURRENT MARKET DATA:
{self._summarize_market(market_data)}

{tech_summary}
PORTFOLIO STATE:
{self._summarize_portfolio(portfolio_mgr)}

RECENT DECISIONS (last 5):
{self._summarize_recent_decisions(journal)}

MARKET SENTIMENT:
{self._summarize_sentiment(sentiment_fng, sentiment_headlines)}
{fng_context}
{self._summarize_research(research_insights)}
RISK RULES:
- Max {int(risk_config.max_position_pct * 100) if hasattr(risk_config, 'max_position_pct') else 20}% of capital per position
- Max {getattr(risk_config, 'max_open_positions', 5)} open positions simultaneously
- Max {int(getattr(risk_config, 'max_drawdown_pct', 0.20) * 100)}% drawdown → pause all trading
- Available symbols: {', '.join(symbols)}
- Only trade on configured exchanges with available data

CRITICAL TRADING RULES:
1. MINIMUM HOLD TIME: You MUST hold positions for at least 15 minutes before selling. Rapid round-trips (buy then sell within minutes) ALWAYS lose money because trading fees eat the tiny price moves. Selling before 15 min is REJECTED by the system.
2. FEE AWARENESS: Each trade pays 0.08-0.26% in fees (OKX 0.08%, Kraken 0.26%, Bitstamp 0.25%). A round-trip costs 0.16-0.52% in fees alone. Only trade if you expect a move larger than 1% to cover fees and profit.
3. MINIMUM EXPECTED MOVE: Do NOT buy unless you expect at least 1.5% price movement. Small micro-moves (<0.5%) are noise, not actionable signals.
4. ONLY SELL WHAT YOU HOLD: You can only sell assets you currently have open positions in. Check the portfolio state above for your open positions.

STRATEGY TYPES (use ONLY these — no other values allowed):
- "momentum": Trend-following trades
- "mean_reversion": Trading against overextended moves
- "breakout": Trading breakouts from ranges
- "contrarian": Counter-trend at extremes
- "dca": Dollar-cost averaging entries
- "risk_management": Hold/defensive decisions

{self.trading_knowledge}

OUTPUT FORMAT — respond with a JSON array of decisions. Each decision MUST have:
- "action": "buy", "sell", or "hold"
- "symbol": e.g. "BTC", "ETH", "SOL"
- "exchange": which exchange to use (okx, kraken, or bitstamp)
- "amount_pct": percentage of available balance to use (0.05 to 0.20 for buy, 1.0 for full close on sell)
- "confidence": your confidence from 0.0 to 1.0
- "reasoning": detailed explanation of WHY you chose this action (1-3 sentences, include expected move direction and magnitude)
- "strategy_type": one of: momentum, mean_reversion, breakout, contrarian, dca, risk_management

If you decide to hold for all symbols, return an empty array []."""

    # ── LLM call ──────────────────────────────────────────────────────

    def ask_brain(self,
                  market_data: Dict[str, Dict[str, Any]],
                  portfolio_mgr: Any,
                  journal: Any,
                  symbols: List[str],
                  risk_config: Any = None,
                  research_insights: Optional[Dict[str, Any]] = None,
                  market_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Send context to the LLM and get structured decisions back.

        Returns a list of decision dicts. Falls back to EMA crossover
        if the LLM call fails.
        """
        # 1. Fetch sentiment data
        fng = {}
        headlines = []
        if self.fear_and_greed_enabled:
            fng = fetch_fear_and_greed()
        if self.news_enabled:
            headlines = fetch_crypto_headlines()

        # 2. Generate technical analysis from candle data
        tech_summary = ""
        tech_indicators: Dict[str, Any] = {}
        try:
            candles_dict = {}
            for key, data in market_data.items():
                candles = data.get("candles", [])
                symbol = data.get("base", key.split(":")[-1] if ":" in key else key)
                if candles:
                    candles_dict[symbol] = candles
            if candles_dict:
                tech_summary, tech_indicators = generate_technical_summary(candles_dict)
                logger.info("Technical analysis generated for %d symbols", len(candles_dict))
                # Store for reuse by market awareness
                self._last_tech_indicators = tech_indicators
        except Exception as e:
            logger.warning("Technical analysis generation failed: %s", e)

        # 3. Build prompt (with technical analysis, research insights, and market context)
        prompt = self._build_prompt(
            market_data, portfolio_mgr, journal,
            fng, headlines, symbols, risk_config,
            research_insights=research_insights,
            tech_summary=tech_summary,
            market_context=market_context,
        )
        self.last_prompt = prompt

        # 3. Try LLM call
        decisions = self._call_llm(prompt)

        if decisions is not None:
            logger.info("LLM returned %d decisions", len(decisions))
            return decisions

        # 4. Fallback: EMA crossover on available data
        logger.warning("LLM unavailable, falling back to EMA crossover signals")
        self.connected = False
        fallback = []
        for key, data in market_data.items():
            candles = data.get("candles", [])
            symbol = data.get("base", "")
            exchange = data.get("exchange", "")
            sig = fallback_ema_signal(candles, symbol, exchange)
            if sig:
                fallback.append(sig)

        # Store fallback info
        self.last_response = "FALLBACK: EMA crossover"
        self.last_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        logger.info("Fallback produced %d signals", len(fallback))
        return fallback

    def _call_llm(self, prompt: str) -> Optional[List[Dict[str, Any]]]:
        """Call the Mistral AI (OpenAI-compatible) LLM API.

        Returns parsed decision list or None on failure.
        """
        # Hot-reload config from file before each call
        self.reload_config()

        if not self.llm_api_key:
            logger.warning("No LLM API key configured, skipping LLM call")
            self.connected = False
            return None

        url = f"{self.llm_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_api_key}",
        }
        payload = {
            "model": self.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Picsou, an autonomous cryptocurrency trading agent. "
                        "You analyze market data, sentiment, and portfolio state to make "
                        "trading decisions. You always respond with valid JSON arrays. "
                        "Do NOT wrap your response in markdown code fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
        }
        # Only add response_format for providers that support it (not Ollama)
        if not self.llm_url.rstrip("/").endswith("/v1") or "mistral" in self.llm_url:
            payload["response_format"] = {"type": "json_object"}

        try:
            # Retry logic: up to 2 retries with exponential backoff for cloud model timeouts
            max_retries = 2
            resp = None  # type: Optional[requests.Response]
            for attempt in range(max_retries + 1):
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=120)
                    break
                except requests.exceptions.Timeout:
                    if attempt < max_retries:
                        wait = 5 * (attempt + 1)
                        logger.warning("LLM API call timeout (attempt %d/%d), retrying in %ds...", attempt + 1, max_retries + 1, wait)
                        time.sleep(wait)
                    else:
                        raise
            if resp is None:
                logger.error("LLM API call failed: no response received after retries")
                return None
            resp.raise_for_status()
            data = resp.json()

            # Extract response — some models (e.g. reasoning models via Ollama)
            # put output in "reasoning" while "content" stays empty.
            msg = data.get("choices", [{}])[0].get("message", {})
            content = msg.get("content", "") or msg.get("reasoning", "") or ""
            usage = data.get("usage", {})
            self.last_tokens = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            self.total_tokens_used += self.last_tokens.get("total_tokens", 0)
            self.last_response = content
            self.connected = True

            logger.info("LLM response received: %d tokens (prompt=%d, completion=%d)",
                        self.last_tokens["total_tokens"],
                        self.last_tokens["prompt_tokens"],
                        self.last_tokens["completion_tokens"])

            # Strip markdown code fences if present (Ollama/proxy models may wrap JSON)
            raw = content.strip()
            if raw.startswith("```"):
                # Remove opening ```json or ``` and closing ```
                first_newline = raw.find("\n")
                if first_newline != -1:
                    raw = raw[first_newline + 1:]
                if raw.endswith("```"):
                    raw = raw[:-3].rstrip()

            # Parse JSON response — may be a list directly or wrapped in an object
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as e:
                # Truncated JSON: try to recover by closing open structures
                if "Unterminated" in str(e) or "Expecting" in str(e):
                    logger.warning("Attempting JSON recovery for truncated response (error: %s)", e)
                    # Try to find the last complete object in an array
                    import re
                    # Find last complete {...} before truncation
                    last_brace = raw.rfind("}")
                    if last_brace > 0:
                        truncated = raw[:last_brace + 1]
                        # Try to close the array if it was opened
                        if truncated.lstrip().startswith("["):
                            truncated = truncated.rstrip() + "]"
                        try:
                            parsed = json.loads(truncated)
                            logger.info("JSON recovery succeeded: %d decisions", len(parsed) if isinstance(parsed, list) else 1)
                        except json.JSONDecodeError:
                            logger.error("JSON recovery failed")
                            raise e
                    else:
                        raise e
                else:
                    raise e

            # Handle {"decisions": [...]} wrapper
            if isinstance(parsed, dict):
                decisions = parsed.get("decisions", parsed.get("actions", []))
            elif isinstance(parsed, list):
                decisions = parsed
            else:
                logger.warning("Unexpected LLM response format: %s", type(parsed))
                return None

            # Validate and normalize each decision
            validated = []
            for d in decisions:
                if not isinstance(d, dict):
                    continue
                action = d.get("action", "").lower()
                if action not in ("buy", "sell", "hold"):
                    continue
                # Ensure required fields
                d.setdefault("symbol", "")
                d.setdefault("exchange", "")
                d.setdefault("amount_pct", 0.05)
                d.setdefault("confidence", 0.5)
                d.setdefault("reasoning", "LLM decision (no reasoning provided)")
                # Validate strategy_type against whitelist
                VALID_STRATEGIES = {"momentum", "mean_reversion", "breakout",
                                    "contrarian", "dca", "risk_management", "llm_driven"}
                strategy_type = d.get("strategy_type", "risk_management")
                if strategy_type not in VALID_STRATEGIES:
                    # Map unknown types to closest match or default to risk_management
                    strategy_type = "risk_management"
                    d["strategy_type"] = strategy_type
                # Clamp confidence
                d["confidence"] = max(0.0, min(1.0, float(d["confidence"])))
                # Clamp amount_pct
                d["amount_pct"] = max(0.01, min(0.20, float(d["amount_pct"])))
                validated.append(d)

            return validated

        except requests.exceptions.Timeout:
            logger.error("LLM API call timed out")
            self.connected = False
            return None
        except requests.exceptions.ConnectionError:
            logger.error("LLM API connection error")
            self.connected = False
            return None
        except json.JSONDecodeError as e:
            logger.error("LLM response JSON parse error: %s", e)
            self.last_response = content if 'content' in dir() else str(data)
            self.connected = True  # Connected but bad response
            return None
        except Exception as e:
            logger.error("LLM API call failed: %s", e)
            self.connected = False
            return None