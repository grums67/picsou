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
from typing import Any, Dict, List, Optional

import requests

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

        # Write initial config file if it doesn't exist
        self._write_default_config()

    # ── Hot-reload config management ──────────────────────────────────────

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

    # ── Prompt builder ─────────────────────────────────────────────────

    def _build_prompt(self, market_data: Dict[str, Dict[str, Any]],
                      portfolio_mgr: Any,
                      journal: Any,
                      sentiment_fng: Dict[str, Any],
                      sentiment_headlines: List[str],
                      symbols: List[str],
                      risk_config: Any,
                      research_insights: Optional[Dict[str, Any]] = None) -> str:
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
{exploration_directive}
CURRENT MARKET DATA:
{self._summarize_market(market_data)}

PORTFOLIO STATE:
{self._summarize_portfolio(portfolio_mgr)}

RECENT DECISIONS (last 5):
{self._summarize_recent_decisions(journal)}

MARKET SENTIMENT:
{self._summarize_sentiment(sentiment_fng, sentiment_headlines)}
{fng_context}
{self._summarize_research(research_insights)}
RISK RULES:
- Max 20% of capital per position
- Max 5 open positions simultaneously
- Max 20% drawdown → pause all trading
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
                  research_insights: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
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

        # 2. Build prompt (with optional research insights)
        prompt = self._build_prompt(
            market_data, portfolio_mgr, journal,
            fng, headlines, symbols, risk_config,
            research_insights=research_insights,
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
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
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