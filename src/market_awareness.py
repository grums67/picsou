"""MarketAwareness - Macro market perception module for Picsou trading agent.

Detects market regime (bull/bear/range/volatile/crunch), scans macro events,
profiles sentiment dynamics, fetches real macro indicators (DXY, yields, VIX,
BTC dominance, total market cap), reasons about macro conditions via LLM,
and produces behavioral adaptation recommendations.

This module allows Picsou to UNDERSTAND the market like a macro economist,
not just scan keywords. It fetches real macro data, knows correlations, and
uses the LLM to produce actionable macro assessments.
"""

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

from src.technical_analysis import (
    _ema,
    calc_atr,
    calc_bollinger_bands,
    calc_rsi,
    calc_volume_profile,
)

logger = logging.getLogger(__name__)


# ── Macro event keywords and their impact mapping (kept for headline scanning) ──

MACRO_KEYWORDS: Dict[str, Dict[str, Any]] = {
    "trump": {"impact": "negative", "magnitude": 4, "category": "geopolitical"},
    "tariff": {"impact": "negative", "magnitude": 4, "category": "geopolitical"},
    "tariffs": {"impact": "negative", "magnitude": 4, "category": "geopolitical"},
    "executive order": {"impact": "negative", "magnitude": 3, "category": "regulation"},
    "sec": {"impact": "negative", "magnitude": 3, "category": "regulation"},
    "sec crackdown": {"impact": "negative", "magnitude": 5, "category": "regulation"},
    "ban": {"impact": "negative", "magnitude": 5, "category": "regulation"},
    "banned": {"impact": "negative", "magnitude": 5, "category": "regulation"},
    "regulation": {"impact": "negative", "magnitude": 3, "category": "regulation"},
    "hack": {"impact": "negative", "magnitude": 4, "category": "security"},
    "hacked": {"impact": "negative", "magnitude": 4, "category": "security"},
    "exploit": {"impact": "negative", "magnitude": 4, "category": "security"},
    "rug pull": {"impact": "negative", "magnitude": 4, "category": "security"},
    "fed": {"impact": "negative", "magnitude": 4, "category": "monetary"},
    "rate hike": {"impact": "negative", "magnitude": 5, "category": "monetary"},
    "rate increase": {"impact": "negative", "magnitude": 5, "category": "monetary"},
    "rate cut": {"impact": "positive", "magnitude": 4, "category": "monetary"},
    "interest rate": {"impact": "negative", "magnitude": 3, "category": "monetary"},
    "hawkish": {"impact": "negative", "magnitude": 4, "category": "monetary"},
    "dovish": {"impact": "positive", "magnitude": 4, "category": "monetary"},
    "etf": {"impact": "positive", "magnitude": 4, "category": "institutional"},
    "approval": {"impact": "positive", "magnitude": 4, "category": "institutional"},
    "etf approval": {"impact": "positive", "magnitude": 5, "category": "institutional"},
    "spot etf": {"impact": "positive", "magnitude": 5, "category": "institutional"},
    "institutional": {"impact": "positive", "magnitude": 3, "category": "institutional"},
    "adoption": {"impact": "positive", "magnitude": 3, "category": "institutional"},
    "musk": {"impact": "positive", "magnitude": 3, "category": "influencer"},
    "elon": {"impact": "positive", "magnitude": 3, "category": "influencer"},
    "saylor": {"impact": "positive", "magnitude": 3, "category": "influencer"},
    "bullish": {"impact": "positive", "magnitude": 2, "category": "sentiment"},
    "bearish": {"impact": "negative", "magnitude": 2, "category": "sentiment"},
    "crash": {"impact": "negative", "magnitude": 5, "category": "market"},
    "plunge": {"impact": "negative", "magnitude": 4, "category": "market"},
    "liquidation": {"impact": "negative", "magnitude": 4, "category": "market"},
    "depeg": {"impact": "negative", "magnitude": 5, "category": "market"},
    "all time high": {"impact": "positive", "magnitude": 4, "category": "market"},
    "ath": {"impact": "positive", "magnitude": 4, "category": "market"},
    "breakout": {"impact": "positive", "magnitude": 3, "category": "market"},
    "geopolitical": {"impact": "negative", "magnitude": 3, "category": "geopolitical"},
    "war": {"impact": "negative", "magnitude": 4, "category": "geopolitical"},
    "sanctions": {"impact": "negative", "magnitude": 3, "category": "geopolitical"},
    "recession": {"impact": "negative", "magnitude": 4, "category": "economic"},
    "inflation": {"impact": "negative", "magnitude": 3, "category": "economic"},
    # ── Additional macro keywords for RSS-enriched detection ──
    "fomc": {"impact": "negative", "magnitude": 4, "category": "monetary"},
    "fomc meeting": {"impact": "negative", "magnitude": 5, "category": "monetary"},
    "fed chair": {"impact": "negative", "magnitude": 4, "category": "monetary"},
    "powell": {"impact": "negative", "magnitude": 4, "category": "monetary"},
    "jerome powell": {"impact": "negative", "magnitude": 4, "category": "monetary"},
    "nonfarm": {"impact": "negative", "magnitude": 3, "category": "economic"},
    "non-farm": {"impact": "negative", "magnitude": 3, "category": "economic"},
    "jobs report": {"impact": "negative", "magnitude": 3, "category": "economic"},
    "unemployment": {"impact": "negative", "magnitude": 3, "category": "economic"},
    "cpi": {"impact": "negative", "magnitude": 4, "category": "economic"},
    "inflation data": {"impact": "negative", "magnitude": 4, "category": "economic"},
    "gdp": {"impact": "positive", "magnitude": 3, "category": "economic"},
    "stimulus": {"impact": "positive", "magnitude": 4, "category": "monetary"},
    "quantitative easing": {"impact": "positive", "magnitude": 4, "category": "monetary"},
    "qe": {"impact": "positive", "magnitude": 4, "category": "monetary"},
    "tapering": {"impact": "negative", "magnitude": 4, "category": "monetary"},
    "etf inflow": {"impact": "positive", "magnitude": 3, "category": "institutional"},
    "etf outflow": {"impact": "negative", "magnitude": 3, "category": "institutional"},
    "spot btc": {"impact": "positive", "magnitude": 4, "category": "institutional"},
    "spot eth": {"impact": "positive", "magnitude": 3, "category": "institutional"},
    "treasury": {"impact": "negative", "magnitude": 2, "category": "monetary"},
    "bond yield": {"impact": "negative", "magnitude": 3, "category": "monetary"},
    "yields rise": {"impact": "negative", "magnitude": 3, "category": "monetary"},
    "yields fall": {"impact": "positive", "magnitude": 3, "category": "monetary"},
    "dxy": {"impact": "negative", "magnitude": 3, "category": "monetary"},
    "dollar strength": {"impact": "negative", "magnitude": 3, "category": "monetary"},
    "stablecoin": {"impact": "neutral", "magnitude": 2, "category": "market"},
    "whale": {"impact": "neutral", "magnitude": 2, "category": "market"},
    "whale move": {"impact": "negative", "magnitude": 3, "category": "market"},
    "liquidation cascade": {"impact": "negative", "magnitude": 5, "category": "market"},
    "funding rate": {"impact": "neutral", "magnitude": 2, "category": "market"},
    "open interest": {"impact": "neutral", "magnitude": 2, "category": "market"},
    "mica": {"impact": "positive", "magnitude": 3, "category": "regulation"},
    "clarity act": {"impact": "positive", "magnitude": 3, "category": "regulation"},
    "strategic reserve": {"impact": "positive", "magnitude": 5, "category": "institutional"},
    "strategic bitcoin": {"impact": "positive", "magnitude": 5, "category": "institutional"},
}

# ── RSS feed sources ──
RSS_SOURCES = {
    "crypto": [
        {"name": "CoinTelegraph", "url": "https://cointelegraph.com/rss"},
        {"name": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/"},
        {"name": "Decrypt", "url": "https://decrypt.co/feed"},
        {"name": "Yahoo Finance Crypto", "url": "https://finance.yahoo.com/news/rssindex?s=cryptocurrency"},
    ],
    "macro": [
        {"name": "Yahoo Finance Market News", "url": "https://finance.yahoo.com/news/rssindex?s=market-news"},
        {"name": "Federal Reserve", "url": "https://www.federalreserve.gov/feeds/press_monetary.xml"},
    ],
}


class MarketAwareness:
    """Macro market awareness module that detects regime, scans events, fetches
    real macro indicators, reasons about them via LLM, and produces behavioral
    adaptation recommendations.

    Usage:
        ma = MarketAwareness(llm_url="...", llm_api_key="...", llm_model="...")
        context = ma.analyze(
            market_data=market_data,
            sentiment=sentiment,
            tech_indicators=tech_indicators,
        )
        # context contains: market_regime, macro_events, sentiment_profile,
        # macro_indicators, macro_assessment, behavioral_recommendations, etc.
    """

    def __init__(self, cache_ttl: int = 900,
                 llm_url: str = "http://127.0.0.1:11434/v1",
                 llm_api_key: str = "",
                 llm_model: str = "kimi-k2.6:cloud",
                 llm_temperature: float = 0.2,
                 config_path: str = "") -> None:
        """Initialize MarketAwareness.

        Args:
            cache_ttl: Cache TTL for macro event search and indicator data (default 15 min).
            llm_url: LLM API URL for macro reasoning.
            llm_api_key: LLM API key.
            llm_model: LLM model name.
            llm_temperature: LLM temperature for macro reasoning (low = more deterministic).
            config_path: Path to config directory for finding macro_knowledge.yaml.
        """
        self.cache_ttl = cache_ttl
        self._macro_events_cache: List[Dict[str, Any]] = []
        self._macro_events_timestamp: float = 0.0
        self._fg_history: List[Dict[str, Any]] = []  # Store F&G readings over time

        # Macro indicator data cache
        self._macro_data_cache: Dict[str, Any] = {}
        self._macro_data_timestamp: float = 0.0

        # RSS headlines cache (15 min TTL)
        self._rss_headlines_cache: List[Dict[str, Any]] = []
        self._rss_headlines_timestamp: float = 0.0

        # Macro assessment (LLM reasoning) cache
        self._macro_assessment_cache: Optional[Dict[str, Any]] = None
        self._macro_assessment_timestamp: float = 0.0

        # Deep research caches
        self._deep_research_cache: List[Dict[str, Any]] = []
        self._deep_research_timestamp: float = 0.0
        self._fg_trend_cache: Dict[str, Any] = {}
        self._fg_trend_timestamp: float = 0.0
        self._coingecko_context_cache: Dict[str, Any] = {}
        self._coingecko_context_timestamp: float = 0.0

        # Config flags for deep research (defaults, can be overridden)
        self.deep_research_enabled: bool = True
        self.deep_research_max_queries: int = 5
        self.deep_research_wikipedia_enabled: bool = True
        self.deep_research_rss_search: bool = True
        self.fg_trend_enabled: bool = True
        self.coingecko_context_enabled: bool = True

        # LLM config for macro reasoning
        self.llm_url = llm_url.rstrip("/")
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature

        # Load macro knowledge base from YAML
        self.macro_knowledge: Dict[str, Any] = {}
        self.macro_knowledge_text: str = ""
        self._load_macro_knowledge(config_path)

    def _load_macro_knowledge(self, config_path: str = "") -> None:
        """Load macro knowledge base from YAML file."""
        knowledge_path = Path(config_path).parent / "macro_knowledge.yaml" if config_path else Path("/root/PROJECTS/picsou/data/macro_knowledge.yaml")
        if not knowledge_path.exists():
            # Try default path
            knowledge_path = Path("/root/PROJECTS/picsou/data/macro_knowledge.yaml")
        if not knowledge_path.exists():
            logger.warning("Macro knowledge file not found: %s", knowledge_path)
            return
        try:
            import yaml
            with open(knowledge_path, "r", encoding="utf-8") as f:
                self.macro_knowledge = yaml.safe_load(f)
            # Build compact text representation for LLM prompt injection
            lines = ["MACRO ECONOMIC KNOWLEDGE:"]
            correlations = self.macro_knowledge.get("macro_correlations", {})
            for name, details in correlations.items():
                lines.append(f"\n  {name.upper()}:")
                for k, v in details.items():
                    if isinstance(v, dict):
                        for threshold, desc in v.items():
                            lines.append(f"    {threshold}: {desc}")
                    else:
                        lines.append(f"    {k}: {v}")

            scenarios = self.macro_knowledge.get("macro_scenarios", {})
            lines.append("\nMACRO SCENARIOS:")
            for name, details in scenarios.items():
                lines.append(f"  {name}: {details.get('description', '')}")
                lines.append(f"    triggers: {', '.join(details.get('triggers', []))}")
                lines.append(f"    action: {details.get('crypto_action', '')}")

            rules = self.macro_knowledge.get("correlation_rules", [])
            lines.append("\nCORRELATION RULES:")
            for rule in rules:
                lines.append(f"  - {rule}")

            self.macro_knowledge_text = "\n".join(lines)
            logger.info("Loaded macro knowledge base (%d chars)", len(self.macro_knowledge_text))
        except ImportError:
            try:
                self.macro_knowledge_text = knowledge_path.read_text(encoding="utf-8")
                logger.info("Loaded macro knowledge as raw text (%d chars)", len(self.macro_knowledge_text))
            except Exception as e:
                logger.warning("Failed to load macro knowledge: %s", e)
        except Exception as e:
            logger.warning("Failed to load macro knowledge YAML: %s", e)

    # ── Main entry point ──────────────────────────────────────────────────

    def analyze(
        self,
        market_data: Dict[str, Dict[str, Any]],
        sentiment: Dict[str, Any],
        tech_indicators: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run full market awareness analysis.

        Args:
            market_data: Market data dict from PicsouAgent.fetch_market_data().
            sentiment: Sentiment dict with 'fear_and_greed' and 'headlines'.
            tech_indicators: Technical indicators dict from generate_technical_summary().

        Returns:
            Complete market_context dict with regime, events, sentiment profile,
            macro indicators, macro assessment, and behavioral recommendations.
        """
        # 1. Detect market regime
        market_regime = self._detect_regime(market_data, tech_indicators)

        # 2. Scan macro events
        macro_events = self._scan_macro_events(sentiment)

        # 3. Profile sentiment dynamics
        sentiment_profile = self._profile_sentiment(sentiment)

        # 4. Fetch macro indicators (DXY, yields, VIX, BTC.D, total market cap)
        macro_indicators = self._fetch_macro_indicators()

        # 5. Generate behavioral recommendations
        behavioral_recs = self._generate_recommendations(
            market_regime, macro_events, sentiment_profile
        )

        # 6. Assemble full context
        market_context = {
            "market_regime": market_regime,
            "macro_events": macro_events,
            "sentiment_profile": sentiment_profile,
            "macro_indicators": macro_indicators,
            "behavioral_recommendations": behavioral_recs,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }

        # 7. Deep research: generate targeted queries and fetch additional context
        if self.deep_research_enabled:
            try:
                # 7a. Fetch Fear & Greed trend (30-day historical)
                if self.fg_trend_enabled:
                    fg_trend = self._get_fear_greed_trend()
                    if fg_trend:
                        market_context["fg_trend"] = fg_trend
                        logger.info("F&G trend: current=%d, week_ago=%d, trend=%s",
                                    fg_trend.get("current", 0),
                                    fg_trend.get("week_ago", 0),
                                    fg_trend.get("trend", "unknown"))
            except Exception as e:
                logger.debug("F&G trend fetch failed (non-critical): %s", e)

            try:
                # 7b. Fetch CoinGecko context (global + trending)
                if self.coingecko_context_enabled:
                    cg_context = self._get_coingecko_context()
                    if cg_context:
                        market_context["coingecko_context"] = cg_context
                        logger.info("CoinGecko context: %d trending coins",
                                    len(cg_context.get("trending", [])))
            except Exception as e:
                logger.debug("CoinGecko context fetch failed (non-critical): %s", e)

            try:
                # 7c. Generate research queries based on current observations
                research_queries = self._generate_research_queries(market_context)
                if research_queries:
                    logger.info("Deep research: generated %d queries: %s",
                                len(research_queries), research_queries)
                    market_context["research_queries"] = research_queries

                    # 7d. Research the queries
                    deep_research = self._research_topics(research_queries)
                    if deep_research:
                        market_context["deep_research"] = deep_research
                        logger.info("Deep research: found %d results", len(deep_research))
            except Exception as e:
                logger.debug("Deep research generation failed (non-critical): %s", e)

        logger.info(
            "Market awareness: regime=%s (conf=%.2f), volatility=%s, "
            "sentiment=%s (%s), macro_events=%d, recs=%d, "
            "macro_data=%s, deep_research=%s",
            market_regime["regime"],
            market_regime["confidence"],
            market_regime["volatility_level"],
            sentiment_profile["classification"],
            sentiment_profile["trend"],
            len(macro_events),
            len(behavioral_recs),
            "available" if macro_indicators else "unavailable",
            "yes" if market_context.get("deep_research") else "no",
        )

        return market_context

    # ── Macro indicator fetching ──────────────────────────────────────────

    def _fetch_macro_indicators(self) -> Dict[str, Any]:
        """Fetch all macro indicators (DXY, 10Y yield, VIX, BTC dominance, total market cap).

        Results are cached for cache_ttl seconds (default 15 min).
        Each indicator fetch is wrapped in try/except so one failure doesn't block others.
        """
        now = time.time()
        if self._macro_data_cache and (now - self._macro_data_timestamp) < self.cache_ttl:
            logger.debug("Using cached macro indicators (age=%.0fs)", now - self._macro_data_timestamp)
            return self._macro_data_cache

        indicators: Dict[str, Any] = {}

        # Fetch each indicator independently
        dxy = self._fetch_dxy()
        if dxy is not None:
            indicators["dxy"] = dxy

        yields = self._fetch_us_yields()
        if yields is not None:
            indicators["us_10y_yield"] = yields

        vix = self._fetch_vix()
        if vix is not None:
            indicators["vix"] = vix

        btc_d = self._fetch_btc_dominance()
        if btc_d is not None:
            indicators["btc_dominance"] = btc_d

        total_mcap = self._fetch_total_market_cap()
        if total_mcap is not None:
            indicators["total_market_cap"] = total_mcap

        # Cache
        if indicators:
            self._macro_data_cache = indicators
            self._macro_data_timestamp = now

        logger.info("Fetched macro indicators: %s",
                     {k: v.get("value") if isinstance(v, dict) else v for k, v in indicators.items()})

        return indicators

    def _fetch_dxy(self) -> Optional[Dict[str, Any]]:
        """Fetch US Dollar Index (DXY).

        Uses exchangerate.host API. DXY is negatively correlated with BTC.
        """
        try:
            # Try exchangerate.host for USD-based index
            url = "https://api.exchangerate.host/latest"
            params = {"base": "USD", "symbols": "EUR,GBP,JPY,CAD,SEK,CHF"}
            headers = {"User-Agent": "PicsouMarketAwareness/1.0"}
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("success") and data.get("rates"):
                rates = data["rates"]
                # Approximate DXY using the 6 major components
                # DXY = 50.14348112 * EUR^(-0.576) * JPY^(0.136) * GBP^(-0.119) * CAD^(0.091) * SEK^(0.042) * CHF^(0.036)
                try:
                    import math
                    eur = rates.get("EUR", 1.0)
                    jpy = rates.get("JPY", 1.0)
                    gbp = rates.get("GBP", 1.0)
                    cad = rates.get("CAD", 1.0)
                    sek = rates.get("SEK", 1.0)
                    chf = rates.get("CHF", 1.0)
                    dxy = 50.14348112 * (1/eur)**0.576 * jpy**0.136 * (1/gbp)**0.119 * cad**0.091 * sek**0.042 * chf**0.036

                    # Determine trend/zone
                    if dxy < 95:
                        zone = "weak_dollar"
                    elif dxy < 103:
                        zone = "neutral"
                    else:
                        zone = "strong_dollar"

                    return {
                        "value": round(dxy, 2),
                        "zone": zone,
                        "description": f"DXY={dxy:.2f} ({zone})",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                except (ValueError, TypeError, ZeroDivisionError) as e:
                    logger.debug("DXY calculation error: %s", e)
        except Exception as e:
            logger.debug("DXY fetch failed: %s", e)

        # Fallback: try Yahoo Finance unofficial endpoint
        try:
            url = "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB"
            params = {"interval": "1d", "range": "5d"}
            headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            result = data.get("chart", {}).get("result", [{}])[0]
            closes = result.get("indicators", {}).get("quote", [{}])[0].get("close", [])
            # Filter None values
            closes = [c for c in closes if c is not None]
            if closes:
                dxy = closes[-1]
                if dxy < 95:
                    zone = "weak_dollar"
                elif dxy < 103:
                    zone = "neutral"
                else:
                    zone = "strong_dollar"

                # Determine trend
                trend = "stable"
                if len(closes) >= 3:
                    recent_change = closes[-1] - closes[-3]
                    if recent_change > 1:
                        trend = "rising"
                    elif recent_change < -1:
                        trend = "falling"

                return {
                    "value": round(dxy, 2),
                    "zone": zone,
                    "trend": trend,
                    "description": f"DXY={dxy:.2f} ({zone}, {trend})",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        except Exception as e:
            logger.debug("DXY Yahoo Finance fetch failed: %s", e)

        return None

    def _fetch_us_yields(self) -> Optional[Dict[str, Any]]:
        """Fetch US 10-Year Treasury Yield.

        Uses Yahoo Finance unofficial endpoint. High yields = risk-off for crypto.
        """
        try:
            # Try Yahoo Finance for ^TNX (10-Year Treasury)
            url = "https://query1.finance.yahoo.com/v8/finance/chart/^TNX"
            params = {"interval": "1d", "range": "5d"}
            headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            result = data.get("chart", {}).get("result", [{}])[0]
            closes = result.get("indicators", {}).get("quote", [{}])[0].get("close", [])
            closes = [c for c in closes if c is not None]
            if closes:
                yield_val = closes[-1]
                if yield_val < 3.5:
                    zone = "favorable"
                elif yield_val < 4.5:
                    zone = "tension"
                else:
                    zone = "danger"

                trend = "stable"
                if len(closes) >= 3:
                    recent_change = closes[-1] - closes[-3]
                    if recent_change > 0.1:
                        trend = "rising"
                    elif recent_change < -0.1:
                        trend = "falling"

                return {
                    "value": round(yield_val, 3),
                    "zone": zone,
                    "trend": trend,
                    "description": f"US 10Y Yield={yield_val:.3f}% ({zone}, {trend})",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        except Exception as e:
            logger.debug("US 10Y Yield fetch failed: %s", e)

        # Fallback: try FRED API (no key needed for basic queries)
        try:
            url = "https://fred.stlouisfed.org/graph/freddata.csv?series_id=DGS10"
            headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            lines = resp.text.strip().split("\n")
            # Find last non-empty value
            for line in reversed(lines[1:]):  # Skip header
                parts = line.split(",")
                if len(parts) >= 2 and parts[1].strip():
                    yield_val = float(parts[1].strip())
                    if yield_val < 3.5:
                        zone = "favorable"
                    elif yield_val < 4.5:
                        zone = "tension"
                    else:
                        zone = "danger"
                    return {
                        "value": round(yield_val, 3),
                        "zone": zone,
                        "trend": "unknown",
                        "description": f"US 10Y Yield={yield_val:.3f}% ({zone})",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
        except Exception as e:
            logger.debug("US 10Y Yield FRED fallback failed: %s", e)

        return None

    def _fetch_vix(self) -> Optional[Dict[str, Any]]:
        """Fetch VIX (S&P 500 Volatility Index).

        Uses Yahoo Finance. VIX > 25 = fear, VIX > 35 = panic.
        """
        try:
            url = "https://query1.finance.yahoo.com/v8/finance/chart/^VIX"
            params = {"interval": "1d", "range": "5d"}
            headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            result = data.get("chart", {}).get("result", [{}])[0]
            closes = result.get("indicators", {}).get("quote", [{}])[0].get("close", [])
            closes = [c for c in closes if c is not None]
            if closes:
                vix_val = closes[-1]
                if vix_val < 15:
                    zone = "complacency"
                elif vix_val < 25:
                    zone = "normal"
                elif vix_val < 35:
                    zone = "fear"
                else:
                    zone = "panic"

                trend = "stable"
                if len(closes) >= 3:
                    recent_change = closes[-1] - closes[-3]
                    if recent_change > 3:
                        trend = "rising"
                    elif recent_change < -3:
                        trend = "falling"

                return {
                    "value": round(vix_val, 2),
                    "zone": zone,
                    "trend": trend,
                    "description": f"VIX={vix_val:.2f} ({zone}, {trend})",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        except Exception as e:
            logger.debug("VIX fetch failed: %s", e)

        return None

    def _fetch_btc_dominance(self) -> Optional[Dict[str, Any]]:
        """Fetch Bitcoin Dominance (BTC.D) from CoinGecko.

        BTC.D rising = altcoin season ending, BTC.D falling = alt season.
        """
        try:
            url = "https://api.coingecko.com/api/v3/global"
            headers = {"User-Agent": "PicsouMarketAwareness/1.0"}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            market_cap_pct = data.get("data", {}).get("market_cap_percentage", {})
            btc_d = market_cap_pct.get("btc")

            if btc_d is not None:
                if btc_d < 40:
                    zone = "alt_season"
                elif btc_d < 55:
                    zone = "neutral"
                else:
                    zone = "btc_dominant"

                return {
                    "value": round(btc_d, 2),
                    "zone": zone,
                    "description": f"BTC.D={btc_d:.2f}% ({zone})",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        except Exception as e:
            logger.debug("BTC Dominance fetch failed: %s", e)

        return None

    def _fetch_total_market_cap(self) -> Optional[Dict[str, Any]]:
        """Fetch total crypto market cap from CoinGecko.

        Used to detect capital flows in/out of crypto.
        """
        try:
            url = "https://api.coingecko.com/api/v3/global"
            headers = {"User-Agent": "PicsouMarketAwareness/1.0"}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            total_mcap = data.get("data", {}).get("total_market_cap", {}).get("usd")
            mcap_change_24h = data.get("data", {}).get("market_cap_change_percentage_24h_usd")

            if total_mcap is not None:
                # Format in billions
                mcap_b = total_mcap / 1e9
                change_str = f"{mcap_change_24h:+.1f}%" if mcap_change_24h is not None else "N/A"

                return {
                    "value": round(total_mcap, 0),
                    "value_billion": round(mcap_b, 1),
                    "change_24h_pct": round(mcap_change_24h, 2) if mcap_change_24h is not None else None,
                    "description": f"Total Crypto Mcap=${mcap_b:.1f}B (24h: {change_str})",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        except Exception as e:
            logger.debug("Total market cap fetch failed: %s", e)

        return None

    # ── Deep research methods ─────────────────────────────────────────────

    # Map of common research queries to Wikipedia article titles
    _WIKI_TOPIC_MAP: Dict[str, str] = {
        "strong us dollar impact cryptocurrency": "United States dollar",
        "dxy index above 103": "U.S. Dollar Index",
        "weak us dollar cryptocurrency rally": "United States dollar",
        "vix volatility spike stock market crash": "VIX",
        "high volatility crypto selloff": "Volatility (finance)",
        "us treasury yields rising crypto impact": "United States Treasury security",
        "interest rates cryptocurrency": "Interest rate",
        "federal reserve monetary policy crypto": "Federal Reserve",
        "sec cryptocurrency regulation 2026": "Securities and Exchange Commission",
        "bitcoin etf inflows outflows impact": "Bitcoin ETF",
        "geopolitical risk cryptocurrency safe haven": "Safe asset",
        "crypto exchange hack market impact": "Cryptocurrency and crime",
        "bitcoin dominance rising altcoin season end": "Bitcoin",
        "altcoin season bitcoin dominance falling": "Altcoin",
        "crypto market extreme_fear what to do": "Market sentiment",
        "crypto market extreme_greed what to do": "Market sentiment",
    }

    def _research_macro_topic(self, topic: str) -> Optional[Dict[str, Any]]:
        """Research a macro topic via Wikipedia API for background context.

        First tries a direct article lookup using a known topic map,
        then falls back to Wikipedia search API to find relevant articles,
        and finally fetches the summary of the best-matching article.
        """
        if not self.deep_research_wikipedia_enabled:
            return None

        # Try known topic map first
        wiki_title = self._WIKI_TOPIC_MAP.get(topic.lower().strip())
        titles_to_try = []
        if wiki_title:
            titles_to_try.append(wiki_title)

        # Also try converting the query to a Wikipedia-style title
        # (replace spaces with underscores, capitalize)
        simplified = topic.replace(" ", "_").strip()
        titles_to_try.append(simplified)

        headers = {"User-Agent": "PicsouBot/1.0 (research)", "Accept": "application/json"}

        # Try direct page lookup for known titles
        for title in titles_to_try:
            try:
                url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    extract = data.get("extract", "")
                    if extract and len(extract) > 50:
                        return {
                            "topic": topic,
                            "summary": extract[:500],
                            "source": "wikipedia",
                            "relevance": "background",
                        }
            except Exception as e:
                logger.debug("Wikipedia direct lookup for '%s' failed: %s", title, e)

        # Fallback: use Wikipedia search API to find relevant articles
        try:
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": topic,
                "format": "json",
                "srlimit": 1,
            }
            resp = requests.get(search_url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                search_results = data.get("query", {}).get("search", [])
                if search_results:
                    best_title = search_results[0]["title"]
                    # Now fetch the summary for this article
                    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(best_title)}"
                    resp2 = requests.get(summary_url, headers=headers, timeout=10)
                    if resp2.status_code == 200:
                        summary_data = resp2.json()
                        extract = summary_data.get("extract", "")
                        if extract and len(extract) > 50:
                            return {
                                "topic": topic,
                                "summary": extract[:500],
                                "source": "wikipedia",
                                "relevance": "background",
                            }
        except Exception as e:
            logger.debug("Wikipedia search for '%s' failed: %s", topic, e)

        return None

    def _get_fear_greed_trend(self) -> Dict[str, Any]:
        """Get Fear & Greed Index trend over the last 30 days.

        Fetches historical F&G data from alternative.me and computes
        trend direction (improving/stable/deteriorating).
        Results are cached for cache_ttl seconds.
        """
        now = time.time()
        if self._fg_trend_cache and (now - self._fg_trend_timestamp) < self.cache_ttl:
            logger.debug("Using cached F&G trend (age=%.0fs)", now - self._fg_trend_timestamp)
            return self._fg_trend_cache

        try:
            url = "https://api.alternative.me/fng/?limit=30"
            headers = {"User-Agent": "PicsouBot/1.0 (research)"}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()["data"]
                values = [int(d["value"]) for d in data]
                # Reverse to get chronological order (API returns newest first)
                values.reverse()
                current = values[-1]
                week_ago = values[-7] if len(values) >= 7 else values[0]
                month_ago = values[0]

                trend = "stable"
                if current - week_ago > 10:
                    trend = "improving"
                elif week_ago - current > 10:
                    trend = "deteriorating"

                # Determine sentiment classification for current and past values
                def _fg_classify(val: int) -> str:
                    if val <= 25:
                        return "Extreme Fear"
                    elif val <= 45:
                        return "Fear"
                    elif val <= 55:
                        return "Neutral"
                    elif val <= 75:
                        return "Greed"
                    else:
                        return "Extreme Greed"

                result = {
                    "current": current,
                    "current_classification": _fg_classify(current),
                    "week_ago": week_ago,
                    "week_ago_classification": _fg_classify(week_ago),
                    "month_ago": month_ago,
                    "month_ago_classification": _fg_classify(month_ago),
                    "change_week": current - week_ago,
                    "change_month": current - month_ago,
                    "trend": trend,
                    "values": values,
                }

                # Cache
                self._fg_trend_cache = result
                self._fg_trend_timestamp = now
                return result
        except Exception as e:
            logger.debug("F&G trend fetch failed: %s", e)
        return {}

    def _get_coingecko_context(self) -> Dict[str, Any]:
        """Get CoinGecko global market data + trending coins.

        Provides broader market context including BTC/ETH dominance,
        total market cap, and which coins are trending (useful for
        detecting meme coin mania, etc.).
        Results are cached for cache_ttl seconds.
        """
        now = time.time()
        if self._coingecko_context_cache and (now - self._coingecko_context_timestamp) < self.cache_ttl:
            logger.debug("Using cached CoinGecko context (age=%.0fs)", now - self._coingecko_context_timestamp)
            return self._coingecko_context_cache

        result: Dict[str, Any] = {"global": {}, "trending": []}

        try:
            # Global data
            resp = requests.get("https://api.coingecko.com/api/v3/global",
                                headers={"User-Agent": "PicsouBot/1.0 (research)"},
                                timeout=10)
            if resp.status_code == 200:
                data = resp.json()["data"]
                result["global"] = {
                    "btc_dominance": data["market_cap_percentage"]["btc"],
                    "eth_dominance": data["market_cap_percentage"]["eth"],
                    "total_market_cap_usd": data["total_market_cap"]["usd"],
                    "total_volume_usd": data["total_volume"]["usd"],
                    "market_cap_change_24h_pct": data["market_cap_change_percentage_24h_usd"],
                }
        except Exception as e:
            logger.debug("CoinGecko global data fetch failed: %s", e)

        try:
            # Trending coins
            resp = requests.get("https://api.coingecko.com/api/v3/search/trending",
                                headers={"User-Agent": "PicsouBot/1.0 (research)"},
                                timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for coin in data.get("coins", [])[:7]:
                    item = coin.get("item", {})
                    result["trending"].append({
                        "name": item.get("name", ""),
                        "symbol": item.get("symbol", ""),
                        "market_cap_rank": item.get("market_cap_rank", 0),
                        "price_btc": item.get("price_btc", 0),
                    })
        except Exception as e:
            logger.debug("CoinGecko trending fetch failed: %s", e)

        # Cache if we got any data
        if result["global"] or result["trending"]:
            self._coingecko_context_cache = result
            self._coingecko_context_timestamp = now
        return result

    def _generate_research_queries(self, market_context: Dict[str, Any]) -> List[str]:
        """Based on current market conditions, decide what to research.

        Analyzes macro indicators, sentiment, and RSS events to formulate
        targeted search queries for Wikipedia and other sources.
        Returns a deduplicated list of max 5 queries.
        """
        queries: List[str] = []

        # Check macro indicators
        indicators = market_context.get("macro_indicators", {})
        dxy = indicators.get("dxy", {}).get("value", 0)
        vix = indicators.get("vix", {}).get("value", 0)
        yield_10y = indicators.get("us_10y_yield", {}).get("value", 0)

        # DXY above 103 = strong dollar, research impact
        if dxy > 103:
            queries.append("Strong US dollar impact cryptocurrency")
            queries.append("DXY index above 103")
        elif dxy < 95:
            queries.append("Weak US dollar cryptocurrency rally")

        # VIX above 25 = fear in markets
        if vix > 25:
            queries.append("VIX volatility spike stock market crash")
            queries.append("High volatility crypto selloff")

        # Yields above 4.5%
        if yield_10y > 4.5:
            queries.append("US Treasury yields rising crypto impact")
            queries.append("Interest rates cryptocurrency")

        # Check sentiment
        sentiment = market_context.get("sentiment_profile", {})
        classification = sentiment.get("classification", "").lower()
        if "extreme" in classification:
            queries.append(f"Crypto market {classification} what to do")

        # Check macro events from RSS
        events = market_context.get("macro_events", [])
        for event in events[:3]:  # Top 3 events
            keyword = event.get("keyword", "")
            if keyword in ["fed", "fomc", "rate hike", "rate cut", "powell"]:
                queries.append("Federal Reserve monetary policy crypto")
                queries.append("2026 Federal Reserve interest rate decision")
            elif keyword in ["sec", "regulation", "ban", "crackdown"]:
                queries.append("SEC cryptocurrency regulation 2026")
            elif keyword in ["etf", "etf approval", "spot etf"]:
                queries.append("Bitcoin ETF inflows outflows impact")
            elif keyword in ["war", "geopolitical", "sanctions"]:
                queries.append("Geopolitical risk cryptocurrency safe haven")
            elif keyword in ["hack", "exploit", "rug pull"]:
                queries.append("Crypto exchange hack market impact")

        # Check BTC dominance
        btc_d = indicators.get("btc_dominance", {}).get("value", 0)
        if btc_d > 55:
            queries.append("Bitcoin dominance rising altcoin season end")
        elif btc_d < 42:
            queries.append("Altcoin season Bitcoin dominance falling")

        # Deduplicate and limit
        seen: set = set()
        unique: List[str] = []
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique.append(q)

        return unique[:self.deep_research_max_queries]

    def _research_topics(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Research multiple topics, trying Wikipedia first, then RSS headlines.

        For each query, attempts a Wikipedia search. If that fails,
        searches existing RSS headlines for keyword matches.
        """
        results: List[Dict[str, Any]] = []

        for query in queries:
            # Try Wikipedia first
            wiki_result = None
            if self.deep_research_wikipedia_enabled:
                wiki_result = self._research_macro_topic(query)

            if wiki_result:
                results.append(wiki_result)
                continue

            # Fallback: search in existing RSS headlines
            if self.deep_research_rss_search:
                try:
                    headlines = self.get_rss_headlines(max_items=15)
                    query_words = query.lower().split()
                    for headline in headlines:
                        title = headline.get("title", "").lower()
                        desc = headline.get("description", "").lower()
                        match_count = sum(
                            1 for w in query_words
                            if w in title or w in desc
                        )
                        if match_count >= 2:
                            results.append({
                                "topic": query,
                                "summary": f"[{headline['source']}] {headline['title']}: {headline.get('description', '')}",
                                "source": headline["source"],
                                "relevance": "headline_match",
                            })
                            break
                except Exception as e:
                    logger.debug("RSS headline search for '%s' failed: %s", query, e)

        return results

    # ── LLM-based macro reasoning ─────────────────────────────────────────

    def reason_about_macro(self, market_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Use the LLM to reason about macro conditions and produce an assessment.

        Gathers all macro data (indicators + events + sentiment) and sends a short
        prompt to the LLM, which returns a structured macro_assessment.

        Args:
            market_context: The full market context from analyze().

        Returns:
            Dict with macro_assessment fields, or None on failure.
            Cached for cache_ttl seconds.
        """
        now = time.time()
        if self._macro_assessment_cache and (now - self._macro_assessment_timestamp) < self.cache_ttl:
            logger.debug("Using cached macro assessment (age=%.0fs)", now - self._macro_assessment_timestamp)
            return self._macro_assessment_cache

        # Gather data
        indicators = market_context.get("macro_indicators", {})
        sentiment_profile = market_context.get("sentiment_profile", {})
        macro_events = market_context.get("macro_events", [])
        regime = market_context.get("market_regime", {})

        # Build indicator summary
        dxy = indicators.get("dxy", {})
        yields = indicators.get("us_10y_yield", {})
        vix = indicators.get("vix", {})
        btc_d = indicators.get("btc_dominance", {})
        total_mcap = indicators.get("total_market_cap", {})

        dxy_val = dxy.get("value", "N/A") if dxy else "N/A"
        dxy_trend = dxy.get("trend", "unknown") if dxy else "unknown"
        yield_val = yields.get("value", "N/A") if yields else "N/A"
        yield_trend = yields.get("trend", "unknown") if yields else "unknown"
        vix_val = vix.get("value", "N/A") if vix else "N/A"
        vix_trend = vix.get("trend", "unknown") if vix else "unknown"
        btc_d_val = btc_d.get("value", "N/A") if btc_d else "N/A"
        btc_d_zone = btc_d.get("zone", "unknown") if btc_d else "unknown"

        fg = sentiment_profile.get("score", 50)
        fg_class = sentiment_profile.get("classification", "neutral")
        fg_trend = sentiment_profile.get("trend", "stable")

        # Events summary (top 3)
        event_summaries = []
        for e in macro_events[:3]:
            event_summaries.append(f"{e.get('keyword', '?')} ({e.get('impact', '?')}, mag {e.get('magnitude', 1)})")
        events_str = "; ".join(event_summaries) if event_summaries else "none detected"

        # Regime info
        regime_name = regime.get("regime", "unknown")
        regime_conf = regime.get("confidence", 0.5)

        # ── Deep research context ──
        fg_trend_data = market_context.get("fg_trend", {})
        fg_trend_section = ""
        if fg_trend_data:
            current_cls = fg_trend_data.get("current_classification", "")
            week_cls = fg_trend_data.get("week_ago_classification", "")
            month_cls = fg_trend_data.get("month_ago_classification", "")
            fg_trend_section = (
                f"\nSENTIMENT TREND (30 days):\n"
                f"- Current: {current_cls} ({fg_trend_data.get('current', 'N/A')})\n"
                f"- Week ago: {week_cls} ({fg_trend_data.get('week_ago', 'N/A')}) "
                f"→ {fg_trend_data.get('change_week', 0):+d} points ({fg_trend_data.get('trend', 'unknown')})\n"
                f"- Month ago: {month_cls} ({fg_trend_data.get('month_ago', 'N/A')}) "
                f"→ {fg_trend_data.get('change_month', 0):+d} points\n"
                f"- Trend: {fg_trend_data.get('trend', 'unknown')}"
            )

        cg_context = market_context.get("coingecko_context", {})
        cg_section = ""
        if cg_context:
            cg_global = cg_context.get("global", {})
            cg_trending = cg_context.get("trending", [])
            if cg_global:
                btc_dom = cg_global.get("btc_dominance", "N/A")
                eth_dom = cg_global.get("eth_dominance", "N/A")
                mcap = cg_global.get("total_market_cap_usd", 0)
                mcap_b = mcap / 1e9 if isinstance(mcap, (int, float)) and mcap > 0 else "N/A"
                chg_24h = cg_global.get("market_cap_change_24h_pct", "N/A")
                chg_str = f"{chg_24h:+.1f}%" if isinstance(chg_24h, (int, float)) else str(chg_24h)

                btc_dominance_note = ""
                if isinstance(btc_dom, (int, float)):
                    if btc_dom > 55:
                        btc_dominance_note = " (BTC dominant, alt season not yet)"
                    elif btc_dom < 42:
                        btc_dominance_note = " (alt season in progress)"
                    else:
                        btc_dominance_note = " (balanced)"

                cg_section = (
                    f"\nMARKET CONTEXT (CoinGecko):\n"
                    f"- BTC Dominance: {btc_dom}%{btc_dominance_note}\n"
                    f"- ETH Dominance: {eth_dom}%\n"
                    f"- Total Market Cap: ${mcap_b}B\n"
                    f"- 24h Change: {chg_str}"
                )

            if cg_trending:
                trending_names = [f"{c['name']} ({c['symbol']})" for c in cg_trending[:5] if c.get("name")]
                meme_keywords = ["pepe", "bonk", "wif", "doge", "shib", "floki", "meme", "pepe"]
                trending_lower = " ".join(trending_names).lower()
                meme_trending = any(kw in trending_lower for kw in meme_keywords)
                meme_note = " (⚠️ meme coins trending = risk-on speculation)" if meme_trending else ""
                cg_section += f"\n- Trending: [{', '.join(trending_names)}]{meme_note}"

        # Deep research findings
        deep_research = market_context.get("deep_research", [])
        research_queries = market_context.get("research_queries", [])
        research_section = ""
        if research_queries:
            research_section += "\nRESEARCH QUERIES GENERATED:"
            for q in research_queries:
                trigger = ""
                indicators_local = market_context.get("macro_indicators", {})
                if "dollar" in q.lower() or "dxy" in q.lower():
                    dxy_v = indicators_local.get("dxy", {}).get("value", 0)
                    trigger = f" (triggered by DXY={dxy_v})"
                elif "vix" in q.lower() or "volatility" in q.lower():
                    vix_v = indicators_local.get("vix", {}).get("value", 0)
                    trigger = f" (triggered by VIX={vix_v})"
                elif "yield" in q.lower() or "interest rate" in q.lower():
                    y = indicators_local.get("us_10y_yield", {}).get("value", 0)
                    trigger = f" (triggered by 10Y yield={y}%)"
                elif "fed" in q.lower() or "reserve" in q.lower():
                    trigger = " (triggered by Fed keyword in headlines)"
                research_section += f"\n- \"{q}\"{trigger}"

        if deep_research:
            research_section += "\n\nDEEP RESEARCH FINDINGS:"
            for r in deep_research:
                source = r.get("source", "unknown")
                topic = r.get("topic", "")
                summary = r.get("summary", "")[:200]
                research_section += f"\n- [{source}] {topic}: {summary}"

        # Build prompt
        prompt = (
            "You are a macro economist analyzing crypto market conditions. Based on the data below, provide a brief assessment.\n"
            f"\nDATA:\n"
            f"- Market regime: {regime_name} (confidence: {regime_conf:.0%})\n"
            f"- DXY: {dxy_val} ({dxy_trend})\n"
            f"- US 10Y Yield: {yield_val}% ({yield_trend})\n"
            f"- VIX: {vix_val} ({vix_trend})\n"
            f"- BTC Dominance: {btc_d_val}% ({btc_d_zone})\n"
            f"- Fear & Greed: {fg}/{100} ({fg_class}, trend: {fg_trend})\n"
            f"- Total crypto mcap: {total_mcap.get('description', 'N/A') if total_mcap else 'N/A'}\n"
            f"- Recent events: {events_str}\n"
            f"{fg_trend_section}\n"
            f"{cg_section}\n"
            f"{research_section}\n"
            f"\nRespond in JSON only:\n"
            '{"regime_confirmation": "bull|bear|range|volatile|crunch", '
            '"risk_outlook": "favorable|neutral|unfavorable|dangerous", '
            '"time_horizon": "hours|days|weeks", '
            '"reasoning": "2-3 sentences explaining why", '
            '"actionable_advice": "1 sentence concrete trading advice"}'
        )

        # Call LLM
        try:
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
                        "content": "You are a macro economist. Analyze market data and respond ONLY with valid JSON. No markdown, no prose, just the JSON object."
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.llm_temperature,
                "max_tokens": 500,
            }

            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                # Try reasoning field (some models)
                content = data.get("choices", [{}])[0].get("message", {}).get("reasoning", "")

            # Strip markdown code fences if present
            raw = content.strip()
            if raw.startswith("```"):
                first_newline = raw.find("\n")
                if first_newline != -1:
                    raw = raw[first_newline + 1:]
                if raw.endswith("```"):
                    raw = raw[:-3].rstrip()

            # Parse JSON
            assessment = json.loads(raw)

            # Validate required fields
            valid_fields = {"regime_confirmation", "risk_outlook", "time_horizon", "reasoning", "actionable_advice"}
            if not valid_fields.intersection(assessment.keys()):
                logger.warning("Macro assessment missing required fields: %s", assessment)
                return None

            # Cache
            self._macro_assessment_cache = assessment
            self._macro_assessment_timestamp = now

            logger.info("Macro assessment: regime=%s, risk=%s, horizon=%s, advice=%s",
                        assessment.get("regime_confirmation"),
                        assessment.get("risk_outlook"),
                        assessment.get("time_horizon"),
                        assessment.get("actionable_advice", "")[:80])

            return assessment

        except json.JSONDecodeError as e:
            logger.warning("Macro assessment JSON parse error: %s", e)
        except requests.exceptions.Timeout:
            logger.warning("Macro assessment LLM call timed out")
        except requests.exceptions.ConnectionError:
            logger.warning("Macro assessment LLM connection error")
        except Exception as e:
            logger.warning("Macro assessment failed: %s", e)

        return None

    # ── Regime detection ──────────────────────────────────────────────────

    def _detect_regime(
        self,
        market_data: Dict[str, Dict[str, Any]],
        tech_indicators: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Detect the current market regime based on price trends, volatility,
        volume, and technical indicators.

        Returns one of: bull, bear, range, volatile, crunch
        """
        trend_signals: List[float] = []  # positive = bullish, negative = bearish
        volatility_signals: List[float] = []  # higher = more volatile
        volume_signals: List[float] = []  # higher = more volume

        for key, data in market_data.items():
            candles = data.get("candles", [])
            if not candles or len(candles) < 20:
                continue

            closes = [c["close"] for c in candles]
            current_price = closes[-1]

            # ── Trend: EMA 50 vs EMA 200 (or shorter windows if not enough data)
            if len(closes) >= 200:
                ema50 = _ema(closes, 50)
                ema200 = _ema(closes, 200)
                ema50_val = ema50[-1] if ema50 else current_price
                ema200_val = ema200[-1] if ema200 else current_price
            elif len(closes) >= 50:
                ema50 = _ema(closes, 50)
                ema200 = None
                ema50_val = ema50[-1] if ema50 else current_price
                ema200_val = None
            else:
                ema50_val = None
                ema200_val = None

            # EMA crossover trend
            if ema50_val is not None and ema200_val is not None:
                ema_sep = (ema50_val - ema200_val) / ema200_val if ema200_val > 0 else 0
                trend_signals.append(max(-1.0, min(1.0, ema_sep * 10)))  # Scale and clamp

            # ── Trend: Price slope over last 24h-ish candles
            if len(closes) >= 10:
                recent = closes[-10:]
                slope = (recent[-1] - recent[0]) / recent[0] if recent[0] > 0 else 0
                trend_signals.append(max(-1.0, min(1.0, slope * 5)))

            # 7-day-ish slope
            if len(closes) >= 30:
                weekly = closes[-30:]
                slope_w = (weekly[-1] - weekly[0]) / weekly[0] if weekly[0] > 0 else 0
                trend_signals.append(max(-1.0, min(1.0, slope_w * 3)))

            # ── Volatility: ATR-based
            atr_result = calc_atr(candles, period=14)
            atr_val = atr_result.get("value", 0)
            if atr_val and atr_val == atr_val and current_price > 0:
                atr_pct = atr_val / current_price
                # Normal crypto ATR is 1-5%; >5% is volatile; >10% is extreme
                volatility_signals.append(min(1.0, atr_pct * 10))

            # ── Volatility: Bollinger Bandwidth
            bb_result = calc_bollinger_bands(closes, period=20, num_std=2.0)
            bw = bb_result.get("bandwidth", 0)
            if bw and bw == bw and current_price > 0:
                bw_pct = bw / current_price
                # Normal BB bandwidth: 3-8%; >8% is volatile; >15% is extreme
                volatility_signals.append(min(1.0, bw_pct * 7))

            # ── Volume trend
            vol_result = calc_volume_profile(candles)
            vol_ratio = vol_result.get("ratio", 1.0)
            if vol_ratio and vol_ratio == vol_ratio:
                volume_signals.append(min(2.0, vol_ratio))

            # ── Use tech_indicators if available for richer analysis
            symbol_key = data.get("base", key.split(":")[-1] if ":" in key else key)
            if tech_indicators and symbol_key in tech_indicators:
                ti = tech_indicators[symbol_key]
                rsi_val = ti.get("rsi", {}).get("value", 50)
                if rsi_val == rsi_val:
                    # Extreme RSI signals
                    if rsi_val >= 75:
                        trend_signals.append(-0.3)  # Overbought → bearish signal
                    elif rsi_val <= 25:
                        trend_signals.append(0.3)  # Oversold → bullish reversal signal

        # ── Aggregate signals ──
        avg_trend = sum(trend_signals) / len(trend_signals) if trend_signals else 0.0
        avg_volatility = sum(volatility_signals) / len(volatility_signals) if volatility_signals else 0.3
        avg_volume = sum(volume_signals) / len(volume_signals) if volume_signals else 1.0

        # ── Determine regime ──
        regime = "range"
        confidence = 0.5
        description = "Market is in a ranging/consolidation phase."

        # Crunch detection: extreme volatility + strong negative trend
        if avg_volatility > 0.7 and avg_trend < -0.4:
            regime = "crunch"
            confidence = min(0.95, 0.6 + avg_volatility * 0.3 + abs(avg_trend) * 0.1)
            description = "CRASH / extreme sell-off in progress! Very high volatility with strong downward momentum."
        # Bear market: negative trend + moderate/high volatility
        elif avg_trend < -0.2:
            regime = "bear"
            confidence = min(0.9, 0.5 + abs(avg_trend))
            if avg_volatility > 0.5:
                description = "Bear market with elevated volatility. Downward trend confirmed."
            else:
                description = "Bear market with moderate volatility. Prices declining steadily."
        # Bull market: positive trend + moderate volatility
        elif avg_trend > 0.2:
            regime = "bull"
            confidence = min(0.9, 0.5 + avg_trend)
            if avg_volatility > 0.5:
                description = "Bull market with high volatility. Strong upward momentum but watch for corrections."
            else:
                description = "Bull market with moderate volatility. Steady upward trend."
        # High volatility without clear trend = volatile/choppy
        elif avg_volatility > 0.5:
            regime = "volatile"
            confidence = min(0.8, 0.4 + avg_volatility * 0.4)
            description = "Volatile/choppy market without clear trend. High risk of whipsaws."
        # Otherwise = range
        else:
            regime = "range"
            confidence = max(0.3, 0.5 - abs(avg_trend) - avg_volatility * 0.2)
            description = "Market is ranging/consolidating. Low volatility, no clear trend direction."

        # Volume confirmation: high volume = more confidence in the regime
        if avg_volume > 1.3:
            confidence = min(0.95, confidence + 0.05)
        elif avg_volume < 0.7:
            confidence = max(0.2, confidence - 0.1)

        # Volatility level classification
        if avg_volatility > 0.7:
            volatility_level = "extreme"
        elif avg_volatility > 0.5:
            volatility_level = "high"
        elif avg_volatility > 0.3:
            volatility_level = "moderate"
        else:
            volatility_level = "low"

        # Trend strength
        trend_strength = min(1.0, abs(avg_trend))

        return {
            "regime": regime,
            "confidence": round(confidence, 3),
            "trend_strength": round(trend_strength, 3),
            "trend_direction": round(avg_trend, 3),
            "volatility_level": volatility_level,
            "volatility_score": round(avg_volatility, 3),
            "volume_ratio": round(avg_volume, 3),
            "description": description,
        }

    # ── Macro event scanning ─────────────────────────────────────────────

    def _scan_macro_events(
        self, sentiment: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Scan for macro events from headlines and web search.

        Uses existing crypto headlines + targeted DuckDuckGo search.
        Results are cached to avoid excessive API calls.
        """
        # Check cache
        now = time.time()
        if self._macro_events_cache and (now - self._macro_events_timestamp) < self.cache_ttl:
            logger.debug("Using cached macro events (age=%.0fs)", now - self._macro_events_timestamp)
            return self._macro_events_cache

        events: List[Dict[str, Any]] = []
        seen_keywords: set = set()

        # ── Source 1: Existing crypto headlines ──
        headlines = sentiment.get("headlines", [])
        for headline in headlines:
            event = self._extract_event_from_text(headline, source="crypto_headlines")
            if event and event["keyword"] not in seen_keywords:
                events.append(event)
                seen_keywords.add(event["keyword"])

        # ── Source 2: Targeted DuckDuckGo search for macro events ──
        ddg_events = self._fetch_macro_news()
        for event in ddg_events:
            if event["keyword"] not in seen_keywords:
                events.append(event)
                seen_keywords.add(event["keyword"])

        # Sort by magnitude (highest impact first)
        events.sort(key=lambda e: e.get("magnitude", 1), reverse=True)

        # Cache
        self._macro_events_cache = events[:10]  # Keep top 10
        self._macro_events_timestamp = now

        return self._macro_events_cache

    def _extract_event_from_text(
        self, text: str, source: str = "unknown"
    ) -> Optional[Dict[str, Any]]:
        """Extract a macro event from a text snippet based on keyword matching.

        Returns an event dict or None if no macro keyword found.
        """
        text_lower = text.lower()

        best_match: Optional[Dict[str, Any]] = None
        best_magnitude = 0

        for keyword, meta in MACRO_KEYWORDS.items():
            if keyword in text_lower:
                if meta["magnitude"] > best_magnitude:
                    best_magnitude = meta["magnitude"]
                    best_match = {
                        "event": text[:150],
                        "source": source,
                        "impact": meta["impact"],
                        "magnitude": meta["magnitude"],
                        "category": meta["category"],
                        "keyword": keyword,
                        "description": f"{meta['impact'].upper()} {meta['category']} event: '{keyword}' detected",
                    }

        return best_match

    def _fetch_macro_news(self) -> List[Dict[str, Any]]:
        """Fetch macro events from RSS feeds + crypto headlines.

        Replaces the old DuckDuckGo-based approach with RSS feeds that
        actually return real-time news data.
        """
        events: List[Dict[str, Any]] = []
        seen_keywords: set = set()

        # Source 1: Crypto news (RSS)
        crypto_headlines = self._fetch_crypto_news()
        for headline in crypto_headlines:
            event = self._extract_event_from_text(
                f"{headline['title']} {headline['description']}",
                source=headline['source'],
            )
            if event and event["keyword"] not in seen_keywords:
                events.append(event)
                seen_keywords.add(event["keyword"])

        # Source 2: Macro news (RSS)
        macro_headlines = self._fetch_macro_news_rss()
        for headline in macro_headlines:
            event = self._extract_event_from_text(
                f"{headline['title']} {headline['description']}",
                source=headline['source'],
            )
            if event and event["keyword"] not in seen_keywords:
                events.append(event)
                seen_keywords.add(event["keyword"])

        # Source 3: Fed schedule (RSS + DDG fallback)
        fed_events = self._fetch_fed_schedule()
        for event in fed_events:
            if event["keyword"] not in seen_keywords:
                events.append(event)
                seen_keywords.add(event["keyword"])

        # Sort by magnitude
        events.sort(key=lambda e: e.get("magnitude", 1), reverse=True)
        return events[:10]

    # ── RSS headline fetching ─────────────────────────────────────────────

    def _fetch_rss_headlines(self, source_name: str, url: str, max_items: int = 10) -> List[Dict[str, Any]]:
        """Fetch and parse headlines from an RSS feed.

        Args:
            source_name: Human-readable name for the source (e.g. 'CoinTelegraph').
            url: RSS feed URL.
            max_items: Maximum number of items to return.

        Returns:
            List of dicts with keys: title, description, source, link.
        """
        headlines: List[Dict[str, Any]] = []
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
                "Accept": "application/rss+xml, application/xml, text/xml, application/atom+xml, */*",
            }
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()

            # Parse XML
            root = ET.fromstring(resp.content)

            # Handle both RSS and Atom feeds
            # RSS 2.0: <channel><item>...
            # Atom: <entry>...
            items = root.findall(".//item")  # RSS 2.0
            if not items:
                items = root.findall(".//{http://www.w3.org/2005/Atom}entry")  # Atom

            for item in items[:max_items]:
                # RSS 2.0 item
                title_el = item.find("title")
                desc_el = item.find("description")
                link_el = item.find("link")

                # Atom entry
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                if title_el is None:
                    title_el = item.find("atom:title", ns)
                if desc_el is None:
                    desc_el = item.find("atom:summary", ns) or item.find("atom:content", ns)
                if link_el is None:
                    link_el = item.find("atom:link", ns)
                    if link_el is not None:
                        link_text = link_el.get("href", "")
                    else:
                        link_text = ""
                else:
                    link_text = link_el.text or ""

                title = title_el.text if title_el is not None and title_el.text else ""
                desc = desc_el.text if desc_el is not None and desc_el.text else ""

                # Clean HTML tags from description
                title = re.sub(r"<[^>]+>", "", title).strip()
                desc = re.sub(r"<[^>]+>", "", desc).strip()

                if title:
                    headlines.append({
                        "title": title[:300],
                        "description": desc[:500],
                        "source": source_name,
                        "link": link_text[:500],
                    })

            logger.debug("RSS %s: fetched %d headlines", source_name, len(headlines))

        except requests.exceptions.Timeout:
            logger.warning("RSS fetch timed out for %s (%s)", source_name, url)
        except requests.exceptions.ConnectionError:
            logger.warning("RSS connection error for %s (%s)", source_name, url)
        except Exception as e:
            logger.warning("RSS fetch failed for %s (%s): %s", source_name, url, e)

        return headlines

    def _fetch_crypto_news(self) -> List[Dict[str, Any]]:
        """Fetch crypto news headlines from RSS feeds.

        Fetches CoinTelegraph, CoinDesk, Decrypt, and Yahoo Finance Crypto.
        Returns the 15 most recent headlines.
        """
        # Check RSS cache
        now = time.time()
        if self._rss_headlines_cache and (now - self._rss_headlines_timestamp) < self.cache_ttl:
            crypto_cached = [h for h in self._rss_headlines_cache if h.get("_category") == "crypto"]
            if crypto_cached:
                logger.debug("Using cached crypto RSS headlines (age=%.0fs)", now - self._rss_headlines_timestamp)
                return crypto_cached[:15]

        all_headlines: List[Dict[str, Any]] = []
        for source in RSS_SOURCES["crypto"]:
            headlines = self._fetch_rss_headlines(source["name"], source["url"], max_items=5)
            for h in headlines:
                h["_category"] = "crypto"
            all_headlines.extend(headlines)

        # Deduplicate by title similarity
        seen_titles: set = set()
        unique: List[Dict[str, Any]] = []
        for h in all_headlines:
            title_lower = h["title"].lower()[:60]
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique.append(h)

        result = unique[:15]
        logger.info("Crypto RSS headlines: %d fetched (total raw=%d)", len(result), len(all_headlines))

        # Update cache
        self._update_rss_cache("crypto", result)
        return result

    def _fetch_macro_news_rss(self) -> List[Dict[str, Any]]:
        """Fetch macro news headlines from RSS feeds.

        Fetches Yahoo Finance Market News and Federal Reserve feeds.
        Returns the 10 most recent headlines.
        """
        # Check RSS cache
        now = time.time()
        if self._rss_headlines_cache and (now - self._rss_headlines_timestamp) < self.cache_ttl:
            macro_cached = [h for h in self._rss_headlines_cache if h.get("_category") == "macro"]
            if macro_cached:
                logger.debug("Using cached macro RSS headlines (age=%.0fs)", now - self._rss_headlines_timestamp)
                return macro_cached[:10]

        all_headlines: List[Dict[str, Any]] = []
        for source in RSS_SOURCES["macro"]:
            headlines = self._fetch_rss_headlines(source["name"], source["url"], max_items=6)
            for h in headlines:
                h["_category"] = "macro"
            all_headlines.extend(headlines)

        # Deduplicate by title similarity
        seen_titles: set = set()
        unique: List[Dict[str, Any]] = []
        for h in all_headlines:
            title_lower = h["title"].lower()[:60]
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique.append(h)

        result = unique[:10]
        logger.info("Macro RSS headlines: %d fetched (total raw=%d)", len(result), len(all_headlines))

        # Update cache
        self._update_rss_cache("macro", result)
        return result

    def _fetch_fed_schedule(self) -> List[Dict[str, Any]]:
        """Fetch FOMC/Fed schedule announcements from RSS.

        Uses the Federal Reserve RSS feed and falls back to DDG search
        for 'fomc schedule 2026'. Returns list of macro event dicts.
        """
        events: List[Dict[str, Any]] = []
        seen_keywords: set = set()

        # Source 1: Federal Reserve RSS
        try:
            headlines = self._fetch_rss_headlines("Federal Reserve", "https://www.federalreserve.gov/feeds/press_monetary.xml", max_items=10)
            for h in headlines:
                event = self._extract_event_from_text(
                    f"{h['title']} {h['description']}", source="Federal Reserve RSS"
                )
                if event and event["keyword"] not in seen_keywords:
                    events.append(event)
                    seen_keywords.add(event["keyword"])
        except Exception as e:
            logger.debug("Fed RSS fetch failed: %s", e)

        # Source 2: DDG search fallback (silent failure acceptable)
        try:
            ddg_snippets = self._ddg_search("fomc schedule 2026")
            for snippet in ddg_snippets[:3]:
                event = self._extract_event_from_text(snippet, source="duckduckgo:fomc_schedule")
                if event and event["keyword"] not in seen_keywords:
                    events.append(event)
                    seen_keywords.add(event["keyword"])
        except Exception as e:
            logger.debug("DDG Fed schedule fallback failed: %s", e)

        return events

    def _update_rss_cache(self, category: str, headlines: List[Dict[str, Any]]) -> None:
        """Update the RSS headlines cache with new data for a given category."""
        # Remove old entries for this category and merge new ones
        other_cached = [h for h in self._rss_headlines_cache if h.get("_category") != category]
        self._rss_headlines_cache = other_cached + headlines
        self._rss_headlines_timestamp = time.time()

    def get_rss_headlines(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """Public method to get recent RSS headlines for prompt injection.

        Returns cached headlines or fetches fresh ones.
        """
        now = time.time()
        if self._rss_headlines_cache and (now - self._rss_headlines_timestamp) < self.cache_ttl:
            all_headlines = self._rss_headlines_cache
        else:
            # Fetch fresh
            crypto = self._fetch_crypto_news()
            macro = self._fetch_macro_news_rss()
            all_headlines = crypto + macro

        # Sort by recency (if available) and return top N
        return all_headlines[:max_items]

    def _ddg_search(self, query: str) -> List[str]:
        """Simple DuckDuckGo search returning text snippets.

        Uses the Instant Answer API first, then HTML fallback.
        Kept as a silent fallback only — RSS feeds are the primary source.
        """
        # Try Instant Answer API
        try:
            url = "https://api.duckduckgo.com/"
            params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
            headers = {"User-Agent": "Mozilla/5.0 (compatible; PicsouMarketAwareness/1.0)"}
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            snippets = []
            abstract = data.get("AbstractText", "")
            if abstract:
                snippets.append(abstract[:300])

            for topic in data.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict):
                    text = topic.get("Text", "")
                    if text:
                        snippets.append(text[:300])

            for result_item in data.get("Results", [])[:3]:
                if isinstance(result_item, dict):
                    text = result_item.get("Text", "")
                    if text:
                        snippets.append(text[:300])

            if snippets:
                return snippets
        except Exception as e:
            logger.debug("DDG Instant Answer failed for '%s': %s", query, e)

        # Fallback: HTML search
        try:
            url = "https://html.duckduckgo.com/html/"
            params = {"q": query}
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.5",
            }
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            html = resp.text

            snippets = []
            # Extract result snippets from HTML
            title_pattern = re.compile(
                r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
                re.DOTALL | re.IGNORECASE,
            )
            snippet_pattern = re.compile(
                r'<(?:a|td)[^>]*class="result__snippet"[^>]*>(.*?)</(?:a|td)>',
                re.DOTALL | re.IGNORECASE,
            )

            titles = title_pattern.findall(html)
            snippet_matches = snippet_pattern.findall(html)

            for i, (href, title_html) in enumerate(titles[:8]):
                clean_title = re.sub(r"<[^>]+>", "", title_html).strip()
                if clean_title and len(clean_title) > 10:
                    snippet_text = ""
                    if i < len(snippet_matches):
                        snippet_text = re.sub(r"<[^>]+>", "", snippet_matches[i]).strip()
                    combined = clean_title
                    if snippet_text:
                        combined += f" — {snippet_text}"
                    snippets.append(combined[:300])

            if not snippets:
                # Broader pattern
                broad_pattern = re.compile(
                    r'<a[^>]*href="(https?://[^"]*)"[^>]*>(.*?)</a>',
                    re.DOTALL | re.IGNORECASE,
                )
                for href, text in broad_pattern.findall(html)[:8]:
                    clean = re.sub(r"<[^>]+>", "", text).strip()
                    if clean and len(clean) > 20 and "duckduckgo" not in href:
                        snippets.append(clean[:300])

            return snippets
        except Exception as e:
            logger.debug("DDG HTML search failed for '%s': %s", query, e)
            return []

    # ── Sentiment profiling ────────────────────────────────────────────────

    def _profile_sentiment(self, sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """Build an advanced sentiment profile from Fear & Greed data.

        Tracks the DYNAMIQUE (trend) of F&G, not just the current value.
        """
        fng = sentiment.get("fear_and_greed", {})
        current_value = fng.get("value", 50)
        yesterday_value = fng.get("yesterday")
        classification = fng.get("classification", "Neutral")

        # ── Classification refinement ──
        if current_value <= 10:
            refined_class = "extreme_fear"
        elif current_value <= 25:
            refined_class = "fear"
        elif current_value <= 45:
            refined_class = "mild_fear"
        elif current_value <= 55:
            refined_class = "neutral"
        elif current_value <= 75:
            refined_class = "greed"
        elif current_value <= 90:
            refined_class = "extreme_greed"
        else:
            refined_class = "extreme_greed"

        # ── Trend calculation ──
        trend = "stable"
        trend_delta = 0
        if yesterday_value is not None and isinstance(yesterday_value, (int, float)):
            trend_delta = current_value - yesterday_value
            if trend_delta > 10:
                trend = "rapidly_increasing"
            elif trend_delta > 3:
                trend = "increasing"
            elif trend_delta < -10:
                trend = "rapidly_decreasing"
            elif trend_delta < -3:
                trend = "decreasing"
            else:
                trend = "stable"

        # ── Store in history for longer-term tracking ──
        self._fg_history.append({
            "value": current_value,
            "classification": refined_class,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        # Keep last 50 readings (≈ 4 hours at 5-min intervals)
        if len(self._fg_history) > 50:
            self._fg_history = self._fg_history[-50:]

        # ── Build description ──
        descriptions = {
            "extreme_fear": "Extreme Fear — Market is in panic. Potential capitulation or buying opportunity for contrarians.",
            "fear": "Fear — Market sentiment is negative. Traders are cautious, potential for oversold bounces.",
            "mild_fear": "Mild Fear — Slightly negative sentiment. Cautious positioning warranted.",
            "neutral": "Neutral — Balanced sentiment. No strong directional bias from sentiment alone.",
            "greed": "Greed — Positive sentiment. Be cautious of FOMO and overextension.",
            "extreme_greed": "Extreme Greed — Euphoric market. High risk of correction. Consider taking profits.",
        }

        # ── Build result ──
        result = {
            "classification": refined_class,
            "original_classification": classification,
            "trend": trend,
            "trend_delta": trend_delta,
            "score": current_value,
            "yesterday_score": yesterday_value,
            "description": descriptions.get(refined_class, "Unknown sentiment state."),
        }

        # Add longer-term trend if we have history
        if len(self._fg_history) >= 5:
            oldest = self._fg_history[0]["value"]
            result["medium_term_delta"] = current_value - oldest
            result["medium_term_trend"] = (
                "improving" if current_value > oldest + 5
                else "deteriorating" if current_value < oldest - 5
                else "stable"
            )

        return result

    # ── Behavioral recommendations ────────────────────────────────────────

    def _generate_recommendations(
        self,
        market_regime: Dict[str, Any],
        macro_events: List[Dict[str, Any]],
        sentiment_profile: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate behavioral adaptation recommendations based on regime,
        macro events, and sentiment profile.

        Returns a list of recommendation dicts.
        """
        recommendations = []
        regime = market_regime["regime"]
        confidence = market_regime["confidence"]
        volatility_level = market_regime["volatility_level"]
        sentiment_class = sentiment_profile["classification"]
        sentiment_trend = sentiment_profile["trend"]

        # ── Regime-based recommendations ──
        if regime == "crunch":
            recommendations.append({
                "category": "position_sizing",
                "recommendation": "Reduce position sizes to 40% of normal (max 8% of capital per position)",
                "priority": "critical",
                "reason": "Market crash in progress — extreme risk of large losses",
            })
            recommendations.append({
                "category": "stop_loss",
                "recommendation": "Widen stop losses to 2x normal distance, or go flat (no positions)",
                "priority": "critical",
                "reason": "Extreme volatility means normal stops will be triggered by noise",
            })
            recommendations.append({
                "category": "strategy",
                "recommendation": "Avoid momentum and DCA strategies. Only consider short-term contrarian entries if sentiment is extreme_fear",
                "priority": "critical",
                "reason": "Momentum strategies are destroyed in crashes. DCA averages into falling knives.",
            })
            recommendations.append({
                "category": "max_positions",
                "recommendation": "Limit to 2 simultaneous open positions maximum",
                "priority": "critical",
                "reason": "Concentrated risk in a crash means every position is correlated downward",
            })

        elif regime == "bear":
            recommendations.append({
                "category": "position_sizing",
                "recommendation": "Reduce position sizes to 50% of normal (max 10% of capital per position)",
                "priority": "high",
                "reason": "Bear market — lower exposure reduces downside risk",
            })
            recommendations.append({
                "category": "strategy",
                "recommendation": "Avoid momentum buy strategies. Favor risk_management and mean_reversion for bounces.",
                "priority": "high",
                "reason": "Momentum longs underperform in bear markets. Mean reversion plays oversold bounces.",
            })
            recommendations.append({
                "category": "stop_loss",
                "recommendation": "Tighten stop losses. Cut losing trades faster.",
                "priority": "medium",
                "reason": "Bear markets have strong downside momentum — don't let losers run",
            })

        elif regime == "bull":
            recommendations.append({
                "category": "position_sizing",
                "recommendation": "Increase position sizes to 120% of normal (max 24% of capital per position)",
                "priority": "medium",
                "reason": "Bull market — higher conviction allows slightly larger positions",
            })
            recommendations.append({
                "category": "strategy",
                "recommendation": "Favor momentum and breakout strategies. Hold winners longer.",
                "priority": "medium",
                "reason": "Momentum strategies outperform in trending up markets",
            })
            recommendations.append({
                "category": "stop_loss",
                "recommendation": "Use wider stops to avoid being shaken out by normal pullbacks",
                "priority": "medium",
                "reason": "Bull markets have healthy pullbacks within the uptrend",
            })

        elif regime == "volatile":
            recommendations.append({
                "category": "position_sizing",
                "recommendation": "Reduce position sizes to 70% of normal (max 14% of capital per position)",
                "priority": "medium",
                "reason": "High volatility means larger swings — reduce position to manage risk",
            })
            recommendations.append({
                "category": "strategy",
                "recommendation": "Prefer mean_reversion and scalping over momentum. Avoid breakout strategies until volatility contracts.",
                "priority": "medium",
                "reason": "Volatile choppy markets whipsaw trend-followers. Mean reversion profits from the oscillation.",
            })
            recommendations.append({
                "category": "stop_loss",
                "recommendation": "Widen stops slightly to accommodate volatility, but set hard risk limits",
                "priority": "medium",
                "reason": "Normal stops get triggered by volatility spikes, but you still need protection",
            })

        elif regime == "range":
            recommendations.append({
                "category": "strategy",
                "recommendation": "Prefer mean_reversion and scalping strategies. Avoid momentum.",
                "priority": "medium",
                "reason": "Ranging markets reward buying at support and selling at resistance",
            })
            recommendations.append({
                "category": "position_sizing",
                "recommendation": "Use normal position sizes or slightly reduce (max 18% of capital per position)",
                "priority": "low",
                "reason": "Range markets have limited directional risk but also limited profit potential",
            })
            recommendations.append({
                "category": "stop_loss",
                "recommendation": "Use tight stops — in a range, if price breaks out of the range it's likely a real move",
                "priority": "medium",
                "reason": "Tight stops work well in ranges because breakouts are meaningful",
            })

        # ── Sentiment-based adjustments ──
        if sentiment_class == "extreme_fear":
            recommendations.append({
                "category": "sentiment",
                "recommendation": "Extreme Fear detected — consider small contrarian buy positions if technical support holds",
                "priority": "high",
                "reason": "Extreme fear often marks capitulation bottoms, but only if price action confirms",
            })
        elif sentiment_class == "extreme_greed":
            recommendations.append({
                "category": "sentiment",
                "recommendation": "Extreme Greed detected — consider taking profits and reducing exposure",
                "priority": "high",
                "reason": "Extreme greed often precedes corrections. Protect gains.",
            })

        # ── Macro event adjustments ──
        high_impact_events = [e for e in macro_events if e.get("magnitude", 0) >= 4]
        negative_events = [e for e in high_impact_events if e.get("impact") == "negative"]
        positive_events = [e for e in high_impact_events if e.get("impact") == "positive"]

        if negative_events:
            event_descs = ", ".join(e.get("keyword", "unknown") for e in negative_events[:3])
            recommendations.append({
                "category": "macro_risk",
                "recommendation": f"Caution: High-impact negative macro events detected ({event_descs}). Reduce exposure and tighten stops.",
                "priority": "high",
                "reason": "Negative macro events create systemic risk that affects all crypto assets",
            })

        if positive_events:
            event_descs = ", ".join(e.get("keyword", "unknown") for e in positive_events[:3])
            recommendations.append({
                "category": "macro_opportunity",
                "recommendation": f"Positive macro events detected ({event_descs}). May support bullish momentum.",
                "priority": "medium",
                "reason": "Positive macro events can sustain or accelerate uptrends",
            })

        # ── Sentiment trend adjustments ──
        if sentiment_trend in ("rapidly_decreasing", "rapidly_increasing"):
            recommendations.append({
                "category": "sentiment_shift",
                "recommendation": f"Sentiment is {sentiment_trend.replace('_', ' ')} — market regime may be shifting, be prepared for volatility",
                "priority": "medium",
                "reason": "Rapid sentiment shifts often precede trend changes or volatility spikes",
            })

        return recommendations

    # ── Risk parameter adaptation ─────────────────────────────────────────

    def get_adapted_risk_params(
        self, market_context: Dict[str, Any], base_risk: Any
    ) -> Dict[str, Any]:
        """Adapt risk parameters based on market context.

        Args:
            market_context: The full market context from analyze().
            base_risk: The base RiskConfig from PicsouConfig.

        Returns:
            Dict with adapted risk parameters.
        """
        regime = market_context["market_regime"]["regime"]
        confidence = market_context["market_regime"]["confidence"]

        # Start with base values
        max_position_pct = base_risk.max_position_pct
        max_open_positions = base_risk.max_open_positions

        # ── Regime-based adaptation ──
        if regime == "crunch":
            max_position_pct = min(max_position_pct, 0.08)  # Cap at 8%
            max_open_positions = min(max_open_positions, 2)
        elif regime == "bear":
            max_position_pct = min(max_position_pct, 0.10)  # Cap at 10%
            max_open_positions = min(max_open_positions, 3)
        elif regime == "bull":
            max_position_pct = min(max_position_pct * 1.20, 0.24)  # Up to 24% (1.2x of 20%)
            max_open_positions = max_open_positions  # Keep normal (5)
        elif regime == "volatile":
            max_position_pct = min(max_position_pct, 0.14)  # Cap at 14%
            max_open_positions = min(max_open_positions, 4)
        elif regime == "range":
            max_position_pct = min(max_position_pct, 0.18)  # Cap at 18%
            max_open_positions = max_open_positions  # Keep normal (5)

        # ── Confidence adjustment ──
        # Low confidence in regime detection = more conservative
        if confidence < 0.4:
            max_position_pct *= 0.8
            max_open_positions = max(2, max_open_positions - 1)

        # ── Macro indicator-based adjustments ──
        # Use real macro data to further adapt risk
        indicators = market_context.get("macro_indicators", {})
        macro_assessment = market_context.get("macro_assessment", {})

        # DXY adjustment: strong dollar = reduce crypto exposure
        dxy = indicators.get("dxy", {})
        if dxy:
            dxy_val = dxy.get("value", 0)
            if dxy_val > 103:
                max_position_pct *= 0.85  # Reduce 15% when dollar very strong
                logger.debug("DXY=%.2f > 103: reducing position size by 15%%", dxy_val)
            elif dxy_val < 95:
                max_position_pct = min(max_position_pct * 1.10, 0.24)  # Increase 10% when dollar weak
                logger.debug("DXY=%.2f < 95: increasing position size by 10%%", dxy_val)

        # VIX adjustment: high fear = reduce exposure
        vix = indicators.get("vix", {})
        if vix:
            vix_val = vix.get("value", 0)
            if vix_val > 35:
                max_position_pct *= 0.70  # Severe risk-off
                max_open_positions = max(1, max_open_positions - 2)
                logger.debug("VIX=%.1f > 35: severe risk-off, reducing positions significantly", vix_val)
            elif vix_val > 25:
                max_position_pct *= 0.85  # Moderate risk-off
                logger.debug("VIX=%.1f > 25: risk-off, reducing position size by 15%%", vix_val)

        # Yield adjustment: high yields = reduce crypto exposure
        yields = indicators.get("us_10y_yield", {})
        if yields:
            yield_val = yields.get("value", 0)
            if yield_val > 4.5:
                max_position_pct *= 0.85  # Dangerous zone
                logger.debug("10Y Yield=%.2f%% > 4.5%%: reducing position size by 15%%", yield_val)

        # LLM macro assessment adjustment
        if macro_assessment:
            risk_outlook = macro_assessment.get("risk_outlook", "")
            if risk_outlook == "dangerous":
                max_position_pct *= 0.70
                max_open_positions = max(1, max_open_positions - 1)
                logger.debug("Macro assessment: dangerous risk outlook, reducing positions significantly")
            elif risk_outlook == "unfavorable":
                max_position_pct *= 0.85
                logger.debug("Macro assessment: unfavorable risk outlook, reducing position size by 15%%")
            elif risk_outlook == "favorable":
                max_position_pct = min(max_position_pct * 1.10, 0.24)
                logger.debug("Macro assessment: favorable risk outlook, increasing position size by 10%%")

        return {
            "max_position_pct": round(max_position_pct, 4),
            "max_open_positions": max_open_positions,
            "regime": regime,
            "confidence": confidence,
            "adapted": True,
        }

    # ── Strategy filter ──────────────────────────────────────────────────

    def get_preferred_strategies(self, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get strategy preferences based on market regime.

        Returns a dict with preferred, neutral, and avoid strategy lists.
        """
        regime = market_context["market_regime"]["regime"]
        sentiment_class = market_context["sentiment_profile"]["classification"]

        strategy_prefs = {
            "bull": {
                "preferred": ["momentum", "breakout"],
                "neutral": ["mean_reversion", "dca"],
                "avoid": [],
            },
            "bear": {
                "preferred": ["risk_management"],
                "neutral": ["mean_reversion"],
                "avoid": ["momentum", "breakout", "dca"],
            },
            "crunch": {
                "preferred": ["risk_management"],
                "neutral": [],
                "avoid": ["momentum", "breakout", "dca", "mean_reversion"],
            },
            "volatile": {
                "preferred": ["mean_reversion", "scalping"],
                "neutral": ["risk_management"],
                "avoid": ["momentum", "breakout"],
            },
            "range": {
                "preferred": ["mean_reversion"],
                "neutral": ["scalping", "risk_management"],
                "avoid": ["momentum"],
            },
        }

        prefs = strategy_prefs.get(regime, strategy_prefs["range"])

        # Sentiment override: extreme fear can make contrarian viable
        if sentiment_class == "extreme_fear" and regime not in ("crunch",):
            if "contrarian" not in prefs["preferred"]:
                prefs["preferred"].append("contrarian")

        # Sentiment override: extreme greed = be cautious
        if sentiment_class == "extreme_greed" and regime == "bull":
            if "momentum" in prefs["preferred"]:
                prefs["preferred"].remove("momentum")
                prefs["neutral"].append("momentum")

        # Macro indicator overrides
        indicators = market_context.get("macro_indicators", {})
        vix = indicators.get("vix", {})
        if vix and vix.get("value", 0) > 30:
            # High VIX: avoid breakout regardless of regime
            if "breakout" in prefs["preferred"]:
                prefs["preferred"].remove("breakout")
                prefs["neutral"].append("breakout")

        dxy = indicators.get("dxy", {})
        if dxy and dxy.get("value", 0) > 105:
            # Very strong dollar: avoid momentum buy
            if "momentum" in prefs["preferred"]:
                prefs["preferred"].remove("momentum")
                prefs["neutral"].append("momentum")

        return prefs