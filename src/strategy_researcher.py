"""StrategyResearcher - Fetches crypto trading strategy insights from online sources.

Aggregates data from:
- CoinGecko trending (reuses brain.fetch_crypto_headlines)
- CryptoCompare news (reuses brain.fetch_crypto_headlines)
- DuckDuckGo web search (no API key needed)
- CryptoCompare OHLCV technical analysis

Returns structured insights injected into the LLM prompt to enrich
decision-making with web-sourced context.
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class StrategyResearcher:
    """Fetches and caches online strategy insights for the LLM prompt.

    Uses multiple web sources to find trending strategies, technical signals,
    risk factors, and market narrative. Results are cached with a configurable
    TTL to avoid spamming APIs.
    """

    # Keywords for extracting strategy signals from text
    STRATEGY_KEYWORDS = {
        "momentum": ["momentum", "trend following", "trend-following", "trending"],
        "mean_reversion": ["mean reversion", "oversold", "overbought", "reversion to mean"],
        "breakout": ["breakout", "break out", "breaks out", "resistance break", "range break"],
        "contrarian": ["contrarian", "counter-trend", "counter trend", "fade", "contrarian buy"],
        "dca": ["dca", "dollar cost averaging", "dollar-cost averaging", "accumulate"],
        "scalping": ["scalping", "scalp", "quick trade", "short-term trade"],
        "swing": ["swing trade", "swing trading", "swing"],
        "grid": ["grid trading", "grid bot", "grid strategy"],
    }

    TECHNICAL_SIGNAL_KEYWORDS = {
        "RSI": ["rsi", "relative strength index", "overbought rsi", "oversold rsi"],
        "MACD": ["macd", "moving average convergence divergence", "macd crossover"],
        "Bollinger": ["bollinger", "bollinger bands", "bollinger band"],
        "EMA": ["ema", "exponential moving average", "ema crossover"],
        "SMA": ["sma", "simple moving average", "sma crossover"],
        "Fibonacci": ["fibonacci", "fib retracement", "fib level"],
        "Volume": ["volume spike", "volume surge", "high volume", "volume profile"],
        "Support/Resistance": ["support level", "resistance level", "support zone", "resistance zone"],
        "Stochastic": ["stochastic", "stoch rsi", "stochastics"],
    }

    def __init__(self, cache_path: str = "", cache_ttl: int = 1800,
                 max_sources: int = 5, enabled: bool = True) -> None:
        self.enabled = enabled
        self.cache_ttl = cache_ttl  # seconds (default 30 min)
        self.max_sources = max_sources
        self.cache_path = Path(cache_path) if cache_path else Path("")
        self._cache: Dict[str, Any] = {}
        self._cache_timestamp: float = 0.0
        self._load_cache()

    # ── Cache management ─────────────────────────────────────────────────

    def _load_cache(self) -> None:
        """Load cached insights from disk."""
        if not self.cache_path:
            return
        try:
            if self.cache_path.exists():
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._cache = data.get("insights", {})
                self._cache_timestamp = data.get("timestamp", 0.0)
                logger.info("Loaded research cache (age=%.0fs)",
                            time.time() - self._cache_timestamp)
        except Exception as e:
            logger.warning("Failed to load research cache: %s", e)
            self._cache = {}
            self._cache_timestamp = 0.0

    def _save_cache(self) -> None:
        """Persist cached insights to disk."""
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump({
                    "insights": self._cache,
                    "timestamp": self._cache_timestamp,
                }, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save research cache: %s", e)

    def _is_cache_valid(self) -> bool:
        """Check if cached insights are still within TTL."""
        if not self._cache:
            return False
        age = time.time() - self._cache_timestamp
        return age < self.cache_ttl

    # ── Public API ───────────────────────────────────────────────────────

    def fetch_strategy_insights(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Fetch and aggregate strategy insights from multiple online sources.

        Args:
            symbols: List of crypto symbols to research (e.g. ['BTC', 'ETH']).

        Returns:
            Dict with keys:
            - market_narrative: str - 2-3 sentence summary of market sentiment
            - trending_strategies: List[str] - strategy names currently popular
            - technical_signals: List[str] - technical indicators mentioned online
            - risk_factors: List[str] - identified risks
            - source_urls: List[str] - URLs of sources used
        """
        if not self.enabled:
            logger.debug("Strategy research disabled, returning empty insights")
            return self._empty_insights()

        # Return cache if valid
        if self._is_cache_valid():
            logger.debug("Using cached research insights (age=%.0fs)",
                         time.time() - self._cache_timestamp)
            return self._cache

        # Fetch from all sources
        logger.info("Fetching fresh strategy insights for symbols=%s", symbols or ["general"])

        all_snippets: List[str] = []
        all_urls: List[str] = []
        strategy_mentions: Dict[str, int] = {}
        signal_mentions: Dict[str, int] = {}
        risk_items: List[str] = []
        narrative_parts: List[str] = []

        # Source 1: DuckDuckGo search
        ddg_results = self._fetch_duckduckgo(symbols)
        if ddg_results:
            all_snippets.extend(ddg_results.get("snippets", []))
            all_urls.extend(ddg_results.get("urls", []))
            strategy_mentions = self._merge_counts(
                strategy_mentions, ddg_results.get("strategy_mentions", {}))
            signal_mentions = self._merge_counts(
                signal_mentions, ddg_results.get("signal_mentions", {}))
            risk_items.extend(ddg_results.get("risks", []))

        # Source 2: CryptoCompare news + analysis
        cc_results = self._fetch_cryptocompare(symbols)
        if cc_results:
            all_snippets.extend(cc_results.get("snippets", []))
            all_urls.extend(cc_results.get("urls", []))
            strategy_mentions = self._merge_counts(
                strategy_mentions, cc_results.get("strategy_mentions", {}))
            signal_mentions = self._merge_counts(
                signal_mentions, cc_results.get("signal_mentions", {}))
            risk_items.extend(cc_results.get("risks", []))

        # Build structured result
        # Market narrative: summarize from snippets
        if all_snippets:
            top_snippets = all_snippets[:5]
            narrative_parts = [s[:200] for s in top_snippets[:3]]
            market_narrative = " ".join(narrative_parts) if narrative_parts else ""
        else:
            market_narrative = "No web research data available this cycle."

        # Trending strategies: sorted by mention count
        trending_strategies = sorted(
            strategy_mentions.keys(), key=lambda k: strategy_mentions[k], reverse=True
        )[:5]

        # Technical signals: sorted by mention count
        technical_signals = sorted(
            signal_mentions.keys(), key=lambda k: signal_mentions[k], reverse=True
        )[:5]

        # Deduplicate risk factors
        seen_risks = set()
        unique_risks = []
        for r in risk_items:
            r_lower = r.lower()
            if r_lower not in seen_risks:
                seen_risks.add(r_lower)
                unique_risks.append(r)
        risk_factors = unique_risks[:5]

        insights = {
            "market_narrative": market_narrative,
            "trending_strategies": trending_strategies,
            "technical_signals": technical_signals,
            "risk_factors": risk_factors,
            "source_urls": all_urls[:self.max_sources],
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

        # Cache the result
        self._cache = insights
        self._cache_timestamp = time.time()
        self._save_cache()

        logger.info("Research insights updated: strategies=%s, signals=%s, risks=%d",
                     trending_strategies, technical_signals, len(risk_factors))
        return insights

    # ── Source: DuckDuckGo ───────────────────────────────────────────────

    def _fetch_duckduckgo(self, symbols: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Search DuckDuckGo for crypto trading strategies.

        No API key required. Tries the Instant Answer API first, then
        falls back to the HTML search endpoint.
        """
        query_parts = ["crypto trading strategy"]
        if symbols:
            query_parts.append(" ".join(symbols[:3]))
        date_str = datetime.now(timezone.utc).strftime("%Y-%m")
        query_parts.append(date_str)
        query = " ".join(query_parts)

        # Try Instant Answer API first (more reliable for bots)
        result = self._fetch_ddg_instant_answer(query)
        if result and result.get("snippets"):
            return result

        # Fallback: HTML search
        result = self._fetch_ddg_html(query)
        return result

    def _fetch_ddg_instant_answer(self, query: str) -> Optional[Dict[str, Any]]:
        """Query DuckDuckGo Instant Answer API."""
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
            }
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; PicsouBot/1.0)",
            }
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            snippets = []
            urls = []

            # Abstract (main answer)
            abstract = data.get("AbstractText", "")
            abstract_url = data.get("AbstractURL", "")
            if abstract:
                snippets.append(abstract[:300])
                if abstract_url:
                    urls.append(abstract_url)

            # Related topics
            for topic in data.get("RelatedTopics", [])[:10]:
                if isinstance(topic, dict):
                    text = topic.get("Text", "")
                    turl = topic.get("FirstURL", "")
                    if text:
                        snippets.append(text[:300])
                    if turl:
                        urls.append(turl)

            # Results
            for result_item in data.get("Results", [])[:5]:
                if isinstance(result_item, dict):
                    text = result_item.get("Text", "")
                    rurl = result_item.get("FirstURL", "")
                    if text:
                        snippets.append(text[:300])
                    if rurl:
                        urls.append(rurl)

            if not snippets:
                logger.debug("DuckDuckGo Instant Answer: no results for query='%s'", query)
                return None

            strategy_mentions = self._extract_strategy_mentions(snippets)
            signal_mentions = self._extract_technical_signals(snippets)
            risks = self._extract_risk_factors(snippets)

            logger.info("DuckDuckGo Instant Answer: fetched %d snippets for query='%s'",
                        len(snippets), query)
            return {
                "snippets": snippets,
                "urls": urls,
                "strategy_mentions": strategy_mentions,
                "signal_mentions": signal_mentions,
                "risks": risks,
            }

        except Exception as e:
            logger.debug("DuckDuckGo Instant Answer failed: %s", e)
            return None

    def _fetch_ddg_html(self, query: str) -> Optional[Dict[str, Any]]:
        """Search DuckDuckGo via HTML endpoint (fallback)."""
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
            urls = []
            # Parse results from DDG HTML
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

            # Alternative broader pattern if DDG changes class names
            if not titles:
                broad_pattern = re.compile(
                    r'<a[^>]*href="(https?://[^"]*)"[^>]*>(.*?)</a>',
                    re.DOTALL | re.IGNORECASE,
                )
                broad_matches = broad_pattern.findall(html)
                for href, text in broad_matches[:10]:
                    clean_text = re.sub(r"<[^>]+>", "", text).strip()
                    clean_href = href.strip()
                    if clean_text and len(clean_text) > 20 and "duckduckgo" not in clean_href:
                        snippets.append(clean_text[:300])
                        urls.append(clean_href)

            for i, (href, title_html) in enumerate(titles[:10]):
                clean_title = re.sub(r"<[^>]+>", "", title_html).strip()
                clean_href = href.strip()
                if clean_title and len(clean_title) > 10:
                    snippet_text = ""
                    if i < len(snippet_matches):
                        snippet_text = re.sub(r"<[^>]+>", "", snippet_matches[i]).strip()
                    combined = clean_title
                    if snippet_text:
                        combined += f" — {snippet_text}"
                    snippets.append(combined[:300])
                    urls.append(clean_href)

            if not snippets:
                logger.debug("DuckDuckGo HTML: no snippets extracted")
                return None

            strategy_mentions = self._extract_strategy_mentions(snippets)
            signal_mentions = self._extract_technical_signals(snippets)
            risks = self._extract_risk_factors(snippets)

            logger.info("DuckDuckGo HTML: fetched %d snippets for query='%s'",
                        len(snippets), query)
            return {
                "snippets": snippets,
                "urls": urls,
                "strategy_mentions": strategy_mentions,
                "signal_mentions": signal_mentions,
                "risks": risks,
            }

        except requests.exceptions.Timeout:
            logger.warning("DuckDuckGo HTML search timed out")
            return None
        except requests.exceptions.ConnectionError:
            logger.warning("DuckDuckGo connection error")
            return None
        except Exception as e:
            logger.warning("DuckDuckGo HTML search failed: %s", e)
            return None

    # ── Source: CryptoCompare ────────────────────────────────────────────

    def _fetch_cryptocompare(self, symbols: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Fetch news and analysis from CryptoCompare.

        Aggregates news headlines and extracts strategy/signal mentions.
        Falls back to CoinGecko trending if CryptoCompare is unavailable.
        """
        snippets = []
        urls = []

        # Source 2a: CryptoCompare news
        try:
            resp = requests.get(
                "https://min-api.cryptocompare.com/data/v2/news/?lang=EN",
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                articles = data.get("Data", [])[:20]

                for a in articles:
                    title = a.get("title", "")
                    body = a.get("body", "")[:200] if a.get("body") else ""
                    url = a.get("url", "")

                    if title:
                        combined = title
                        if body:
                            combined += f" — {body}"
                        snippets.append(combined[:300])
                    if url:
                        urls.append(url)

                logger.info("CryptoCompare news: fetched %d articles", len(articles))
            else:
                logger.debug("CryptoCompare news returned status %d, skipping", resp.status_code)
        except requests.exceptions.Timeout:
            logger.warning("CryptoCompare news timed out")
        except requests.exceptions.ConnectionError:
            logger.warning("CryptoCompare connection error")
        except Exception as e:
            logger.debug("CryptoCompare news failed: %s", e)

        # Source 2b: CoinGecko trending (fallback/supplement)
        try:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/search/trending",
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                coins = data.get("coins", [])
                for c in coins[:10]:
                    item = c.get("item", {})
                    name = item.get("name", "")
                    symbol = item.get("symbol", "")
                    market_cap = item.get("data", {}).get("market_cap", "N/A")
                    price_btc = item.get("data", {}).get("price_btc", "N/A")
                    if name:
                        snippet = f"Trending: {name} ({symbol}) - market_cap={market_cap}, price_btc={price_btc}"
                        snippets.append(snippet[:300])
                if coins:
                    logger.info("CoinGecko trending: fetched %d items", len(coins))
            else:
                logger.debug("CoinGecko trending returned status %d", resp.status_code)
        except requests.exceptions.Timeout:
            logger.warning("CoinGecko trending timed out")
        except requests.exceptions.ConnectionError:
            logger.warning("CoinGecko connection error")
        except Exception as e:
            logger.debug("CoinGecko trending failed: %s", e)

        if not snippets:
            logger.debug("CryptoCompare+CoinGecko: no snippets fetched")
            return None

        # Fetch technical analysis data for specific symbols
        if symbols:
            for sym in symbols[:2]:  # Limit to 2 symbols to avoid rate limiting
                ta_data = self._fetch_cryptocompare_ta(sym)
                if ta_data:
                    snippets.append(ta_data)

        # Extract structured signals
        strategy_mentions = self._extract_strategy_mentions(snippets)
        signal_mentions = self._extract_technical_signals(snippets)
        risks = self._extract_risk_factors(snippets)

        logger.info("CryptoCompare+CoinGecko: fetched %d total snippets", len(snippets))
        return {
            "snippets": snippets,
            "urls": urls,
            "strategy_mentions": strategy_mentions,
            "signal_mentions": signal_mentions,
            "risks": risks,
        }

    def _fetch_cryptocompare_ta(self, symbol: str) -> Optional[str]:
        """Fetch technical analysis summary from CryptoCompare for a symbol.

        Uses the CryptoCompare OHLCV data endpoint to derive basic
        technical observations (trend direction, volatility).
        """
        try:
            # CryptoCompare uses different symbol format
            cc_symbol = symbol.upper()
            if cc_symbol == "BTC":
                cc_symbol = "BTC"
            elif cc_symbol == "ETH":
                cc_symbol = "ETH"

            url = f"https://min-api.cryptocompare.com/data/v2/histohour"
            params = {
                "fsym": cc_symbol,
                "tsym": "USD",
                "limit": 24,  # Last 24 hours
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            candles = data.get("Data", {}).get("Data", [])

            if len(candles) < 10:
                return None

            # Simple technical summary
            closes = [c.get("close", 0) for c in candles if c.get("close", 0) > 0]
            if len(closes) < 5:
                return None

            current = closes[-1]
            avg_6h = sum(closes[-6:]) / len(closes[-6:]) if len(closes) >= 6 else current
            change_24h = (current - closes[0]) / closes[0] * 100 if closes[0] > 0 else 0
            volatility = max(closes) - min(closes)

            direction = "bullish" if current > avg_6h else "bearish"
            strength = abs(change_24h)

            summary = (
                f"CC TA {symbol}/USD: 24h change={change_24h:+.2f}%, "
                f"6h avg={avg_6h:.2f}, current={current:.2f}, "
                f"trend={direction}, 24h range={volatility:.2f}"
            )
            return summary

        except Exception as e:
            logger.debug("CryptoCompare TA for %s failed: %s", symbol, e)
            return None

    # ── Text extraction helpers ───────────────────────────────────────────

    def _extract_strategy_mentions(self, texts: List[str]) -> Dict[str, int]:
        """Extract strategy type mentions from text snippets.

        Returns dict mapping strategy name to mention count.
        """
        mentions: Dict[str, int] = {}
        combined = " ".join(texts).lower()
        for strategy, keywords in self.STRATEGY_KEYWORDS.items():
            for kw in keywords:
                count = combined.count(kw.lower())
                if count > 0:
                    mentions[strategy] = mentions.get(strategy, 0) + count
        return mentions

    def _extract_technical_signals(self, texts: List[str]) -> Dict[str, int]:
        """Extract technical signal mentions from text snippets.

        Returns dict mapping signal name to mention count.
        """
        mentions: Dict[str, int] = {}
        combined = " ".join(texts).lower()
        for signal, keywords in self.TECHNICAL_SIGNAL_KEYWORDS.items():
            for kw in keywords:
                count = combined.count(kw.lower())
                if count > 0:
                    mentions[signal] = mentions.get(signal, 0) + count
        return mentions

    def _extract_risk_factors(self, texts: List[str]) -> List[str]:
        """Extract risk-related statements from text snippets.

        Returns list of risk factor strings.
        """
        risks = []
        risk_keywords = [
            "crash", "dump", "bearish", "correction", "pullback",
            "risk", "warning", "caution", "volatile", "uncertainty",
            "regulation", "ban", "hack", "exploit", "liquidation",
            "whale sell", "whale selling", "overbought",
        ]
        combined = " ".join(texts).lower()
        # Find sentences containing risk keywords
        for text in texts:
            sentences = re.split(r'[.!?]', text)
            for sentence in sentences:
                sentence_lower = sentence.lower().strip()
                if any(kw in sentence_lower for kw in risk_keywords):
                    clean = sentence.strip()[:150]
                    if len(clean) > 20:
                        risks.append(clean)
        return risks[:5]

    # ── Utility ───────────────────────────────────────────────────────────

    @staticmethod
    def _merge_counts(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
        """Merge two count dicts, summing values for matching keys."""
        result = dict(a)
        for k, v in b.items():
            result[k] = result.get(k, 0) + v
        return result

    @staticmethod
    def _empty_insights() -> Dict[str, Any]:
        """Return an empty insights dict for when research is disabled."""
        return {
            "market_narrative": "Research disabled — no web insights available.",
            "trending_strategies": [],
            "technical_signals": [],
            "risk_factors": [],
            "source_urls": [],
            "fetched_at": "",
        }