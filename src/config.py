"""Picsou autonomous crypto trading agent - configuration module."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class ExchangeConfig:
    """Configuration for a single exchange."""
    name: str
    rest_url: str
    ws_url: str
    fee_rate: float  # Trading fee as decimal (e.g. 0.0008 = 0.08%)
    symbol_format: str  # e.g. "BTC-USDT" for OKX


@dataclass
class RiskConfig:
    """Risk management rules."""
    max_position_pct: float = 0.20  # Max 20% of capital per position
    max_open_positions: int = 5
    max_drawdown_pct: float = 0.20  # Max 20% drawdown before pause


@dataclass
class LearningConfig:
    """Learning phase thresholds."""
    win_rate_threshold: float = 0.55
    min_trades: int = 50
    min_days: int = 14
    elimination_win_rate: float = 0.50
    elimination_max_drawdown: float = 0.30


@dataclass
class DataPaths:
    """Filesystem paths for data, logs, and backtests."""
    base: Path = Path("/root/PROJECTS/picsou")
    data: Path = Path("/root/PROJECTS/picsou/data")
    logs: Path = Path("/root/PROJECTS/picsou/logs")
    backtests: Path = Path("/root/PROJECTS/picsou/backtests")

    def __post_init__(self) -> None:
        """Ensure directories exist."""
        for p in (self.data, self.logs, self.backtests):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class PicsouConfig:
    """Main configuration for the Picsou agent."""

    # Current phase: "learning" (paper trading) or "live"
    phase: str = "learning"

    # Paper trading starting capital in EUR (10x real amount for learning)
    starting_capital: float = 10000.0  # 1000€ * 10x simulation multiplier

    # Real capital this represents
    real_capital: float = 1000.0

    # Exchange configurations
    exchanges: Dict[str, ExchangeConfig] = field(default_factory=dict)

    # Risk rules
    risk: RiskConfig = field(default_factory=RiskConfig)

    # Learning thresholds
    learning: LearningConfig = field(default_factory=LearningConfig)

    # Data paths
    paths: DataPaths = field(default_factory=DataPaths)

    # Symbols to trade (USDT pairs)
    symbols: list = field(default_factory=lambda: ["BTC", "ETH", "SOL"])

    # Candle interval for analysis
    candle_interval: str = "1h"

    # How often to run the main loop (seconds)
    loop_interval: int = 300  # 5 minutes

    # ── LLM Brain configuration ─────────────────────────────────────────
    # Provider: Mistral AI
    llm_url: str = "http://127.0.0.1:11434/v1"
    llm_api_key: str = ""  # From PICSOU_LLM_KEY or OLLAMA_API_KEY env var or .env
    llm_model: str = "kimi-k2.6:cloud"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2000
    # Hot-reload config file path (brain.py reloads this each cycle)
    llm_config_path: str = ""  # Auto-set to data/llm_config.json in __post_init__

    # ── Sentiment data sources ───────────────────────────────────────────
    fear_and_greed_enabled: bool = True
    news_enabled: bool = True

    def __post_init__(self) -> None:
        """Set up default exchange configs and load env vars."""
        if not self.exchanges:
            self.exchanges = {
                "okx": ExchangeConfig(
                    name="okx",
                    rest_url="https://www.okx.com/api/v5",
                    ws_url="wss://ws.okx.com:8443/ws/v5/public",
                    fee_rate=0.0008,
                    symbol_format="{base}-USDT",
                ),
                "kraken": ExchangeConfig(
                    name="kraken",
                    rest_url="https://api.kraken.com/0/public",
                    ws_url="wss://ws.kraken.com",
                    fee_rate=0.0026,
                    symbol_format="{base}USDT",
                ),
                "bitstamp": ExchangeConfig(
                    name="bitstamp",
                    rest_url="https://www.bitstamp.net/api/v2",
                    ws_url="wss://ws.bitstamp.net",
                    fee_rate=0.0025,
                    symbol_format="{base}usdt",
                ),
            }
        self.paths.__post_init__()

        # Load LLM API key from environment or .env file
        # Priority: PICSOU_LLM_KEY > MISTRAL_API_KEY > OLLAMA_API_KEY
        if not self.llm_api_key:
            self.llm_api_key = os.environ.get("PICSOU_LLM_KEY", "")
        if not self.llm_api_key:
            self.llm_api_key = os.environ.get("MISTRAL_API_KEY", "")
        if not self.llm_api_key:
            self.llm_api_key = os.environ.get("OLLAMA_API_KEY", "")

        # Allow env override for LLM URL
        env_url = os.environ.get("PICSOU_LLM_URL", "")
        if env_url:
            self.llm_url = env_url

        # Auto-set llm_config_path if not specified
        if not self.llm_config_path:
            self.llm_config_path = str(self.paths.data / "llm_config.json")

        # Try loading from .env file if still no key
        if not self.llm_api_key:
            env_file = Path(self.paths.base) / ".env"
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("#") or not line:
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key in ("PICSOU_LLM_KEY", "MISTRAL_API_KEY") and val:
                            self.llm_api_key = val
                            break


# Singleton config instance
def get_config() -> PicsouConfig:
    """Return the default Picsou configuration."""
    return PicsouConfig()