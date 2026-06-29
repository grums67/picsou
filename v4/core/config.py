"""Picsou v4 configuration — minimal, overridable by the agent itself."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class SafetyConfig:
    """Hard safety limits — the agent CANNOT override these."""
    max_position_pct: float = 0.20      # Max 20% capital per position
    max_open_positions: int = 5          # Max 5 simultaneous positions
    max_drawdown_pct: float = 0.20       # Circuit breaker at 20% drawdown
    min_position_usd: float = 10.0       # Minimum $10 per trade
    max_strategies_active: int = 8       # Max simultaneous active strategies
    cooldown_seconds: int = 300          # Seconds between cycles (heartbeat)
    brain_interval_cycles: int = 3       # Run LLM brain every N cycles (~15min)

@dataclass
class ExchangeConfig:
    """Configuration for a single exchange."""
    name: str
    rest_url: str
    fee_rate: float
    symbol_format: str


@dataclass
class LLMConfig:
    """LLM brain configuration."""
    url: str = "http://127.0.0.1:11434/v1"
    api_key: str = ""
    model: str = "deepseek-v4-flash:cloud"
    temperature: float = 0.55
    max_tokens: int = 8192


@dataclass
class TelegramConfig:
    """Telegram bot configuration."""
    token: str = ""
    authorized_user_ids: list = field(default_factory=lambda: [])


@dataclass
class PicsouConfig:
    """Main configuration — everything the agent can read and modify."""
    phase: str = "learning"  # "learning" (paper) or "live"
    starting_capital: float = 10000.0
    symbols: list = field(default_factory=lambda: ["BTC", "ETH", "SOL"])
    candle_interval: str = "1h"

    safety: SafetyConfig = field(default_factory=SafetyConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    exchanges: Dict[str, ExchangeConfig] = field(default_factory=dict)

    # Paths
    base_path: Path = Path("/root/PROJECTS/picsou/v4")
    data_path: Path = Path("/root/PROJECTS/picsou/v4/data")
    strategies_path: Path = Path("/root/PROJECTS/picsou/v4/strategies")

    def __post_init__(self):
        if not self.exchanges:
            self.exchanges = {
                "okx": ExchangeConfig(
                    name="okx",
                    rest_url="https://www.okx.com/api/v5",
                    fee_rate=0.0008,
                    symbol_format="{base}-USDT",
                ),
            }
        for p in (self.data_path, self.strategies_path):
            p.mkdir(parents=True, exist_ok=True)

        # Load LLM key from env or .env
        if not self.llm.api_key:
            self.llm.api_key = os.environ.get("PICSOU_LLM_KEY", "")
        if not self.llm.api_key:
            env_file = self.base_path.parent / ".env"
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("#") or not line:
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key in ("PICSOU_LLM_KEY",) and val:
                            self.llm.api_key = val
                            break

        # Load Telegram config from env or .env
        if not self.telegram.token:
            self.telegram.token = os.environ.get("PICSOU_TELEGRAM_TOKEN", "")
        if not self.telegram.authorized_user_ids:
            env_ids = os.environ.get("PICSOU_TELEGRAM_USERS", "")
            if env_ids:
                self.telegram.authorized_user_ids = [
                    int(uid.strip()) for uid in env_ids.split(",") if uid.strip()
                ]
        if not self.telegram.token:
            env_file = self.base_path.parent / ".env"
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("#") or not line:
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key == "PICSOU_TELEGRAM_TOKEN" and val:
                            self.telegram.token = val
                        elif key == "PICSOU_TELEGRAM_USERS" and val:
                            self.telegram.authorized_user_ids = [
                                int(uid.strip()) for uid in val.split(",") if uid.strip()
                            ]


def get_config() -> PicsouConfig:
    return PicsouConfig()