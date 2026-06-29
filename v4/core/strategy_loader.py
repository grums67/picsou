"""Picsou v4 — Strategy interface and dynamic loader.

Strategies are Python files in v4/strategies/ that the LLM creates.
Each MUST implement the IAStrategy interface:
  - signal(market_data, portfolio, memory) → dict
  - metadata() → dict

The loader discovers, validates, and loads them dynamically.
"""

import importlib.util
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Interface contract ──────────────────────────────────────────────────

STRATEGY_INTERFACE_CODE = '''"""Strategy interface — every strategy MUST implement these methods."""


def signal(market_data: dict, portfolio: dict, memory: dict) -> dict:
    """Analyze market data and return a trading signal.

    Args:
        market_data: dict with keys per symbol, e.g. "BTC-USDT": {candles: [...], ticker: {...}}
        portfolio: dict with balance, positions, pnl
        memory: dict with recent trades, lessons, observations

    Returns:
        dict with keys:
            action: "buy" | "sell" | "hold"
            symbol: str (e.g. "BTC")
            confidence: float 0.0-1.0
            size_pct: float 0.0-1.0 (% of balance to use)
            reasoning: str (why this signal)
    """
    raise NotImplementedError("Every strategy must implement signal()")


def metadata() -> dict:
    """Return strategy metadata.

    Returns:
        dict with keys:
            name: str (unique identifier)
            version: str
            type: str (momentum, mean_reversion, breakout, etc.)
            description: str
            created_by: str ("llm" or "human")
    """
    return {
        "name": "template",
        "version": "0.1",
        "type": "generic",
        "description": "Template strategy",
        "created_by": "human",
    }
'''


class StrategyValidationError(Exception):
    """Raised when a strategy file doesn't implement the interface correctly."""
    pass


class StrategyLoader:
    """Discovers, validates, and loads strategy modules dynamically."""

    def __init__(self, strategies_path: Path):
        self.strategies_path = strategies_path
        self.strategies_path.mkdir(parents=True, exist_ok=True)
        self._loaded: Dict[str, Any] = {}  # name → module

    def discover(self) -> List[str]:
        """Find all .py strategy files in the strategies directory."""
        return sorted([
            f.stem for f in self.strategies_path.glob("*.py")
            if f.stem != "__init__" and not f.stem.startswith("_")
        ])

    def load(self, name: str) -> Optional[Any]:
        """Load a strategy module by name. Returns the module or None on error."""
        if name in self._loaded:
            return self._loaded[name]

        filepath = self.strategies_path / f"{name}.py"
        if not filepath.exists():
            logger.error("Strategy file not found: %s", filepath)
            return None

        try:
            spec = importlib.util.spec_from_file_location(f"strategy_{name}", str(filepath))
            if spec is None or spec.loader is None:
                logger.error("Cannot create module spec for %s", name)
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Validate interface
            if not hasattr(module, 'signal'):
                raise StrategyValidationError(f"{name} missing signal() function")
            if not hasattr(module, 'metadata'):
                raise StrategyValidationError(f"{name} missing metadata() function")

            # Call metadata to get info
            meta = module.metadata()
            if not isinstance(meta, dict) or 'name' not in meta:
                raise StrategyValidationError(f"{name} metadata() must return dict with 'name'")

            self._loaded[name] = module
            logger.info("Loaded strategy: %s v%s (%s)", meta.get('name'), meta.get('version', '?'), meta.get('type', '?'))
            return module

        except StrategyValidationError as e:
            logger.error("Strategy validation failed for %s: %s", name, e)
            return None
        except Exception as e:
            logger.error("Failed to load strategy %s: %s\n%s", name, e, traceback.format_exc())
            return None

    def reload(self, name: str) -> Optional[Any]:
        """Force reload a strategy (discard cached version)."""
        if name in self._loaded:
            del self._loaded[name]
        return self.load(name)

    def run_signal(self, name: str, market_data: Dict, portfolio: Dict,
                   memory: Dict) -> Optional[Dict]:
        """Run a strategy's signal() function safely. Returns result or None."""
        module = self.load(name)
        if module is None:
            return None

        try:
            result = module.signal(market_data, portfolio, memory)
            if not isinstance(result, dict):
                logger.warning("Strategy %s signal() returned non-dict: %s", name, type(result))
                return None
            if 'action' not in result:
                logger.warning("Strategy %s signal() missing 'action' key", name)
                return None
            return result
        except Exception as e:
            logger.error("Strategy %s signal() error: %s", name, e)
            return None

    def get_metadata(self, name: str) -> Optional[Dict]:
        """Get strategy metadata."""
        module = self.load(name)
        if module is None:
            return None
        try:
            return module.metadata()
        except Exception as e:
            logger.error("Strategy %s metadata() error: %s", name, e)
            return None

    def validate_code(self, code: str) -> tuple:
        """Validate strategy code before writing to disk.

        Returns (is_valid, error_message).
        """
        try:
            compile(code, '<strategy>', 'exec')
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Check for required functions
        if 'def signal(' not in code:
            return False, "Missing signal() function"
        if 'def metadata(' not in code:
            return False, "Missing metadata() function"

        # Check for dangerous operations
        dangerous = ['import os', 'import subprocess', 'import shutil',
                      '__import__', 'exec(', 'eval(', 'open(',
                      'import sys', 'os.system', 'os.remove']
        for pattern in dangerous:
            if pattern in code:
                return False, f"Dangerous operation detected: {pattern}"

        return True, ""

    def write_strategy(self, name: str, code: str) -> bool:
        """Write a strategy to disk after validation.

        Returns True if written successfully.
        """
        is_valid, error = self.validate_code(code)
        if not is_valid:
            logger.error("Strategy validation failed: %s", error)
            return False

        filepath = self.strategies_path / f"{name}.py"
        filepath.write_text(code)
        logger.info("Wrote strategy: %s", filepath)

        # Force reload
        if name in self._loaded:
            del self._loaded[name]

        return True