"""Picsou v4 — Main entry point.

Two loops running in sync:
  - Heartbeat (fast, every 5 min): runs active strategies, no LLM
  - Brain (slow, every ~1h): LLM analyzes, creates/modifies strategies

Also runs a lightweight HTTP health endpoint on port 3035.

Usage:
  python run.py                    # Run both loops forever
  python run.py --heartbeat-only   # Run only heartbeat
  python run.py --brain-only       # Run one brain cycle and exit
  python run.py --test             # Quick test of all components
"""

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from threading import Thread

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import PicsouConfig, get_config
from core.memory import Memory
from core.portfolio import Portfolio
from core.heartbeat import Heartbeat
from core.brain_loop import BrainLoop
from core.strategy_loader import StrategyLoader
from core.exchanges.okx import OKXExchange

logger = logging.getLogger("picsou_v4")

# ── Global state for health endpoint ──────────────────────────────────
AGENT_STATE = {
    "status": "starting",
    "cycle": 0,
    "balance": 0,
    "pnl_pct": 0,
    "last_heartbeat": None,
    "last_brain": None,
    "started_at": None,
    "pid": os.getpid(),
}


def write_pid(data_path: Path):
    """Write PID file for watchdog."""
    pid_path = data_path / "picsou.pid"
    pid_path.write_text(str(os.getpid()))
    logger.info("PID %d written to %s", os.getpid(), pid_path)


def start_health_server(port=3035):
    """Lightweight HTTP health endpoint — same port as v3."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json

    class HealthHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Suppress access logs

        def do_GET(self):
            if self.path in ("/api/health", "/api/status", "/"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                payload = json.dumps(AGENT_STATE, default=str)
                self.wfile.write(payload.encode())
            else:
                self.send_response(404)
                self.end_headers()

    server = HTTPServer(("127.0.0.1", port), HealthHandler)
    logger.info("Health endpoint on port %d", port)
    server.serve_forever()


def setup_logging(data_path: Path):
    """Configure logging."""
    data_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(data_path / "picsou_v4.log"),
        ],
    )


def init_exchanges(config: PicsouConfig):
    """Initialize exchange adapters."""
    exchanges = {}
    for name, exc_cfg in config.exchanges.items():
        if name == "okx":
            exchanges[name] = OKXExchange(
                rest_url=exc_cfg.rest_url,
                fee_rate=exc_cfg.fee_rate,
            )
    return exchanges


def run_brain_loop(config: PicsouConfig):
    """Run combined heartbeat + brain loop forever."""
    global AGENT_STATE

    exchanges = init_exchanges(config)
    memory = Memory(config.data_path / "picsou.db")
    portfolio = Portfolio(config.starting_capital, memory=memory)
    heartbeat = Heartbeat(config, portfolio, memory, exchanges)
    brain_loop = BrainLoop(config, portfolio, memory, exchanges)

    heartbeat_cycle = 0
    interval = config.safety.cooldown_seconds

    AGENT_STATE["status"] = "running"
    AGENT_STATE["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    logger.info("Starting Picsou v4 main loop (heartbeat=%ds, brain every %d cycles)",
                interval, config.safety.brain_interval_cycles)

    # Graceful shutdown on SIGTERM/SIGINT
    def shutdown(signum, frame):
        logger.info("Received signal %s — shutting down gracefully", signum)
        AGENT_STATE["status"] = "stopping"
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    while True:
        heartbeat_cycle += 1
        AGENT_STATE["cycle"] = heartbeat_cycle

        # ── Heartbeat (every cycle) ──
        try:
            summary = heartbeat.run_once()
            AGENT_STATE["balance"] = summary.get("balance", 0)
            AGENT_STATE["pnl_pct"] = summary.get("pnl", {}).get("return_pct", 0)
            AGENT_STATE["last_heartbeat"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            logger.info("Heartbeat #%d: balance=$%.2f, PnL=%.2f%%, trades=%d",
                        heartbeat_cycle, summary.get("balance", 0),
                        summary.get("pnl", {}).get("return_pct", 0),
                        summary.get("trades_executed", 0))
        except Exception as e:
            logger.error("Heartbeat error: %s", e, exc_info=True)

        # ── Brain (every N cycles) ──
        if heartbeat_cycle % config.safety.brain_interval_cycles == 0:
            try:
                result = brain_loop.run_once()
                AGENT_STATE["last_brain"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                logger.info("Brain cycle result: %s", result)
            except Exception as e:
                logger.error("Brain error: %s", e, exc_info=True)

        time.sleep(interval)


def run_test(config: PicsouConfig):
    """Quick test of all components."""
    print("=" * 50)
    print("Picsou v4 — Component Test")
    print("=" * 50)

    # 1. Memory
    print("\n[1/6] Testing Memory (SQLite)...")
    memory = Memory(config.data_path / "picsou.db")
    memory.add_observation("test", "Memory working")
    memory.add_lesson("Test lesson: always check your data")
    ctx = memory.get_context_for_llm()
    print(f"  ✓ Memory OK — {len(ctx)} context keys")

    # 2. Portfolio
    print("\n[2/6] Testing Portfolio...")
    portfolio = Portfolio(config.starting_capital)
    state = portfolio.get_state()
    print(f"  ✓ Portfolio OK — balance=${state['balance']:,.2f}")

    # 3. Safety
    print("\n[3/6] Testing Safety...")
    from core.safety import Safety
    safety = Safety(config)
    check = safety.check_trade("buy", 100, 10000, 0, 10000, 0.0)
    print(f"  ✓ Safety OK — trade allowed: {check.allowed}")

    # 4. Exchanges
    print("\n[4/6] Testing Exchange (OKX)...")
    exchanges = init_exchanges(config)
    if "okx" in exchanges:
        ticker = exchanges["okx"].get_ticker("BTC-USDT")
        if ticker and ticker.get("last"):
            print(f"  ✓ OKX OK — BTC=${ticker['last']:,.2f}")
        else:
            print(f"  ⚠ OKX returned no data (may be rate limited)")

    # 5. Strategy Loader
    print("\n[5/6] Testing Strategy Loader...")
    loader = StrategyLoader(config.strategies_path)
    strategies = loader.discover()
    print(f"  ✓ Strategy Loader OK — {len(strategies)} strategies found: {strategies}")

    # 6. Brain
    print("\n[6/6] Testing Brain (LLM connection)...")
    brain_loop = BrainLoop(config, portfolio, memory, exchanges)
    print(f"  ✓ Brain OK — model={config.llm.model}")

    print("\n" + "=" * 50)
    print("All components tested. Ready to run.")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Picsou v4 — Autonomous Crypto Agent")
    parser.add_argument("--heartbeat-only", action="store_true", help="Run only heartbeat (no LLM)")
    parser.add_argument("--brain-only", action="store_true", help="Run one brain cycle and exit")
    parser.add_argument("--test", action="store_true", help="Test all components")
    parser.add_argument("--port", type=int, default=3035, help="Health endpoint port")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()

    config = get_config()
    setup_logging(config.data_path)
    write_pid(config.data_path)

    logger.info("Picsou v4 starting — phase=%s, capital=$%.0f, symbols=%s",
                config.phase, config.starting_capital, config.symbols)

    if args.test:
        run_test(config)
    elif args.heartbeat_only:
        # Still start health server
        t = Thread(target=start_health_server, args=(args.port,), daemon=True)
        t.start()
        exchanges = init_exchanges(config)
        memory = Memory(config.data_path / "picsou.db")
        portfolio = Portfolio(config.starting_capital, memory=memory)
        heartbeat = Heartbeat(config, portfolio, memory, exchanges)
        AGENT_STATE["status"] = "heartbeat_only"
        while True:
            try:
                heartbeat.run_once()
            except Exception as e:
                logger.error("Heartbeat error: %s", e, exc_info=True)
            time.sleep(config.safety.cooldown_seconds)
    elif args.brain_only:
        exchanges = init_exchanges(config)
        memory = Memory(config.data_path / "picsou.db")
        portfolio = Portfolio(config.starting_capital, memory=memory)
        brain = BrainLoop(config, portfolio, memory, exchanges)
        result = brain.run_once()
        print(f"Brain result: {result}")
    else:
        # Start health server in background
        t = Thread(target=start_health_server, args=(args.port,), daemon=True)
        t.start()
        run_brain_loop(config)


if __name__ == "__main__":
    main()