# Picsou v4 — Autonomous Crypto Trading Agent

## Architecture

Two-speed loop:
- **Heartbeat** (every 5 min): runs active strategies deterministically, no LLM
- **Brain** (every ~1 hour): LLM analyzes, creates/modifies strategies, adjusts weights

## Core Principles

1. **L'agent écrit son propre code** — strategies are Python files created by the LLM
2. **Rien ne meurt** — strategies go dormant, never eliminated
3. **Le LLM est le cerveau** — it decides, creates, modifies
4. **Sécurités hard** — max 20% per position, max 5 positions, circuit breaker at 20% drawdown

## Structure

```
v4/
  core/
    config.py         — Configuration
    memory.py          — SQLite persistent memory
    safety.py          — Hard safety limits (circuit breakers)
    portfolio.py       — Paper trading portfolio manager
    observer.py        — Market data collection
    executor.py        — Trade execution with safety
    brain.py           — LLM brain with function calling
    brain_loop.py      — Slow loop (LLM decisions)
    heartbeat.py       — Fast loop (strategy execution)
    strategy_loader.py — Dynamic strategy loading
    backtest.py        — Immutable backtest engine
    system_prompt.py   — LLM personality prompt
  strategies/          — Strategy files (LLM creates new ones)
    ema_crossover_v1.py — Starter strategy
  data/                — SQLite DB, logs
  run.py               — Entry point
```

## Run

```bash
python run.py               # Both loops
python run.py --test        # Test all components
python run.py --heartbeat-only  # Fast loop only (no LLM)
python run.py --brain-only  # One brain cycle
```