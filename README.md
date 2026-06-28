# 🪙 Picsou — Autonomous Crypto Trading Agent

**Self-learning paper-trading agent** that uses an LLM brain, live market data, sentiment analysis, and web-sourced strategy research to trade cryptocurrencies autonomously.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Picsou Agent                     │
│                  (run_once loop)                  │
├─────────┬──────────┬──────────┬─────────────────┤
│  Market │   LLM    │ Learning │   Research       │
│  Data   │  Brain   │  Engine  │   Insights       │
│ (3 exch)│(Kimi K2) │(weights) │  (DuckDuckGo +   │
│         │          │          │   CoinGecko +     │
│         │          │          │   CryptoCompare)  │
├─────────┴──────────┴──────────┴─────────────────┤
│           Portfolio Manager (paper trading)       │
│           Decision Journal (JSONL)                │
│           Dashboard (FastAPI + PWA)               │
└──────────────────────────────────────────────────┘
```

## How It Works

### Trading Cycle (every 15 min)

1. **Fetch market data** — BTC, ETH, SOL prices, candles, order books from OKX, Kraken, Bitstamp
2. **Fetch sentiment** — Fear & Greed Index + crypto headlines
3. **Fetch research insights** — DuckDuckGo strategy search + CoinGecko trending + CryptoCompare analysis
4. **LLM decision** — Brain analyzes everything and outputs structured buy/sell/hold decisions with strategy type and reasoning
5. **Execute trades** — Portfolio manager opens/closes positions (paper trading)
6. **Learn** — Evaluate strategy performance, adjust weights, eliminate underperformers

### Learning System

Picsou evaluates **6 strategy categories** the LLM assigns to its own decisions:

| Strategy | Description |
|---|---|
| `momentum` | Trend-following trades |
| `mean_reversion` | Trading against overextended moves |
| `breakout` | Trading breakouts from ranges |
| `contrarian` | Counter-trend at extremes |
| `dca` | Dollar-cost averaging entries |
| `risk_management` | Hold/defensive decisions |

**Adaptive weights:**
- Strategies with **win rate ≥ 55%** and **Sharpe > 0.5** get promoted (weight increased)
- Strategies with **win rate < 50%** or **max drawdown > 30%** get eliminated
- Weights are normalized so active strategies sum to 1.0
- Current weights are injected into the LLM prompt to influence future decisions

**Exploration phase:**
- New strategies start untested — Picsou forces small trades (2% capital) to collect data
- Strategies with < 3 trades are **protected from elimination**
- Exploration auto-disables once all strategies have enough data

### Research Insights (Web-Sourced)

Picsou searches the web for current strategy intelligence:

- **DuckDuckGo** — Crypto trading strategy search results
- **CoinGecko Trending** — Currently trending coins and market momentum
- **CryptoCompare News** — Headlines with technical analysis signals

Extracted insights (trending strategies, technical signals, risk factors) are injected into the LLM prompt as supplementary context. Results are cached for 30 minutes.

### Fallback

If the LLM is unavailable, Picsou falls back to **EMA9/EMA21 crossover signals** on available market data.

## Configuration

Key parameters in `src/config.py`:

```python
# Trading
phase = "learning"              # "learning" (paper) or "live"
starting_capital = 10000.0      # Paper trading capital (10x real)
symbols = ["BTC", "ETH", "SOL"]
loop_interval = 300             # Seconds between cycles

# Risk
max_position_pct = 0.20         # Max 20% capital per position
max_open_positions = 5
max_drawdown_pct = 0.20         # Pause at 20% drawdown

# Learning
min_trades = 5                  # Min trades before evaluating a strategy
min_exploration_trades = 3      # Min trades before a strategy can be eliminated
exploration_phase = True         # Force exploration of untested strategies
exploration_position_pct = 0.02  # 2% position for exploration trades

# Research
research_enabled = True
research_cache_ttl = 1800       # 30 min cache
research_max_sources = 5
```

## Exchanges

| Exchange | Symbol Format | Fee |
|---|---|---|
| OKX | `BTC-USDT` | 0.08% |
| Kraken | `XBTUSDT` | 0.26% |
| Bitstamp | `BTCUSDT` | 0.25% |

All MiCA-compliant. API keys go in `.env` (trade-only, no withdrawal).

## Dashboard

Real-time PWA dashboard at `http://localhost:3037`:

- Portfolio value, P&L, positions
- Decision journal with reasoning
- Strategy weights and win rates
- LLM model status
- Kill switch

Service: `systemctl status picsou-dashboard`

## Project Structure

```
picsou/
├── src/
│   ├── picsou.py              # Main agent loop
│   ├── brain.py               # LLM decision engine
│   ├── config.py              # Configuration
│   ├── portfolio.py           # Portfolio manager
│   ├── journal.py             # Decision journal
│   ├── learning.py            # Learning engine (adaptive weights)
│   ├── strategy_researcher.py # Web strategy research
│   ├── exchanges/
│   │   ├── base.py            # Exchange interface
│   │   ├── okx.py
│   │   ├── kraken.py
│   │   └── bitstamp.py
│   └── strategies/
│       ├── base.py             # Strategy interface
│       ├── momentum.py
│       ├── mean_reversion.py
│       ├── grid.py
│       └── dca.py
├── dashboard/
│   ├── app.py                 # FastAPI server
│   ├── templates/
│   └── static/
├── data/                      # Runtime data (gitignored)
│   ├── portfolio.json
│   ├── journal.jsonl
│   ├── learning.json
│   ├── brain_status.json
│   └── research_cache.json
├── run_agent.py               # CLI runner
├── run.sh                     # Shell runner
└── requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure .env with API keys
cp .env.example .env

# Run one cycle (test)
python run_agent.py

# Run continuously
bash run.sh
```

## Safety

- **Paper trading by default** — no real money at risk in learning phase
- **Kill switch** — `picsou stop` via Telegram or dashboard
- **Max drawdown 20%** — trading pauses automatically
- **No withdrawal keys** — API keys are trade-only
- **All data gitignored** — no secrets or portfolio data in git

## License

Private project — all rights reserved.