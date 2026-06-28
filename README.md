# 🪙 Picsou — Agent de Trading Crypto Autonome

**Agent de paper-trading auto-apprenant** qui utilise un cerveau LLM, des données de marché en temps réel, l'analyse de sentiment et la recherche web de stratégies pour trader les cryptomonnaies en autonomie.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  Agent Picsou                      │
│                (boucle run_once)                   │
├──────────┬──────────┬──────────┬─────────────────┤
│  Marché  │  Cerveau │ Apprent. │   Recherche      │
│  (3 exh) │  (LLM)   │  (poids) │   web             │
│          │          │          │  DuckDuckGo +     │
│          │          │          │  CoinGecko +       │
│          │          │          │  CryptoCompare    │
├──────────┴──────────┴──────────┴─────────────────┤
│         Gestionnaire de Portfolio (paper trading) │
│         Journal de Décisions (JSONL)              │
│         Dashboard (FastAPI + PWA)                 │
└──────────────────────────────────────────────────┘
```

## Comment ça marche

### Cycle de trading (toutes les 15 min)

1. **Récupérer les données marché** — Prix BTC, ETH, SOL, chandeliers et carnets d'ordres depuis OKX, Kraken, Bitstamp
2. **Récupérer le sentiment** — Fear & Greed Index + titres crypto
3. **Rechercher des insights** — Recherche DuckDuckGo + tendances CoinGecko + analyses CryptoCompare
4. **Décision LLM** — Le cerveau analyse l'ensemble et produit des décisions structurées achat/vente/hold avec type de stratégie et raisonnement
5. **Exécuter les trades** — Ouverture/fermeture de positions (paper trading)
6. **Apprendre** — Évaluer les performances par stratégie, ajuster les poids, éliminer les sous-performantes

### Système d'apprentissage

Picsou évalue **6 catégories de stratégies** que le LLM assigne à ses propres décisions :

| Stratégie | Description |
|---|---|
| `momentum` | Suivi de tendance |
| `mean_reversion` | Contre les mouvements surextendus |
| `breakout` | Cassure de ranges |
| `contrarian` | Contre-tendance aux extrêmes |
| `dca` | Entrées en moyenne à cours réduit |
| `risk_management` | Décisions défensives / hold |

**Poids adaptatifs :**
- Les stratégies avec **win rate ≥ 55%** et **Sharpe > 0.5** sont promues (poids augmenté)
- Les stratégies avec **win rate < 50%** ou **drawdown max > 30%** sont éliminées
- Les poids sont normalisés pour sommer à 1.0
- Les poids actuels sont injectés dans le prompt LLM pour influencer les décisions futures

**Phase d'exploration :**
- Les nouvelles stratégies non testées reçoivent des petits trades forcés (2% du capital) pour collecter des données
- Les stratégies avec < 3 trades sont **protégées contre l'élimination**
- L'exploration se désactive automatiquement quand toutes les stratégies ont suffisamment de données

### Recherche d'insights (source web)

Picsou cherche des intelligences de stratégie en ligne :

- **DuckDuckGo** — Résultats de recherche sur les stratégies crypto
- **CoinGecko Trending** — Coins en vogue et dynamique de marché
- **CryptoCompare News** — Titres avec signaux d'analyse technique

Les insights extraits (stratégies tendance, signaux techniques, facteurs de risque) sont injectés dans le prompt LLM comme contexte supplémentaire. Résultats mis en cache pendant 30 minutes.

### Fallback

Si le LLM est indisponible, Picsou utilise un **signal de croisement EMA9/EMA21** sur les données de marché disponibles.

## Configuration

Paramètres clés dans `src/config.py` :

```python
# Trading
phase = "learning"              # "learning" (paper) ou "live"
starting_capital = 10000.0      # Capital paper trading (10x le réel)
symbols = ["BTC", "ETH", "SOL"]
loop_interval = 300             # Secondes entre les cycles

# Risque
max_position_pct = 0.20         # Max 20% du capital par position
max_open_positions = 5          # Max 5 positions simultanées
max_drawdown_pct = 0.20         # Pause à 20% de drawdown

# Apprentissage
min_trades = 5                  # Trades min avant d'évaluer une stratégie
min_exploration_trades = 3      # Trades min avant qu'une stratégie puisse être éliminée
exploration_phase = True        # Forcer l'exploration des stratégies non testées
exploration_position_pct = 0.02 # 2% de position pour les trades d'exploration

# Recherche
research_enabled = True
research_cache_ttl = 1800       # Cache 30 min
research_max_sources = 5
```

## Exchanges

| Exchange | Format symbole | Frais |
|---|---|---|
| OKX | `BTC-USDT` | 0.08% |
| Kraken | `XBTUSDT` | 0.26% |
| Bitstamp | `BTCUSDT` | 0.25% |

Tous conformes MiCA. Clés API dans `.env` (trade uniquement, pas de retrait).

## Dashboard

Dashboard PWA temps réel sur `http://localhost:3037` :

- Valeur du portefeuille, P&L, positions
- Journal des décisions avec raisonnement
- Poids des stratégies et taux de réussite
- Statut du modèle LLM
- Kill switch d'urgence

Service : `systemctl status picsou-dashboard`

## Structure du projet

```
picsou/
├── src/
│   ├── picsou.py              # Boucle principale de l'agent
│   ├── brain.py               # Moteur de décision LLM
│   ├── config.py              # Configuration
│   ├── portfolio.py           # Gestionnaire de portefeuille
│   ├── journal.py             # Journal de décisions
│   ├── learning.py            # Moteur d'apprentissage (poids adaptatifs)
│   ├── strategy_researcher.py # Recherche web de stratégies
│   ├── exchanges/
│   │   ├── base.py            # Interface exchange
│   │   ├── okx.py
│   │   ├── kraken.py
│   │   └── bitstamp.py
│   └── strategies/
│       ├── base.py             # Interface stratégie
│       ├── momentum.py
│       ├── mean_reversion.py
│       ├── grid.py
│       └── dca.py
├── dashboard/
│   ├── app.py                 # Serveur FastAPI
│   ├── templates/
│   └── static/
├── data/                      # Données runtime (gitignored)
│   ├── portfolio.json
│   ├── journal.jsonl
│   ├── learning.json
│   ├── brain_status.json
│   └── research_cache.json
├── run_agent.py               # Lanceur CLI
├── run.sh                     # Lanceur shell
└── requirements.txt
```

## Démarrage rapide

```bash
# Installer les dépendances
pip install -r requirements.txt

# Configurer les clés API dans .env
cp .env.example .env

# Lancer un cycle (test)
python run_agent.py

# Lancer en continu
bash run.sh
```

## Sécurité

- **Paper trading par défaut** — aucun argent réel en phase d'apprentissage
- **Kill switch** — `picsou stop` via Telegram ou dashboard
- **Drawdown max 20%** — le trading se met en pause automatiquement
- **Pas de clés de retrait** — les clés API sont trade uniquement
- **Données gitignored** — aucun secret ni données de portfolio dans git

## Licence

Projet privé — tous droits réservés.