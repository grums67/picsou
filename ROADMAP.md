# Picsou v4 — Agent Autonome de Trading Crypto

## Philosophie

> Un agent qui n'a qu'un seul objectif : **apprendre seul à gagner de l'argent avec les cryptomonnaies**.
> On lui donne les outils de base, un objectif, et on le laisse expérimenter, créer, modifier, évoluer.
> Aucune stratégie codée en dur. Aucune élimination permanente. L'agent est libre d'inventer.

---

## Principes fondateurs

1. **L'agent écrit son propre code** — Les stratégies sont des fichiers Python générés par le LLM, pas des classes prédéfinies
2. **Rien ne meurt** — Les stratégies dorment, jamais éliminées. Un ratio nul aujourd'hui peut redevenir pertinent demain
3. **Le LLM est le cerveau** — Pas de règles rigides codées en Python pour les décisions. Le LLM observe, raisonne, décide
4. **Apprentissage par l'expérience** — Chaque trade est journalisé, chaque cycle enrichit la mémoire. L'agent s'améliore continuellement
5. **Auto-modification** — L'agent peut modifier ses propres prompts, ses stratégies, ses paramètres de risque

---

## Architecture cible

```
┌─────────────────────────────────────────────────┐
│                  BOUCLE PRINCIPALE               │
│                                                  │
│  1. OBSERVE  ─→ marché, portefeuille, mémoire   │
│  2. RÉFLÉCHIT ─→ LLM analyse tout le contexte   │
│  3. DÉCIDE   ─→ buy/sell/hold + taille position  │
│  4. EXÉCUTE  ─→ ordres sur exchange              │
│  5. APPREND  ─→ journal, métriques, reflet      │
│  6. ÉVOLUE   ─→ crée/modifie stratégies & code   │
│                                                  │
│  ──── puis on recommence ────                    │
└─────────────────────────────────────────────────┘
```

### Modules

| Module | Rôle | Détail |
|--------|------|--------|
| **OBSERVE** | Collecte données | Prix OHLCV, ordres, sentiment F&G, RSS, indicateurs macro |
| **MÉMOIRE** | Persistance inter-cycles | Journal des trades, scores par stratégie, contexte macro récent, règles auto-générées |
| **CERVEAU** | LLM décide | Analyse tout le contexte, produit des décisions structurées |
| **PORTEFEUILLE** | Gestion positions | Open/close/track, PnL, drawdown |
| **EXÉCUTION** | Passer d'ordres | API exchange (OKX, Kraken, Bitstamp) |
| **APPRENTISSAGE** | Mesurer performance | Win rate, Sharpe, drawdown par stratégie — poids ajustés, jamais éliminés |
| **ÉVOLUTION** | Auto-modification | LLM écrit/modifie des fichiers stratégie Python, backteste, intègre |
| **SANTÉ** | Monitoring | Dashboard, health endpoint, alertes |

---

## Les "outils de base" qu'on fournit

L'agent reçoit un kit minimal. C'est à lui de l'utiliser et de l'enrichir.

### Outils fournis (ne pas modifier)

- **Données marché** : prix OHLCV multi-exchange, ticker, orderbook
- **Sentiment** : Fear & Greed Index, RSS crypto, CoinGecko trending
- **Indicateurs techniques** : EMA, RSI, MACD, Bollinger, volume (bibliothèque dispo, pas forcément utilisée)
- **Portefeuille** : suivi positions, PnL, balance
- **Exécution** : passer des ordres buy/sell sur les exchanges configurés
- **Journal** : logger chaque décision avec contexte complet
- **Backtest** : rejouer une stratégie sur données historiques
- **Exécution de code** : sandbox pour créer/tester des scripts Python

### Ce que l'agent crée lui-même

- **Stratégies** : fichiers Python dans `strategies/` — générés par le LLM, chargés dynamiquement
- **Règles d'adaptation** : modifications de ses propres paramètres de risque, taille de position
- **Prompts** : le LLM peut modifier son propre prompt système en fonction de ce qu'il apprend
- **Nouveaux indicateurs** : s'il veut un indicateur qui n'existe pas, il l'écrit

---

## Format d'une stratégie (générée par l'agent)

```python
# strategies/momentum_v2.py
# Créé par Picsou le 2026-06-29
# Basé sur l'observation que les pump-and-pull SOL suivent souvent 
# un pattern de volume croissant sur 3 bougies 1h

def signal(market_data, portfolio, memory):
    """
    Retourne une décision : {'action': 'buy'|'sell'|'hold', 
                              'symbol': 'BTC', 
                              'confidence': 0.0-1.0,
                              'size_pct': 0.0-1.0,
                              'reasoning': '...'}
    """
    candles = market_data.get('SOL-USDT', {}).get('candles', [])
    if len(candles) < 5:
        return {'action': 'hold', 'reasoning': 'pas assez de données'}
    
    # Logique créée par l'agent...
    recent_volume = sum(c['volume'] for c in candles[-3:])
    avg_volume = sum(c['volume'] for c in candles[-20:]) / 20
    
    if recent_volume / 3 > avg_volume * 1.5:
        return {
            'action': 'buy',
            'symbol': 'SOL',
            'confidence': 0.6,
            'size_pct': 0.05,
            'reasoning': 'Volume spike détecté, momentum haussier probable'
        }
    
    return {'action': 'hold', 'reasoning': 'Pas de signal'}

def metadata():
    return {
        'name': 'momentum_v2',
        'created': '2026-06-29',
        'parent': 'momentum_v1',  # si évolution d'une stratégie existante
        'type': 'momentum',
        'description': 'Détection de spikes de volume sur 3 bougies'
    }
```

---

## Cycle de vie d'une stratégie

```
CRÉATION ─→ PROBATION (poids minimal) ─→ APPRENTISSAGE (poids ajusté par performance)
                ↑                                    │
                │                                    ↓
                └──── DORMANCE (poids ≈ 0, mais vivante) ←─ PERFORMANCE FAIBLE
                
RÉVEIL AUTOMATIQUE quand :
  - Les conditions de marché correspondent au type de stratégie
  - Assez de cycles sont passés depuis la dormance (ex: 48h minimum)
  - D'autres stratégies du même type performent mal → opportunité
```

**JAMAIS d'élimination.** Une stratégie dormante est une stratégie en attente.

---

## Mémoire de l'agent

Pas un simple fichier de scores. Une mémoire riche que le LLM lit chaque cycle :

```json
{
  "portfolio_snapshot": { "balance": 9847, "pnl_pct": -1.5, "positions": [...] },
  "strategy_performance": {
    "momentum_v2": { "trades": 15, "win_rate": 0.47, "sharpe": 0.3, 
                     "weight": 0.25, "status": "active", "dormant_since": null },
    "mean_revert_v1": { "trades": 8, "win_rate": 0.12, "sharpe": -1.2,
                        "weight": 0.02, "status": "dormant", "dormant_since": "2026-06-29" }
  },
  "recent_observations": [
    "SOL volume spike non capturé par les stratégies actuelles",
    "BTC range étroit depuis 12h — mean reversion pourrait fonctionner"
  ],
  "self_generated_rules": [
    "Quand F&G < 15, réduire tailles de position de 50%",
    "Ne jamais ouvrir plus de 3 positions simultanées en marché range"
  ],
  "lessons_learned": [
    "Les stratégies momentum perdent en marché range → adapter",
    "Le DCA simple ne fonctionne pas en bear → essayer DCA conditionnel"
  ]
}
```

---

## Phases de développement

### Phase 1 — Fondations propres (reset + architecture)
- [ ] Nettoyer le repo : supprimer les stratégies hardcoded
- [ ] Nouveau module `observe.py` — collecte de données marché/sentiment
- [ ] Nouveau module `memory.py` — mémoire riche, JSON, lue par le LLM
- [ ] Nouveau module `portfolio.py` — gestion positions/PnL (garder l'existant, améliorer)
- [ ] Nouveau module `execution.py` — passage d'ordres (garder l'existant)
- [ ] Nouveau `brain.py` — prompt système minimal, le LLM décide de tout
- [ ] Boucle principale minimale : observe → think → act → learn

### Phase 2 — Apprentissage adaptatif
- [ ] Module `learning.py` v4 — dormancy jamais élimination, auto-réveil
- [ ] Poids dynamiques basés sur performance récente (fenêtre glissante)
- [ ] Journal enrichi : chaque décision = contexte complet + résultat a posteriori
- [ ] Auto-évaluation : l'agent note lui-même la qualité de ses décisions passées

### Phase 3 — Création de stratégies
- [ ] Module `evolution.py` — le LLM peut écrire des fichiers Python
- [ ] Sandbox d'exécution : backtester une stratégie avant déploiement
- [ ] Chargement dynamique : `importlib` des stratégies créées
- [ ] Cycle de création : observer un pattern → écrire une stratégie → backtester → déployer en probation

### Phase 4 — Auto-modification
- [ ] L'agent modifie son propre prompt système
- [ ] L'agent ajuste ses paramètres de risque en fonction de l'expérience
- [ ] L'agent crée de nouveaux indicateurs techniques si besoin
- [ ] Mémoire à long terme : leçons persistantes entre sessions

### Phase 5 — Robustesse & Monitoring
- [ ] Dashboard temps réel (garder le dashboard existant)
- [ ] Alertes Telegram en cas de comportement anormal
- [ ] Limites de sécurité : max drawdown hard, circuit breaker
- [ ] Sauvegardes automatiques de l'état

---

## Ce qui change fondamentalement

| Aspect | Picsou v3 (actuel) | Picsou v4 (cible) |
|--------|-------------------|-------------------|
| Stratégies | 6 hardcoded en Python | Créées par le LLM, fichiers dynamiques |
| Élimination | Stratégies tuées après 3 trades | Jamais. Dormance avec réveil automatique |
| Apprentissage | Règles rigides (WR < 50% → mort) | Poids continus, fenêtre glissante, auto-évaluation |
| Décisions | LLM parmi un menu fixe | LLM libre de raisonner et décider |
| Prompt | Fixe, codé en dur | Évolutif, modifiable par l'agent |
| Code | Statique | L'agent peut écrire/modifier du Python |
| Création | Impossible | Observer → Hypothèse → Code → Backtest → Probation |

---

## Risques et mitigeations

| Risque | Mitigation |
|--------|-----------|
| LLM génère du code cassé | Backtest obligatoire avant déploiement. Syntax check. Probation avec position minime |
| LLM fait n'importe quoi | Circuit breaker hard : max drawdown, max positions, taille max |
| Coût tokens élevé | Cycle 5 min, prompt compact, résumé auto du contexte |
| Stratégies en boucle | Limite de N stratégies actives. Les plus anciennes en dormance si plein |
| LLM hallucine des signaux | Confidence threshold, validation des prix avant exécution |

---

## Objectif mesurable

- **PnL positif** sur 30 jours en paper trading
- **Sharpe ratio > 0** sur la période
- **Max drawdown < 15%**
- L'agent doit avoir créé au moins 2 stratégies lui-même qui performent