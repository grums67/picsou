#!/usr/bin/env python3
"""Picsou Self-Improvement Script

Analyzes trading performance, calls LLM for suggestions, and applies
safe improvements to learning weights and LLM configuration.

Usage: python3 /root/PROJECTS/picsou/picsou_self_improve.py
"""

import json
import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ─── Constants ────────────────────────────────────────────────────────────────
DATA_DIR = Path("/root/PROJECTS/picsou/data")
LEARNING_FILE = DATA_DIR / "learning.json"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
JOURNAL_FILE = DATA_DIR / "journal.jsonl"
LLM_CONFIG_FILE = DATA_DIR / "llm_config.json"

# Safe bounds for adjustments
MAX_WEIGHT_DELTA = 0.10       # max change per run
MIN_WEIGHT = 0.05
MAX_WEIGHT = 0.50
MIN_TEMPERATURE = 0.1
MAX_TEMPERATURE = 0.7
RETRY_DAYS = 7               # days before re-enabling eliminated strategy
MIN_TRADES_FOR_DATA = 10     # minimum trades to consider a strategy has "enough data"

LLM_TIMEOUT = 60  # seconds


def load_json(path: Path) -> Optional[Dict]:
    """Load JSON file, return None on failure."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[WARN] Impossible de charger {path}: {e}", file=sys.stderr)
        return None


def save_json(path: Path, data: Dict) -> bool:
    """Save JSON file atomically."""
    try:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(path)
        return True
    except Exception as e:
        print(f"[ERREUR] Impossible de sauvegarder {path}: {e}", file=sys.stderr)
        return False


def read_last_n_journal_lines(path: Path, n: int = 200) -> List[Dict]:
    """Read the last N lines of the journal JSONL file."""
    entries = []
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines[-n:]:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"[WARN] Journal non trouvé: {path}", file=sys.stderr)
    return entries


def analyze_performance(learning: Dict, portfolio: Dict, journal: List[Dict]) -> Dict:
    """Analyze trading performance from learning, portfolio and journal data."""
    analysis = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pnl": {},
        "strategies": {"active": {}, "eliminated": {}, "underexplored": []},
        "overall": {},
        "journal_summary": {},
    }

    # ─── Portfolio PnL ────────────────────────────────────────────────────
    if portfolio:
        balance = portfolio.get("balance", 0)
        starting = portfolio.get("starting_capital", 10000)
        total_pnl = balance - starting
        pnl_pct = (total_pnl / starting * 100) if starting else 0
        analysis["pnl"] = {
            "balance": round(balance, 2),
            "starting_capital": starting,
            "total_pnl": round(total_pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "open_positions": len(portfolio.get("positions", {})),
            "closed_trades": len(portfolio.get("trades", [])),
        }

        # Compute win rate from portfolio trades
        trades = portfolio.get("trades", [])
        if trades:
            winning = sum(1 for t in trades if t.get("pnl", 0) > 0)
            losing = sum(1 for t in trades if t.get("pnl", 0) < 0)
            total = len(trades)
            avg_pnl = sum(t.get("pnl", 0) for t in trades) / total
            analysis["overall"] = {
                "total_closed_trades": total,
                "winning_trades": winning,
                "losing_trades": losing,
                "win_rate": round(winning / total, 4) if total else 0,
                "avg_pnl_per_trade": round(avg_pnl, 4),
            }

    # ─── Strategy analysis from learning ──────────────────────────────────
    scores = learning.get("scores", {}) if learning else {}
    for name, s in scores.items():
        entry = {
            "win_rate": s.get("win_rate", 0),
            "total_trades": s.get("total_trades", 0),
            "winning_trades": s.get("winning_trades", 0),
            "losing_trades": s.get("losing_trades", 0),
            "avg_profit": s.get("avg_profit", 0),
            "total_profit": s.get("total_profit", 0),
            "max_drawdown": s.get("max_drawdown", 0),
            "sharpe_ratio": s.get("sharpe_ratio", 0),
            "weight": s.get("weight", 0),
        }
        if s.get("active", True):
            analysis["strategies"]["active"][name] = entry
            # Check underexplored
            if s.get("total_trades", 0) < MIN_TRADES_FOR_DATA:
                analysis["strategies"]["underexplored"].append(name)
        else:
            analysis["strategies"]["eliminated"][name] = entry

    # ─── Journal summary ──────────────────────────────────────────────────
    if journal:
        strategies_count: Dict[str, int] = {}
        actions_count: Dict[str, int] = {}
        for entry in journal:
            strat = entry.get("strategy", "unknown")
            action = entry.get("action", "unknown")
            strategies_count[strat] = strategies_count.get(strat, 0) + 1
            actions_count[action] = actions_count.get(action, 0) + 1
        analysis["journal_summary"] = {
            "entries_analyzed": len(journal),
            "strategies_seen": strategies_count,
            "actions_distribution": actions_count,
        }

    return analysis


def call_llm_for_suggestions(llm_config: Dict, analysis: Dict) -> Optional[str]:
    """Call the LLM via Ollama proxy to get improvement suggestions."""
    model = llm_config.get("llm_model", "kimi-k2.6:cloud")
    url = llm_config.get("llm_url", "http://127.0.0.1:11434/v1")
    temperature = min(llm_config.get("llm_temperature", 0.3), MAX_TEMPERATURE)
    max_tokens = llm_config.get("llm_max_tokens", 2048)

    # Build the prompt
    prompt = f"""Tu es un analyste de trading crypto. Voici les performances de l'agent Picsou. 
Analyse les données et propose des améliorations concrètes.

PERFORMANCES GÉNÉRALES:
{json.dumps(analysis.get('overall', {}), indent=2, ensure_ascii=False)}

PnL:
{json.dumps(analysis.get('pnl', {}), indent=2, ensure_ascii=False)}

STRATÉGIES ACTIVES:
{json.dumps(analysis.get('strategies', {}).get('active', {}), indent=2, ensure_ascii=False)}

STRATÉGIES ÉLIMINÉES:
{json.dumps(analysis.get('strategies', {}).get('eliminated', {}), indent=2, ensure_ascii=False)}

STRATÉGIES SOUS-EXPLORÉES:
{json.dumps(analysis.get('strategies', {}).get('underexplored', []), indent=2, ensure_ascii=False)}

RÉSUMÉ JOURNAL:
{json.dumps(analysis.get('journal_summary', {}), indent=2, ensure_ascii=False)}

CONTRAINTES DE SÉCURITÉ:
- Ajustement max de poids: ±0.10 par exécution
- Poids min: 0.05, max: 0.50
- Température: entre 0.1 et 0.7 uniquement
- Stratégies ré-activées: poids initial 0.10
- JAMAIS de trades — analyse uniquement

Propose en JSON:
{{
  "weight_adjustments": {{"stratégie": delta}},
  "temperature_adjustment": delta_float,
  "reenable_strategies": ["stratégie"],
  "disable_exploration": bool,
  "recommendations": ["recommandation en français"]
}}"""

    try:
        response = requests.post(
            f"{url}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Tu es un analyste quantitatif expert en trading crypto. Réponds uniquement en JSON valide."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=LLM_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content
    except requests.exceptions.RequestException as e:
        print(f"[WARN] LLM indisponible: {e}", file=sys.stderr)
        return None
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"[WARN] Réponse LLM invalide: {e}", file=sys.stderr)
        return None


def parse_llm_suggestions(raw: Optional[str]) -> Dict:
    """Parse LLM response into structured suggestions."""
    if not raw:
        return {}
    # Try to extract JSON from the response
    # Look for JSON blocks
    text = raw.strip()
    # Remove markdown code fences if present
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    try:
        suggestions = json.loads(text.strip())
        if isinstance(suggestions, dict):
            return suggestions
    except json.JSONDecodeError:
        pass
    # Try to find any JSON object in the text
    import re
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def apply_improvements(
    learning: Dict,
    llm_config: Dict,
    suggestions: Dict,
    analysis: Dict,
) -> Dict:
    """Apply safe improvements based on analysis and LLM suggestions.

    Returns a dict of changes made.
    """
    changes = {
        "weight_changes": {},
        "temperature_change": None,
        "reenabled": [],
        "exploration_disabled": False,
    }
    now = datetime.now(timezone.utc)

    # ─── 1. Weight adjustments from LLM suggestions ───────────────────────
    weight_adj = suggestions.get("weight_adjustments", {})
    scores = learning.get("scores", {})

    for strat_name, delta in weight_adj.items():
        if strat_name not in scores:
            continue
        s = scores[strat_name]
        if not s.get("active", True):
            continue
        # Clamp delta to safe bounds
        delta = max(-MAX_WEIGHT_DELTA, min(MAX_WEIGHT_DELTA, float(delta)))
        new_weight = s.get("weight", 0.25) + delta
        new_weight = max(MIN_WEIGHT, min(MAX_WEIGHT, new_weight))
        old_weight = s.get("weight", 0.25)
        s["weight"] = round(new_weight, 4)
        changes["weight_changes"][strat_name] = {
            "old": old_weight,
            "new": round(new_weight, 4),
            "delta": round(new_weight - old_weight, 4),
        }

    # ─── 2. Auto weight adjustment based on performance ───────────────────
    for name, s in scores.items():
        if not s.get("active", True):
            continue
        if name in changes["weight_changes"]:
            continue  # Already adjusted by LLM

        win_rate = s.get("win_rate", 0)
        total_trades = s.get("total_trades", 0)
        sharpe = s.get("sharpe_ratio", 0)

        # Only auto-adjust if enough data
        if total_trades < MIN_TRADES_FOR_DATA:
            continue

        # Boost winning strategies, reduce losing ones
        if win_rate >= 0.55 and sharpe > 0:
            delta = min(MAX_WEIGHT_DELTA, 0.05)  # Conservative boost
            new_weight = min(MAX_WEIGHT, s.get("weight", 0.25) + delta)
            old_weight = s.get("weight", 0.25)
            s["weight"] = round(new_weight, 4)
            changes["weight_changes"][name] = {
                "old": old_weight,
                "new": round(new_weight, 4),
                "delta": round(new_weight - old_weight, 4),
            }
        elif win_rate < 0.40 and total_trades >= 5:
            delta = max(-MAX_WEIGHT_DELTA, -0.05)
            new_weight = max(MIN_WEIGHT, s.get("weight", 0.25) + delta)
            old_weight = s.get("weight", 0.25)
            s["weight"] = round(new_weight, 4)
            changes["weight_changes"][name] = {
                "old": old_weight,
                "new": round(new_weight, 4),
                "delta": round(new_weight - old_weight, 4),
            }

    # ─── 3. Re-enable eliminated strategies if enough time has passed ─────
    reenable_list = suggestions.get("reenable_strategies", [])
    for name in reenable_list:
        if name not in scores:
            continue
        s = scores[name]
        if s.get("active", True):
            continue  # Already active
        # Check if enough time has passed (7+ days since last evaluation)
        last_eval = learning.get("last_evaluation")
        if last_eval:
            try:
                last_dt = datetime.fromisoformat(last_eval)
                days_since = (now - last_dt).days
                if days_since < RETRY_DAYS:
                    continue  # Not enough time
            except (ValueError, TypeError):
                pass  # If we can't parse the date, allow re-enable
        # Re-enable with safe weight
        s["active"] = True
        s["weight"] = 0.10
        changes["reenabled"].append(name)

    # ─── 4. Normalize weights ────────────────────────────────────────────
    total_weight = sum(
        s.get("weight", 0) for s in scores.values() if s.get("active", True)
    )
    if total_weight > 0:
        for s in scores.values():
            if s.get("active", True) and s.get("weight", 0) > 0:
                s["weight"] = round(s["weight"] / total_weight, 4)

    # ─── 5. Temperature adjustment ────────────────────────────────────────
    temp_adj = suggestions.get("temperature_adjustment")
    if temp_adj is not None:
        try:
            temp_delta = float(temp_adj)
            old_temp = llm_config.get("llm_temperature", 0.3)
            new_temp = max(MIN_TEMPERATURE, min(MAX_TEMPERATURE, old_temp + temp_delta))
            new_temp = round(new_temp, 2)
            if new_temp != old_temp:
                llm_config["llm_temperature"] = new_temp
                changes["temperature_change"] = {
                    "old": old_temp,
                    "new": new_temp,
                }
        except (ValueError, TypeError):
            pass

    # ─── 6. Disable exploration_phase if all strategies have enough data ──
    all_have_data = True
    for name, s in scores.items():
        if s.get("active", True) and s.get("total_trades", 0) < MIN_TRADES_FOR_DATA:
            all_have_data = False
            break

    if suggestions.get("disable_exploration") or all_have_data:
        # Check if exploration_phase exists in llm_config
        # We don't modify config.py, but we can set it in llm_config.json
        # as a signal for the agent
        if llm_config.get("exploration_phase", True) is not False:
            llm_config["exploration_phase"] = False
            changes["exploration_disabled"] = True

    # Update evaluation count and timestamp
    learning["evaluation_count"] = learning.get("evaluation_count", 0) + 1
    learning["last_evaluation"] = now.isoformat()

    return changes


def generate_french_report(
    analysis: Dict,
    changes: Dict,
    suggestions: Dict,
) -> str:
    """Generate a concise French performance and changes report."""
    lines = []
    lines.append("=" * 60)
    lines.append("  RAPPORT D'AUTO-AMÉLIORATION — PICSOU")
    lines.append(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("=" * 60)

    # ─── PnL ─────────────────────────────────────────────────────────────
    pnl = analysis.get("pnl", {})
    if pnl:
        lines.append("")
        lines.append("📊 PERFORMANCE")
        lines.append(f"  Capital initial : {pnl.get('starting_capital', 'N/A'):,.2f} €")
        lines.append(f"  Solde actuel     : {pnl.get('balance', 'N/A'):,.2f} €")
        lines.append(f"  PnL total        : {pnl.get('total_pnl', 'N/A'):,.2f} € ({pnl.get('pnl_pct', 'N/A'):,.2f}%)")
        lines.append(f"  Positions ouvertes : {pnl.get('open_positions', 0)}")
        lines.append(f"  Trades clôturés  : {pnl.get('closed_trades', 0)}")

    # ─── Overall stats ────────────────────────────────────────────────────
    overall = analysis.get("overall", {})
    if overall:
        lines.append("")
        lines.append("📈 STATISTIQUES GLOBALES")
        wr = overall.get("win_rate", 0)
        lines.append(f"  Taux de réussite : {wr:.1%}")
        lines.append(f"  Trades gagnants  : {overall.get('winning_trades', 0)}")
        lines.append(f"  Trades perdants  : {overall.get('losing_trades', 0)}")
        lines.append(f"  PnL moyen/trade  : {overall.get('avg_pnl_per_trade', 0):,.4f} €")

    # ─── Strategies ────────────────────────────────────────────────────────
    active = analysis.get("strategies", {}).get("active", {})
    eliminated = analysis.get("strategies", {}).get("eliminated", {})
    underexplored = analysis.get("strategies", {}).get("underexplored", [])

    lines.append("")
    lines.append("🎯 STRATÉGIES ACTIVES")
    if active:
        for name, s in active.items():
            wr = s.get("win_rate", 0)
            trades = s.get("total_trades", 0)
            profit = s.get("total_profit", 0)
            weight = s.get("weight", 0)
            lines.append(f"  {name:20s} | WR: {wr:.1%} | {trades:3d} trades | PnL: {profit:,.2f}€ | poids: {weight:.2f}")
    else:
        lines.append("  (aucune)")

    lines.append("")
    lines.append("❌ STRATÉGIES ÉLIMINÉES")
    if eliminated:
        for name, s in eliminated.items():
            wr = s.get("win_rate", 0)
            trades = s.get("total_trades", 0)
            lines.append(f"  {name:20s} | WR: {wr:.1%} | {trades:3d} trades")
    else:
        lines.append("  (aucune)")

    if underexplored:
        lines.append("")
        lines.append("🔍 STRATÉGIES SOUS-EXPLORÉES")
        for name in underexplored:
            s = active.get(name, {})
            trades = s.get("total_trades", 0) if s else 0
            lines.append(f"  {name} ({trades} trades)")

    # ─── Changes made ─────────────────────────────────────────────────────
    lines.append("")
    lines.append("🔧 CHANGEMENTS APPLIQUÉS")

    weight_changes = changes.get("weight_changes", {})
    if weight_changes:
        for name, change in weight_changes.items():
            direction = "↑" if change["delta"] > 0 else "↓"
            lines.append(f"  {name:20s} | poids: {change['old']:.4f} → {change['new']:.4f} {direction}")
    else:
        lines.append("  (aucun ajustement de poids)")

    if changes.get("temperature_change"):
        tc = changes["temperature_change"]
        lines.append(f"  température LLM   : {tc['old']} → {tc['new']}")

    if changes.get("reenabled"):
        lines.append(f"  stratégies ré-activées : {', '.join(changes['reenabled'])}")

    if changes.get("exploration_disabled"):
        lines.append("  phase d'exploration : DÉSACTIVÉE")

    # ─── Recommendations ──────────────────────────────────────────────────
    recommendations = suggestions.get("recommendations", [])
    if recommendations:
        lines.append("")
        lines.append("💡 RECOMMANDATIONS")
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"  {i}. {rec}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    """Main self-improvement routine."""
    print("[INFO] Démarrage de l'auto-amélioration Picsou...", file=sys.stderr)

    # ─── 1. Load data ────────────────────────────────────────────────────
    learning = load_json(LEARNING_FILE)
    portfolio = load_json(PORTFOLIO_FILE)
    llm_config = load_json(LLM_CONFIG_FILE)
    journal = read_last_n_journal_lines(JOURNAL_FILE, 200)

    if not learning:
        print("[ERREUR] Impossible de charger learning.json", file=sys.stderr)
        sys.exit(1)
    if not llm_config:
        print("[ERREUR] Impossible de charger llm_config.json", file=sys.stderr)
        sys.exit(1)
    if not portfolio:
        portfolio = {}

    # ─── 2. Analyze performance ──────────────────────────────────────────
    print("[INFO] Analyse des performances...", file=sys.stderr)
    analysis = analyze_performance(learning, portfolio, journal)

    # ─── 3. Call LLM for suggestions ──────────────────────────────────────
    print(f"[INFO] Appel LLM ({llm_config.get('llm_model', 'unknown')})...", file=sys.stderr)
    raw_suggestions = call_llm_for_suggestions(llm_config, analysis)
    suggestions = parse_llm_suggestions(raw_suggestions)

    if not suggestions:
        print("[WARN] Aucune suggestion LLM reçue, utilisation des heuristiques locales uniquement", file=sys.stderr)
        suggestions = {
            "weight_adjustments": {},
            "temperature_adjustment": None,
            "reenable_strategies": [],
            "disable_exploration": False,
            "recommendations": [
                "Aucune suggestion LLM — ajustements basés uniquement sur les performances observées",
                "Envisager de ré-activer des stratégies éliminées après 7+ jours",
                "Surveiller les stratégies sous-explorées",
            ],
        }

    # ─── 4. Apply safe improvements ───────────────────────────────────────
    print("[INFO] Application des améliorations...", file=sys.stderr)
    # Deep copy to track changes
    import copy
    original_learning = copy.deepcopy(learning)
    original_llm_config = copy.deepcopy(llm_config)

    changes = apply_improvements(learning, llm_config, suggestions, analysis)

    # ─── 5. Save changes ──────────────────────────────────────────────────
    # NEVER modify portfolio.json or journal.jsonl
    learning_changed = (learning != original_learning)
    config_changed = (llm_config != original_llm_config)

    if learning_changed:
        if save_json(LEARNING_FILE, learning):
            print("[INFO] learning.json mis à jour", file=sys.stderr)
        else:
            print("[ERREUR] Échec de la sauvegarde de learning.json", file=sys.stderr)
            sys.exit(1)

    if config_changed:
        if save_json(LLM_CONFIG_FILE, llm_config):
            print("[INFO] llm_config.json mis à jour", file=sys.stderr)
        else:
            print("[ERREUR] Échec de la sauvegarde de llm_config.json", file=sys.stderr)
            sys.exit(1)

    if not learning_changed and not config_changed:
        print("[INFO] Aucun changement appliqué", file=sys.stderr)

    # ─── 6. Generate and print report ─────────────────────────────────────
    report = generate_french_report(analysis, changes, suggestions)
    print(report)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[ERREUR FATALE] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)