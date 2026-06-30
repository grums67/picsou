"""Picsou v4 — LLM Brain with function calling.

The brain is called every N heartbeat cycles (slow loop).
It receives context, calls tools, and produces decisions.
Uses OpenAI-compatible API with function calling.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

import requests

from .config import PicsouConfig
from .system_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ── Tool definitions (function calling schema) ──────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "voir_marche",
            "description": "Obtenir les données de marché pour un ou plusieurs symboles (prix, bougies, volume)",
            "parameters": {
                "type": "object",
                "properties": {
                    "symboles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Liste des symboles, ex: ['BTC', 'ETH', 'SOL']"
                    }
                },
                "required": ["symboles"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "voir_portefeuille",
            "description": "Obtenir l'état du portefeuille (balance, positions, PnL)",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "voir_memoire",
            "description": "Obtenir les trades récents, leçons apprises, observations, et stats",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["trades", "lessons", "observations", "stats", "all"],
                        "description": "Quel type de mémoire consulter"
                    },
                    "limite": {
                        "type": "integer",
                        "description": "Nombre d'entrées à récupérer (défaut 20)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "acheter",
            "description": "Ouvrir une position long sur un symbole",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbole": {"type": "string", "description": "Symbole, ex: BTC"},
                    "taille_pct": {
                        "type": "number",
                        "description": "Pourcentage du portefeuille à investir (0.01 à 0.20)"
                    },
                    "confiance": {
                        "type": "number",
                        "description": "Niveau de confiance (0.0 à 1.0)"
                    },
                    "strategie": {"type": "string", "description": "Nom de la stratégie"},
                    "raison": {"type": "string", "description": "Pourquoi cet achat"}
                },
                "required": ["symbole", "taille_pct", "confiance", "raison"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vendre",
            "description": "Fermer des positions long sur un symbole. Par défaut ferme 1 position. Utilise nombre='tout' pour fermer TOUTES les positions du symbole.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbole": {"type": "string", "description": "Symbole, ex: BTC"},
                    "nombre": {
                        "type": "string",
                        "description": "Nombre de positions à fermer: '1', '2', '3' ou 'tout'",
                        "default": "1"
                    },
                    "confiance": {
                        "type": "number",
                        "description": "Niveau de confiance (0.0 à 1.0)"
                    },
                    "strategie": {"type": "string", "description": "Nom de la stratégie"},
                    "raison": {"type": "string", "description": "Pourquoi cette vente"}
                },
                "required": ["symbole", "raison"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ecrire_strategie",
            "description": "Créer ou modifier un fichier de stratégie Python. Le code DOIT implémenter signal() et metadata()",
            "parameters": {
                "type": "object",
                "properties": {
                    "nom": {"type": "string", "description": "Nom du fichier stratégie (sans .py)"},
                    "code": {"type": "string", "description": "Code Python complet de la stratégie"},
                    "raison": {"type": "string", "description": "Pourquoi cette stratégie"}
                },
                "required": ["nom", "code", "raison"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tester_strategie",
            "description": "Backtester une stratégie sur données historiques",
            "parameters": {
                "type": "object",
                "properties": {
                    "nom": {"type": "string", "description": "Nom de la stratégie à tester"},
                    "symbole": {
                        "type": "string",
                        "description": "Symbole pour le backtest (défaut BTC)",
                        "default": "BTC"
                    }
                },
                "required": ["nom"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ecrire_memoire",
            "description": "Enregistrer une observation ou une leçon pour les cycles futurs",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["observation", "lecon"],
                        "description": "Type de mémoire"
                    },
                    "contenu": {"type": "string", "description": "Le contenu à retenir"},
                    "contexte": {"type": "string", "description": "Contexte optionnel"}
                },
                "required": ["type", "contenu"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ajuster_poids",
            "description": "Modifier le statut ou poids d'une stratégie (activer, désactiver, mettre en veille)",
            "parameters": {
                "type": "object",
                "properties": {
                    "strategie": {"type": "string", "description": "Nom de la stratégie"},
                    "action": {
                        "type": "string",
                        "enum": ["activer", "desactiver", "veille", "boost"],
                        "description": "activer=active, desactiver=désactive,veille=met en dormance,boost=augmente le poids"
                    },
                    "raison": {"type": "string", "description": "Pourquoi ce changement"}
                },
                "required": ["strategie", "action", "raison"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lire_code",
            "description": "Lire un fichier de ton propre code source. Utile pour comprendre comment tu fonctionnes, diagnostiquer un bug, ou trouver comment améliorer un outil. Chemins: strategies/, core/, dashboard/",
            "parameters": {
                "type": "object",
                "properties": {
                    "fichier": {
                        "type": "string",
                        "description": "Chemin du fichier, ex: 'core/brain_loop.py' ou 'strategies/ema_crossover_v1.py' ou 'core/system_prompt.py'"
                    }
                },
                "required": ["fichier"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "modifier_code",
            "description": "Modifier ou créer un fichier dans ton propre code source. Tu peux corriger des bugs, améliorer des outils, ajouter des fonctionnalités, ou créer de nouvelles stratégies. NE JAMAIS modifier les fichiers core/ sauf pour corriger un bug critique.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fichier": {
                        "type": "string",
                        "description": "Chemin du fichier, ex: 'strategies/nouvelle_strat.py' ou 'core/system_prompt.py'"
                    },
                    "contenu": {
                        "type": "string",
                        "description": "Contenu complet du fichier (sera écrit tel quel)"
                    },
                    "raison": {
                        "type": "string",
                        "description": "Pourquoi cette modification"
                    }
                },
                "required": ["fichier", "contenu", "raison"]
            }
        }
    },
]


class Brain:
    """LLM brain that makes decisions using function calling."""

    def __init__(self, config: PicsouConfig, memory=None):
        self.config = config
        self.url = config.llm.url
        self.api_key = config.llm.api_key
        self.model = config.llm.model
        self.temperature = config.llm.temperature
        self.max_tokens = config.llm.max_tokens
        self.last_prompt = ""
        self.last_response = ""
        self.memory = memory

    def think(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Send context to LLM and get decisions back.

        Uses multi-turn function calling: the LLM can call tools
        to gather more info before making a final decision.
        """
        # Build the user message with full context
        user_msg = self._build_context_message(context)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        self.last_prompt = user_msg

        # Up to 5 rounds of tool calling
        for _ in range(5):
            response = self._call_llm(messages)

            if response is None:
                return {"action": "hold", "reasoning": "LLM unavailable"}

            # Check if LLM wants to call tools
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})

            # If no tool calls, return the final message
            if "tool_calls" not in message or not message["tool_calls"]:
                # Parse the text response as a decision
                content = message.get("content", "")
                self.last_response = content
                return self._parse_decision(content)

            # Process tool calls
            messages.append(message)
            for tool_call in message["tool_calls"]:
                tool_result = self._execute_tool_call(tool_call, context)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(tool_result),
                })

        # If we exhausted tool call rounds, get final decision
        response = self._call_llm(messages, tools=None)
        if response:
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            self.last_response = content
            return self._parse_decision(content)

        return {"action": "hold", "reasoning": "LLM max rounds reached"}

    def _call_llm(self, messages: list, tools: list = None) -> Optional[Dict]:
        """Make an API call to the LLM."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            resp = requests.post(
                f"{self.url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return None

    def _build_context_message(self, context: Dict) -> str:
        """Build the user message with all context for the LLM."""
        parts = [f"## Cycle du {context.get('timestamp', 'maintenant')}"]

        # Portfolio
        portfolio = context.get("portfolio", {})
        parts.append(f"\n### Portefeuille\n"
                     f"- Balance: ${portfolio.get('balance', 0):,.2f}\n"
                     f"- Positions ouvertes: {portfolio.get('open_position_count', 0)}\n"
                     f"- PnL total: {portfolio.get('pnl', {}).get('total_pnl', 0):,.2f} "
                     f"({portfolio.get('pnl', {}).get('return_pct', 0):+.2f}%)\n"
                     f"- Win rate: {portfolio.get('pnl', {}).get('win_rate', 0):.1%}")

        # Market summary
        market = context.get("market", {})
        if market:
            parts.append("\n### Marché")
            for key, md in market.items():
                price = md.get("price", 0)
                change = md.get("change_24h", 0)
                vol = md.get("volume_24h", 0)
                parts.append(f"- {key}: ${price:,.2f} ({change:+.2f}%) vol={vol:,.0f}")

        # Sentiment
        sentiment = context.get("sentiment", {})
        fng = sentiment.get("fear_and_greed", {})
        if fng:
            parts.append(f"\n### Sentiment\n- Fear & Greed: {fng.get('value', '?')} ({fng.get('classification', '?')})")

        # Active strategies
        strategies = context.get("memory", {}).get("active_strategies", [])
        if strategies:
            parts.append("\n### Stratégies actives")
            for s in strategies:
                parts.append(f"- {s['name']}: status={s['status']} weight={s.get('weight', 0):.2f} "
                             f"trades={s.get('total_trades', 0)} WR={s.get('win_rate', 0):.0%}")

        # Recent lessons
        lessons = context.get("memory", {}).get("lessons", [])
        if lessons:
            parts.append("\n### Leçons apprises")
            for l in lessons[:5]:
                parts.append(f"- {l.get('lesson', '')}")

        # Recent observations
        observations = context.get("memory", {}).get("recent_observations", [])
        if observations:
            parts.append("\n### Observations récentes")
            for o in observations[:5]:
                parts.append(f"- [{o.get('category', '')}] {o.get('content', '')}")

        # Recent trades
        trades = context.get("memory", {}).get("recent_trades", [])
        if trades:
            parts.append("\n### Trades récents")
            for t in trades[:5]:
                parts.append(f"- {t.get('side', '?').upper()} {t.get('symbol', '?')} "
                             f"@ {t.get('price', 0):.2f} PnL={t.get('pnl', '?')}")

        parts.append("\n\nQue décides-tu ? Utilise les outils si tu as besoin d'informations supplémentaires, ou donne ta décision directement.")
        return "\n".join(parts)

    def _execute_tool_call(self, tool_call: Dict, context: Dict) -> Dict:
        """Execute a tool call from the LLM. Returns result for the LLM."""
        func_name = tool_call["function"]["name"]
        try:
            args = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError:
            return {"error": "Invalid JSON arguments"}

        # These tools return data from the context that was pre-loaded
        if func_name == "voir_marche":
            market = context.get("market", {})
            symbols = args.get("symboles", [])
            result = {k: v for k, v in market.items()
                      if any(s.lower() in k.lower() for s in symbols)}
            return result if result else market  # Return all if no match

        elif func_name == "voir_portefeuille":
            return context.get("portfolio", {})

        elif func_name == "voir_memoire":
            return context.get("memory", {})

        elif func_name == "lire_code":
            fichier = args.get("fichier", "")
            # Security: only allow reading within the project
            base_path = self.config.data_path.parent  # project root
            filepath = (base_path / fichier).resolve()
            # Prevent path traversal
            if not str(filepath).startswith(str(base_path)):
                return {"error": "Accès refusé: chemin hors du projet"}
            if not filepath.exists():
                return {"error": f"Fichier '{fichier}' non trouvé"}
            if not filepath.suffix == '.py' and filepath.name not in ('system_prompt.py',):
                # Allow .py files and system_prompt
                pass
            try:
                content = filepath.read_text(encoding='utf-8')
                lines = content.split('\n')
                return {
                    "fichier": fichier,
                    "lignes": len(lines),
                    "contenu": content[:5000],  # Limit to 5000 chars
                    "tronque": len(content) > 5000
                }
            except Exception as e:
                return {"error": f"Erreur lecture: {e}"}

        elif func_name == "modifier_code":
            fichier = args.get("fichier", "")
            contenu = args.get("contenu", "")
            raison = args.get("raison", "")
            # Security: only allow modifying strategies/ and core/system_prompt.py
            base_path = self.config.data_path.parent
            filepath = (base_path / fichier).resolve()
            # Prevent path traversal and restrict writable paths
            if not str(filepath).startswith(str(base_path)):
                return {"error": "Accès refusé: chemin hors du projet"}
            allowed_prefixes = ["strategies/", "core/system_prompt.py"]
            if not any(fichier.startswith(p) or fichier == p for p in allowed_prefixes):
                return {"error": f"Seuls les fichiers strategies/ et core/system_prompt.py peuvent être modifiés. Chemin demandé: '{fichier}'"}
            try:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(contenu, encoding='utf-8')
                # Record the modification in memory
                if self.memory:
                    self.memory.add_observation(
                        category="code_modification",
                        content=f"Modified {fichier}: {raison}",
                        relevance="high"
                    )
                logger.info("Code modified by LLM: %s — %s", fichier, raison)
                return {
                    "status": "ok",
                    "message": f"Fichier '{fichier}' modifié avec succès",
                    "raison": raison,
                    "taille": len(contenu)
                }
            except Exception as e:
                return {"error": f"Erreur écriture: {e}"}

        else:
            # These tools are handled by the caller (heartbeat or brain loop)
            # We return a placeholder — the actual execution happens in the loop
            return {"status": "queued", "tool": func_name, "args": args}

    def _parse_decision(self, content: str) -> Dict:
        """Parse LLM text response into a structured decision."""
        # Try to extract JSON from the response
        try:
            # Look for JSON block
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            elif "{" in content and "}" in content:
                start = content.index("{")
                end = content.rindex("}") + 1
                json_str = content[start:end]
            else:
                json_str = content

            decision = json.loads(json_str)
            # Ensure required fields
            decision.setdefault("action", "hold")
            decision.setdefault("reasoning", content)
            return decision

        except (json.JSONDecodeError, ValueError):
            # If we can't parse JSON, return a hold with the raw content
            return {
                "action": "hold",
                "reasoning": content[:500],
                "raw_response": True,
            }

    def get_config_status(self) -> Dict:
        """Get current brain config for dashboard."""
        return {
            "model": self.model,
            "url": self.url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }