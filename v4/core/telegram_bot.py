"""Picsou v4 — Telegram Bot for direct chat with Picsou.

Allows the owner (grums67) to:
- Ask questions about trades, strategies, portfolio
- Give feedback that becomes lessons/observations
- Picsou explains decisions in real-time
- Auto-modification: can adjust lessons/strategies from feedback

Messages trigger REFLECTION, not immediate trades.
Guardrails remain untouchable.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from .config import PicsouConfig
from .brain import Brain
from .executor import Executor
from .memory import Memory
from .portfolio import Portfolio
from .observer import Observer
from .safety import Safety

logger = logging.getLogger(__name__)

# ── Default authorized user IDs ────────────────────────────────────────
# grums67's Telegram user ID — also configurable via env/config
DEFAULT_AUTHORIZED_USER_IDS = []  # Populated from config.telegram.authorized_user_ids


# ── System prompt for chat mode (separate from trading brain) ─────────
CHAT_SYSTEM_PROMPT = """Tu es Picsou, un agent de trading crypto autonome. Tu es en mode conversation avec ton créateur via Telegram.

## Qui tu es
Tu es un trader qui apprend de ses erreurs et s'améliore constamment. Tu observes, tu expérimentes, tu apprends.

## Règles pour cette conversation
1. Tu réponds en français, de manière concise et naturelle
2. Tu peux utiliser les outils pour consulter tes données (marché, portefeuille, mémoire)
3. Les messages ici déclenchent une RÉFLEXION, pas un trade immédiat
4. Les guardrails de sécurité sont intouchables — tu ne peux pas les contourner
5. Si l'utilisateur te donne du feedback, tu peux enregistrer des leçons ou observations
6. Si l'utilisateur te demande d'ajuster une stratégie, tu peux le faire mais les règles de sécurité s'appliquent

## Outils disponibles
- **voir_marche** : prix, volume pour des symboles
- **voir_portefeuille** : balance, positions, PnL
- **voir_memoire** : trades récents, leçons, observations, stats
- **ecrire_memoire** : enregistrer une observation ou une leçon
- **ajuster_poids** : modifier le statut/poids d'une stratégie (activer/désactiver/mettre en veille/boost)

## Format
Réponds naturellement en français. Si tu utilises un outil, fais-le. Après avoir reçu les résultats, explique ce que tu vois et ce que tu en penses.

Tu penses en français, tes stratégies sont en Python. Tu es libre, mais responsable."""


class PicsouTelegramBot:
    """Telegram bot that connects to Picsou's brain for conversations."""

    def __init__(self, config: PicsouConfig, memory: Memory,
                 portfolio: Portfolio, exchanges: Dict):
        self.config = config
        self.memory = memory
        self.portfolio = portfolio
        self.exchanges = exchanges
        self.observer = Observer(config, exchanges)
        self.brain = Brain(config)
        self.executor = Executor(Safety(config), portfolio, memory)
        self.token = config.telegram.token
        self.authorized_users = set(config.telegram.authorized_user_ids) if config.telegram.authorized_user_ids else set()
        self._app: Optional[Application] = None

        # Conversation history per user (simple in-memory, last N messages)
        self._history: Dict[int, list] = {}

    def _get_history(self, user_id: int) -> list:
        """Get conversation history for a user."""
        if user_id not in self._history:
            self._history[user_id] = []
        return self._history[user_id]

    def _add_to_history(self, user_id: int, role: str, content: str):
        """Add message to history, keep last 20 messages."""
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append({"role": role, "content": content})
        # Keep only last 20 messages
        if len(self._history[user_id]) > 20:
            self._history[user_id] = self._history[user_id][-20:]

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user = update.effective_user
        if user.id not in self.authorized_users:
            await update.message.reply_text("⛔ Accès refusé. Picsou ne reconnaît pas cet utilisateur.")
            logger.warning("Unauthorized Telegram user: %s (id=%d)", user.username, user.id)
            return

        await update.message.reply_text(
            f"🦆 Salut {user.first_name} ! C'est Picsou.\n\n"
            "Je suis ton agent de trading crypto. Tu peux me poser des questions, "
            "me donner du feedback, ou me demander d'ajuster mes stratégies.\n\n"
            "Commandes :\n"
            "/status — État du portefeuille et des positions\n"
            "/market — Prix actuels du marché\n"
            "/trades — Derniers trades\n"
            "/lessons — Mes leçons apprises\n"
            "/strategies — Stratégies actives\n"
            "/help — Aide\n\n"
            "Ou envoie-moi simplement un message ! 🎯"
        )
        logger.info("Telegram /start from authorized user %s (id=%d)", user.username, user.id)

    async def help_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        user = update.effective_user
        if user.id not in self.authorized_users:
            return

        await update.message.reply_text(
            "🦆 Picsou — Commandes disponibles :\n\n"
            "/start — Message de bienvenue\n"
            "/status — Portefeuille et PnL\n"
            "/market — Prix actuels (BTC, ETH, SOL)\n"
            "/trades — Derniers trades exécutés\n"
            "/lessons — Leçons apprises\n"
            "/strategies — Stratégies et leurs stats\n"
            "/reset — Effacer l'historique de conversation\n"
            "/help — Cette aide\n\n"
            "Tu peux aussi me parler directement :\n"
            "• \"Que penses-tu du BTC en ce moment ?\"\n"
            "• \"Pourquoi as-tu acheté du SOL ?\"\n"
            "• \"N'achète plus quand le RSI est au-dessus de 70\"\n"
            "• \"Désactive la stratégie momentum\"\n"
        )

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command — show portfolio state."""
        user = update.effective_user
        if user.id not in self.authorized_users:
            return

        try:
            state = self.portfolio.get_state()
            pnl = state.get("pnl", {})
            positions = state.get("positions", {})

            lines = [
                "📊 **Portefeuille Picsou**",
                f"💰 Balance: ${state.get('balance', 0):,.2f}",
                f"📈 PnL total: ${pnl.get('total_pnl', 0):,.2f} ({pnl.get('return_pct', 0):+.2f}%)",
                f"🏆 Win rate: {pnl.get('win_rate', 0):.0%} ({pnl.get('winning_trades', 0)}/{pnl.get('total_trades', 0)} trades)",
                f"📐 Positions ouvertes: {state.get('open_position_count', 0)}",
            ]

            if positions:
                lines.append("\n**Positions ouvertes :**")
                for pid, pos in positions.items():
                    lines.append(
                        f"  • {pos.get('symbol', '?')} — {pos.get('side', '?').upper()} "
                        f"{pos.get('amount', 0):.6f} @ ${pos.get('entry_price', 0):,.2f}"
                    )

            await update.message.reply_text("\n".join(lines))
        except Exception as e:
            logger.error("Error in /status: %s", e, exc_info=True)
            await update.message.reply_text(f"❌ Erreur: {e}")

    async def market(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /market command — show current prices."""
        user = update.effective_user
        if user.id not in self.authorized_users:
            return

        try:
            market_data = self.observer.fetch_market_data()
            lines = ["📈 **Marché actuel**"]
            for key, md in market_data.items():
                ticker = md.get("ticker", {})
                price = ticker.get("last", 0) if ticker else 0
                change = ticker.get("change_24h", 0) if ticker else 0
                vol = ticker.get("volume_24h", 0) if ticker else 0
                emoji = "🟢" if change >= 0 else "🔴"
                lines.append(
                    f"{emoji} {key}: ${price:,.2f} ({change:+.2f}%) vol=${vol:,.0f}"
                )
            await update.message.reply_text("\n".join(lines))
        except Exception as e:
            logger.error("Error in /market: %s", e, exc_info=True)
            await update.message.reply_text(f"❌ Erreur: {e}")

    async def trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /trades command — show recent trades."""
        user = update.effective_user
        if user.id not in self.authorized_users:
            return

        try:
            recent = self.memory.get_recent_trades(limit=10)
            if not recent:
                await update.message.reply_text("📋 Aucun trade récent.")
                return

            lines = ["📋 **Derniers trades**"]
            for t in recent[:10]:
                side = t.get("side", "?").upper()
                symbol = t.get("symbol", "?")
                price = t.get("price", 0)
                pnl = t.get("pnl")
                status = t.get("status", "?")
                pnl_str = f" PnL=${pnl:.2f}" if pnl is not None else ""
                lines.append(f"  • {side} {symbol} @ ${price:,.2f} [{status}]{pnl_str}")

            await update.message.reply_text("\n".join(lines))
        except Exception as e:
            logger.error("Error in /trades: %s", e, exc_info=True)
            await update.message.reply_text(f"❌ Erreur: {e}")

    async def lessons(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /lessons command — show active lessons."""
        user = update.effective_user
        if user.id not in self.authorized_users:
            return

        try:
            active_lessons = self.memory.get_active_lessons(limit=15)
            if not active_lessons:
                await update.message.reply_text("📖 Aucune leçon active.")
                return

            lines = ["📖 **Leçons apprises**"]
            for l in active_lessons:
                lines.append(f"  • {l.get('lesson', '')}")

            await update.message.reply_text("\n".join(lines))
        except Exception as e:
            logger.error("Error in /lessons: %s", e, exc_info=True)
            await update.message.reply_text(f"❌ Erreur: {e}")

    async def strategies(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /strategies command — show strategy states."""
        user = update.effective_user
        if user.id not in self.authorized_users:
            return

        try:
            all_strats = self.memory.get_all_strategies()
            if not all_strats:
                await update.message.reply_text("⚙️ Aucune stratégie enregistrée.")
                return

            lines = ["⚙️ **Stratégies**"]
            for s in all_strats:
                status_emoji = {"active": "🟢", "probation": "🟡", "dormant": "🔴"}.get(s.get("status", ""), "⚪")
                lines.append(
                    f"  {status_emoji} {s.get('name', '?')} — status={s.get('status', '?')} "
                    f"weight={s.get('weight', 0):.2f} trades={s.get('total_trades', 0)} "
                    f"WR={s.get('win_rate', 0):.0%}"
                )

            await update.message.reply_text("\n".join(lines))
        except Exception as e:
            logger.error("Error in /strategies: %s", e, exc_info=True)
            await update.message.reply_text(f"❌ Erreur: {e}")

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /reset command — clear conversation history."""
        user = update.effective_user
        if user.id not in self.authorized_users:
            return

        self._history.pop(user.id, None)
        await update.message.reply_text("🔄 Historique de conversation effacé. Tu repars de zéro !")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages — the core chat handler."""
        user = update.effective_user
        if user.id not in self.authorized_users:
            await update.message.reply_text("⛔ Accès refusé.")
            logger.warning("Unauthorized message from Telegram user: %s (id=%d)", user.username, user.id)
            return

        text = update.message.text.strip()
        if not text:
            return

        logger.info("Telegram message from %s (id=%d): %s", user.username, user.id, text[:100])

        # Show "typing" indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        try:
            # Process the message through Picsou's brain
            response = self._process_message(user.id, text)
            await update.message.reply_text(response)
        except Exception as e:
            logger.error("Error processing Telegram message: %s", e, exc_info=True)
            await update.message.reply_text(f"❌ Oups, erreur interne : {e}")

    def _process_message(self, user_id: int, text: str) -> str:
        """Process a user message through Picsou's brain and return a response.

        This is the key method: it builds context, sends to the LLM with chat tools,
        processes any tool calls, and returns the final text response.
        """
        # Add user message to history
        self._add_to_history(user_id, "user", text)

        # Build context for the LLM
        context = self._build_chat_context()

        # Build messages list with system prompt + conversation history
        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT + "\n\n## Contexte actuel\n" + context}]

        # Add conversation history
        for msg in self._get_history(user_id):
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Call LLM with tool support (up to 5 rounds of tool calling)
        from .brain import TOOL_DEFINITIONS
        # Chat tools: vendre (close positions) but not acheter (no impulsive buys from chat)
        chat_tools = [t for t in TOOL_DEFINITIONS
                      if t["function"]["name"] in (
                          "voir_marche", "voir_portefeuille", "voir_memoire",
                          "ecrire_memoire", "ajuster_poids", "vendre"
                      )]

        for round_num in range(5):
            response = self.brain._call_llm(messages, tools=chat_tools)
            if response is None:
                return "🤯 Je n'arrive pas à réfléchir en ce moment. Le LLM ne répond pas."

            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})

            # If no tool calls, we have a text response
            if "tool_calls" not in message or not message["tool_calls"]:
                content = message.get("content", "")
                # Add to history
                self._add_to_history(user_id, "assistant", content)
                return self._format_response(content)

            # Process tool calls
            messages.append(message)
            for tool_call in message["tool_calls"]:
                tool_result = self._execute_chat_tool(tool_call, user_id)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(tool_result, ensure_ascii=False, default=str),
                })

        # If we exhausted tool call rounds, get final response without tools
        response = self.brain._call_llm(messages, tools=None)
        if response:
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            self._add_to_history(user_id, "assistant", content)
            return self._format_response(content)

        return "🤯 J'ai fait trop de calculs et je n'arrive plus à répondre. Réessaie dans un moment."

    def _build_chat_context(self) -> str:
        """Build a concise context string for the chat LLM."""
        parts = []

        # Portfolio
        try:
            state = self.portfolio.get_state()
            pnl = state.get("pnl", {})
            parts.append(
                f"**Portefeuille:** Balance=${state.get('balance', 0):,.2f} | "
                f"PnL=${pnl.get('total_pnl', 0):,.2f} ({pnl.get('return_pct', 0):+.2f}%) | "
                f"Positions ouvertes: {state.get('open_position_count', 0)} | "
                f"Win rate: {pnl.get('win_rate', 0):.0%}"
            )
        except Exception:
            parts.append("**Portefeuille:** Données non disponibles")

        # Market (cached or quick fetch)
        try:
            market_data = self.observer.fetch_market_data()
            market_lines = []
            for key, md in market_data.items():
                ticker = md.get("ticker", {})
                if ticker:
                    price = ticker.get("last", 0)
                    change = ticker.get("change_24h", 0)
                    market_lines.append(f"{key}: ${price:,.2f} ({change:+.2f}%)")
            if market_lines:
                parts.append("**Marché:** " + " | ".join(market_lines))
        except Exception:
            pass

        # Active strategies
        try:
            strats = self.memory.get_active_strategies()
            if strats:
                strat_lines = [f"{s['name']}({s['status']},w={s.get('weight', 0):.2f})" for s in strats]
                parts.append("**Stratégies actives:** " + ", ".join(strat_lines))
        except Exception:
            pass

        # Recent lessons
        try:
            lessons = self.memory.get_active_lessons(limit=5)
            if lessons:
                lesson_lines = [l.get("lesson", "") for l in lessons]
                parts.append("**Leçons récentes:** " + " | ".join(lesson_lines))
        except Exception:
            pass

        # Time
        parts.append(f"**Heure:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

        return "\n".join(parts)

    def _execute_chat_tool(self, tool_call: Dict, user_id: int) -> Dict:
        """Execute a tool call from the chat LLM.

        Only allows read-only tools + memory writing + strategy weight adjustment.
        Buy/sell are NOT allowed from Telegram chat.
        """
        func_name = tool_call["function"]["name"]
        try:
            args = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError:
            return {"error": "Arguments JSON invalides"}

        if func_name == "voir_marche":
            market_data = self.observer.fetch_market_data()
            symbols = args.get("symboles", [])
            if symbols:
                result = {k: v for k, v in market_data.items()
                          if any(s.lower() in k.lower() for s in symbols)}
                return result if result else market_data
            return market_data

        elif func_name == "voir_portefeuille":
            return self.portfolio.get_state()

        elif func_name == "voir_memoire":
            mem_type = args.get("type", "all")
            limite = args.get("limite", 20)
            context = self.memory.get_context_for_llm()
            if mem_type == "trades":
                return {"recent_trades": self.memory.get_recent_trades(limit=limite)}
            elif mem_type == "lessons":
                return {"lessons": self.memory.get_active_lessons(limit=limite)}
            elif mem_type == "observations":
                return {"observations": self.memory.get_recent_observations(limit=limite)}
            elif mem_type == "stats":
                return {"stats": self.memory.get_trade_stats()}
            return context

        elif func_name == "ecrire_memoire":
            mem_type = args.get("type", "observation")
            contenu = args.get("contenu", "")
            contexte = args.get("contexte", "")

            if mem_type == "observation":
                self.memory.add_observation(category="telegram_feedback", content=contenu)
            elif mem_type == "lecon":
                self.memory.add_lesson(lesson=contenu, context=contexte or "Telegram feedback")
            else:
                self.memory.add_observation(category="telegram_feedback", content=contenu)

            return {"status": "ok", "message": f"Mémoire enregistrée ({mem_type})"}

        elif func_name == "ajuster_poids":
            strategie = args.get("strategie", "")
            action = args.get("action", "")
            raison = args.get("raison", "")

            # Map action to status
            status_map = {
                "activer": "active",
                "desactiver": "dormant",
                "veille": "probation",
                "boost": "active",  # boost also activates
            }
            new_status = status_map.get(action)

            if not strategie:
                return {"error": "Nom de stratégie requis"}

            strat = self.memory.get_strategy(strategie)
            if not strat:
                return {"error": f"Stratégie '{strategie}' non trouvée"}

            if new_status:
                self.memory.set_strategy_status(strategie, new_status)

            # Adjust weight
            if action == "boost":
                current_weight = strat.get("weight", 0.1)
                new_weight = min(current_weight * 1.5, 0.4)  # Cap at 40%, safety
                self.memory.update_strategy(strategie, weight=new_weight)
            elif action == "desactiver":
                self.memory.update_strategy(strategie, weight=0.01)

            self.memory.add_observation(
                category="telegram_adjustment",
                content=f"Stratégie {strategie}: action={action}, raison={raison}",
                relevance="high"
            )

            return {
                "status": "ok",
                "message": f"Stratégie {strategie}: {action} effectué",
                "details": f"status={new_status}" if new_status else ""
            }

        elif func_name == "vendre":
            symbole = args.get("symbole", "")
            raison = args.get("raison", "")
            confiance = args.get("confiance", 0.7)
            nombre = args.get("nombre", "1")

            if not symbole:
                return {"error": "Symbole requis pour la vente"}

            # Determine how many positions to close
            import re
            def _base(sym):
                return re.sub(r'[-_/]?[Uu][Ss][Dd][Tt]$', '', sym)

            base = _base(symbole.upper())
            matching = [p for p in self.portfolio.get_open_positions()
                        if _base(p.symbol) == base and p.side == "long"]

            if not matching:
                return {"error": f"Aucune position ouverte sur {symbole}"}

            if nombre == "tout":
                to_close = matching
            else:
                try:
                    n = int(nombre)
                except ValueError:
                    n = 1
                to_close = matching[:n]

            results = []
            for pos in to_close:
                trade_decision = {
                    "action": "sell",
                    "symbol": symbole.upper(),
                    "size_pct": 1.0,
                    "confidence": confiance,
                    "strategy": "telegram_chat",
                    "reasoning": raison,
                }
                executed = self.executor.execute([trade_decision], self.exchanges)
                if executed:
                    results.append(executed)

            if results:
                self.memory.add_observation(
                    category="telegram_trade",
                    content=f"Vente {len(results)} position(s) {symbole} via chat: {raison}",
                    relevance="high"
                )
                return {
                    "status": "ok",
                    "message": f"Vente {len(results)} position(s) {symbole} exécutée(s)",
                    "positions_fermees": len(results),
                    "details": str(results)
                }
            else:
                return {"error": f"Impossible de vendre {symbole} — aucune position ouverte ou erreur"}

        else:
            return {"error": f"Outil '{func_name}' non autorisé en mode chat"}

    def _format_response(self, text: str) -> str:
        """Format the LLM response for Telegram.

        Telegram messages have a 4096 char limit.
        Also strip markdown code blocks that might not render well.
        """
        # Truncate if needed
        if len(text) > 3900:
            text = text[:3850] + "\n\n[...]"

        # Clean up excessive markdown that Telegram doesn't support well
        # Replace ```json ... ``` with readable text
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)

        return text.strip()

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Log errors from the Telegram bot."""
        logger.error("Telegram bot error: %s", context.error, exc_info=context.error)

    def create_application(self) -> Application:
        """Create and configure the Telegram bot application."""
        app = Application.builder().token(self.token).build()

        # Command handlers
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("help", self.help_cmd))
        app.add_handler(CommandHandler("status", self.status))
        app.add_handler(CommandHandler("market", self.market))
        app.add_handler(CommandHandler("trades", self.trades))
        app.add_handler(CommandHandler("lessons", self.lessons))
        app.add_handler(CommandHandler("strategies", self.strategies))
        app.add_handler(CommandHandler("reset", self.reset))

        # Message handler (text messages)
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        # Error handler
        app.add_error_handler(self.error_handler)

        self._app = app
        return app

    async def start_bot(self):
        """Start the Telegram bot (async)."""
        app = self.create_application()
        logger.info("Starting Picsou Telegram bot...")
        await app.initialize()
        await app.start()

        # Register slash commands in Telegram UI
        from telegram import BotCommand
        commands = [
            BotCommand("start", "🚀 Message de bienvenue"),
            BotCommand("help", "📖 Aide et commandes disponibles"),
            BotCommand("status", "📊 Portefeuille et PnL"),
            BotCommand("market", "📈 Prix actuels (BTC, ETH, SOL)"),
            BotCommand("trades", "💰 Derniers trades exécutés"),
            BotCommand("lessons", "🧠 Leçons apprises"),
            BotCommand("strategies", "⚙️ Stratégies et leurs stats"),
            BotCommand("reset", "🔄 Réinitialiser la conversation"),
        ]
        await app.bot.set_my_commands(commands)
        logger.info("Telegram commands registered")

        await app.updater.start_polling(drop_pending_updates=True)
        logger.info("Picsou Telegram bot is running!")

    async def stop_bot(self):
        """Stop the Telegram bot gracefully."""
        if self._app:
            logger.info("Stopping Picsou Telegram bot...")
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("Picsou Telegram bot stopped.")


def run_telegram_bot(config: PicsouConfig, memory: Memory,
                     portfolio: Portfolio, exchanges: Dict):
    """Run the Telegram bot in its own async event loop.

    This is meant to be called from a thread in run.py.
    """
    import asyncio

    bot = PicsouTelegramBot(config, memory, portfolio, exchanges)

    async def _run():
        await bot.start_bot()
        # Keep running forever
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await bot.stop_bot()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run())
    except KeyboardInterrupt:
        loop.run_until_complete(bot.stop_bot())
    finally:
        loop.close()