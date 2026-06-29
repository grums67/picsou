"""
Picsou v4 Dashboard - FastAPI server with authentication and real-time crypto trading monitoring.
Adapted to read data from SQLite instead of JSON files.
"""
import json
import os
import hashlib
import hmac
import secrets
import time
import asyncio
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import quote

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

DATA_DIR = Path("/root/PROJECTS/picsou/v4/data")
DB_PATH = DATA_DIR / "picsou.db"
TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
START_TIME = time.time()

AUTH_FILE = DATA_DIR / "auth.json"
AUTH_COOKIE = "picsou_session"
SESSION_TTL = 60 * 60 * 24 * 30  # 30 days

app = FastAPI(title="Picsou v4 Dashboard", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_kill_switch_active = False


# ─── SQLite Helpers ────────────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    """Get a connection to the v4 SQLite database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def get_v4_portfolio() -> dict:
    """Read trades and snapshots from SQLite to reconstruct portfolio data."""
    result = {
        "balance": 0.0,
        "starting_capital": 10000.0,
        "positions": [],
        "trades": [],
        "pnl": 0.0,
        "pnl_pct": 0.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "phase": "LEARNING",
        "last_cycle": None,
    }
    try:
        conn = _get_db()
        # Get latest snapshot for balance and totals
        snap = conn.execute(
            "SELECT * FROM snapshots ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if snap:
            result["balance"] = float(snap["balance"] or 0)
            result["realized_pnl"] = float(snap["total_pnl"] or 0)
            result["pnl_pct"] = float(snap["return_pct"] or 0)
            result["last_cycle"] = snap["timestamp"]
            # active_strategies is stored as JSON array string
            active_strats = snap["active_strategies"]
            if active_strats:
                try:
                    result["active_strategies"] = json.loads(active_strats)
                except (json.JSONDecodeError, TypeError):
                    result["active_strategies"] = []
            result["cycle_number"] = snap["cycle_number"]

        # Get all trades
        trades_rows = conn.execute(
            "SELECT * FROM trades ORDER BY timestamp DESC"
        ).fetchall()
        trades_list = []
        for t in trades_rows:
            trade = {
                "id": t["id"],
                "timestamp": t["timestamp"],
                "exchange": t["exchange"],
                "symbol": t["symbol"],
                "side": t["side"],
                "amount": float(t["amount"] or 0),
                "price": float(t["price"] or 0),
                "fee": float(t["fee"] or 0),
                "strategy": t["strategy"],
                "confidence": float(t["confidence"] or 0),
                "reasoning": t["reasoning"],
                "pnl": float(t["pnl"] or 0),
                "close_price": float(t["close_price"] or 0) if t["close_price"] else None,
                "close_timestamp": t["close_timestamp"],
                "status": t["status"],
            }
            trades_list.append(trade)
        result["trades"] = trades_list

        # Open positions are trades with status='open'
        open_positions = []
        for t in trades_rows:
            if t["status"] == "open":
                pos = {
                    "id": t["id"],
                    "timestamp": t["timestamp"],
                    "exchange": t["exchange"],
                    "symbol": t["symbol"],
                    "side": t["side"],
                    "amount": float(t["amount"] or 0),
                    "entry_price": float(t["price"] or 0),
                    "fee": float(t["fee"] or 0),
                    "strategy": t["strategy"],
                    "pnl": float(t["pnl"] or 0),
                    "status": t["status"],
                }
                open_positions.append(pos)
        result["positions"] = open_positions

        # Calculate starting capital from first snapshot or fallback
        first_snap = conn.execute(
            "SELECT * FROM snapshots ORDER BY timestamp ASC LIMIT 1"
        ).fetchone()
        if first_snap:
            result["starting_capital"] = float(first_snap["balance"] or 10000.0)

        # Recalculate PnL
        realized_pnl = sum(float(t["pnl"] or 0) for t in trades_rows if t["status"] == "closed")
        result["realized_pnl"] = realized_pnl
        total_value = result["balance"] + sum(
            float(p["amount"]) * float(p["entry_price"]) for p in open_positions
        )
        result["pnl"] = total_value - result["starting_capital"]
        result["pnl_pct"] = (result["pnl"] / result["starting_capital"] * 100) if result["starting_capital"] else 0.0

        conn.close()
    except Exception as e:
        # Return defaults if DB not available
        pass
    return result


def get_v4_strategies() -> dict:
    """Read strategies table from SQLite."""
    result = {}
    try:
        conn = _get_db()
        rows = conn.execute("SELECT * FROM strategies").fetchall()
        for r in rows:
            name = r["name"]
            metadata = {}
            if r["metadata"]:
                try:
                    metadata = json.loads(r["metadata"])
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            strat = {
                "name": name,
                "filename": r["filename"],
                "status": r["status"],
                "weight": float(r["weight"] or 0),
                "total_trades": int(r["total_trades"] or 0),
                "winning_trades": int(r["winning_trades"] or 0),
                "total_pnl": float(r["total_pnl"] or 0),
                "win_rate": float(r["win_rate"] or 0),
                "sharpe": float(r["sharpe"] or 0),
                "max_drawdown": float(r["max_drawdown"] or 0),
                "created_at": r["created_at"],
                "dormant_since": r["dormant_since"],
                "last_evaluated": r["last_evaluated"],
                "metadata": metadata,
                # Compatibility fields for v3 dashboard format
                "active": r["status"] == "active",
                "probation": r["status"] == "probation",
                "probation_trades": 0,
                "avg_profit": float(r["total_pnl"] or 0) / max(int(r["total_trades"] or 0), 1),
                "sharpe_ratio": float(r["sharpe"] or 0),
                "total_profit": float(r["total_pnl"] or 0),
                "losing_trades": int(r["total_trades"] or 0) - int(r["winning_trades"] or 0),
            }
            result[name] = strat
        conn.close()
    except Exception:
        pass
    return result


def get_v4_observations() -> list:
    """Read observations table from SQLite."""
    result = []
    try:
        conn = _get_db()
        rows = conn.execute("SELECT * FROM observations ORDER BY timestamp DESC").fetchall()
        for r in rows:
            result.append({
                "id": r["id"],
                "timestamp": r["timestamp"],
                "category": r["category"],
                "content": r["content"],
                "relevance": r["relevance"] if "relevance" in r.keys() else "medium",
            })
        conn.close()
    except Exception:
        pass
    return result


def get_v4_lessons() -> list:
    """Read lessons table from SQLite."""
    result = []
    try:
        conn = _get_db()
        rows = conn.execute("SELECT * FROM lessons ORDER BY timestamp DESC").fetchall()
        for r in rows:
            result.append({
                "id": r["id"],
                "timestamp": r["timestamp"],
                "lesson": r["lesson"],
                "context": r["context"],
                "active": bool(r["active"]) if "active" in r.keys() else True,
            })
        conn.close()
    except Exception:
        pass
    return result


async def get_v4_status() -> dict:
    """Query the v4 health endpoint at localhost:3038."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:3038/api/health")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return {}


# ─── Auth ────────────────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    """Hash password with pbkdf2_sha256."""
    salt = os.urandom(16).hex()
    iterations = 600000
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), iterations).hex()
    return f"pbkdf2_sha256${iterations}${salt}${digest}"


def _verify_password(password: str, stored: str) -> bool:
    """Verify password against stored hash."""
    if not stored:
        return False
    if stored.startswith("pbkdf2_sha256$"):
        try:
            _, iterations_raw, salt_hex, digest_hex = stored.split("$", 3)
            iterations = int(iterations_raw)
            expected = bytes.fromhex(digest_hex)
            actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), iterations)
            return hmac.compare_digest(actual, expected)
        except (ValueError, Exception):
            return False
    return hmac.compare_digest(password, stored)


def _load_auth() -> dict:
    """Load auth config. If no user exists, returns setup_needed=True."""
    if AUTH_FILE.exists():
        try:
            with open(AUTH_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"users": {}, "setup_needed": True}


def _save_auth(auth: dict):
    """Save auth config."""
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(AUTH_FILE, "w") as f:
        json.dump(auth, f, indent=2)


def _session_secret() -> bytes:
    """Generate or load a persistent session signing secret."""
    secret_file = DATA_DIR / ".session_secret"
    if secret_file.exists():
        return secret_file.read_bytes()
    secret = os.urandom(32)
    secret_file.write_bytes(secret)
    return secret


def _create_session_token(username: str) -> str:
    """Create a signed session token."""
    issued = int(time.time())
    secret = _session_secret()
    payload = f"{username}:{issued}".encode()
    sig = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    return f"{username}:{issued}:{sig}"


def _verify_session(token: str) -> Optional[str]:
    """Verify session token, returns username if valid."""
    if not token:
        return None
    try:
        username, issued_raw, sig = token.split(":", 2)
        issued = int(issued_raw)
    except ValueError:
        return None
    if abs(int(time.time()) - issued) > SESSION_TTL:
        return None
    secret = _session_secret()
    payload = f"{username}:{issued}".encode()
    expected = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    if hmac.compare_digest(sig, expected):
        return username
    return None


def _is_setup_needed() -> bool:
    """Check if initial setup (account creation) is needed."""
    auth = _load_auth()
    return auth.get("setup_needed", True) and len(auth.get("users", {})) == 0


def _auth_exempt(path: str) -> bool:
    """Paths that don't require authentication."""
    return path in {
        "/login",
        "/setup",
        "/api/market",
        "/api/health",
        "/manifest.json",
        "/sw.js",
        "/static/icons/icon.svg",
        "/static/icons/icon-192.png",
        "/static/icons/icon-512.png",
    } or path.startswith("/static/")


def _get_current_user(request: Request) -> Optional[str]:
    """Get current user from session cookie, or None."""
    token = request.cookies.get(AUTH_COOKIE, "")
    return _verify_session(token)


# ─── Auth Middleware ─────────────────────────────────────────────────────────

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path.replace("/#", "")
    # Exempt paths
    if _auth_exempt(path):
        return await call_next(request)
    # API endpoints require auth (return 401 instead of redirect)
    if path.startswith("/api/"):
        user = _get_current_user(request)
        if not user:
            return JSONResponse({"error": "Authentication required"}, status_code=401)
        return await call_next(request)
    # Dashboard pages: redirect to login or setup
    user = _get_current_user(request)
    if not user:
        if _is_setup_needed():
            return RedirectResponse(url="/setup", status_code=303)
        target = request.url.path or "/"
        if request.url.query:
            target = f"{target}?{request.url.query}"
        return RedirectResponse(url=f"/login?next={quote(target, safe='/%?=&')}", status_code=303)
    return await call_next(request)


# ─── Static Files & Manifest ────────────────────────────────────────────────

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/manifest.json", response_class=JSONResponse)
async def manifest():
    manifest_path = STATIC_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return JSONResponse(content=json.load(f))
    return JSONResponse(content={"error": "manifest not found"}, status_code=404)


@app.get("/sw.js", response_class=HTMLResponse)
async def service_worker():
    sw_path = STATIC_DIR / "sw.js"
    if sw_path.exists():
        with open(sw_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), media_type="application/javascript")
    return HTMLResponse(content="// SW not found", status_code=404)


# ─── Auth Pages ──────────────────────────────────────────────────────────────

LOGIN_PAGE_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0; min-height: 100vh; display: grid; place-items: center;
  padding: 20px env(safe-area-inset-right, 20px) env(safe-area-inset-bottom, 20px) env(safe-area-inset-left, 20px);
  background: #0a0e17; color: #f8fafc; font-family: Inter, system-ui, -apple-system, sans-serif;
}
.card {
  width: min(100%, 420px); padding: 32px; border-radius: 24px;
  background: linear-gradient(135deg, #111827, #1a1f2e);
  border: 1px solid rgba(255,255,255,.08);
  box-shadow: 0 20px 60px rgba(0,0,0,.5);
}
.logo { font-size: 48px; text-align: center; margin-bottom: 8px; }
h1 { margin: 0 0 8px; font-size: clamp(1.6rem, 5vw, 2.2rem); letter-spacing: -.03em; text-align: center; }
.subtitle { margin: 0 0 24px; color: #94a3b8; font-size: 14px; text-align: center; line-height: 1.5; }
form { display: grid; gap: 14px; }
label { display: grid; gap: 6px; font-size: 14px; color: #e2e8f0; font-weight: 500; }
input {
  width: 100%; min-height: 48px; padding: 0 16px; border-radius: 12px;
  border: 1px solid rgba(255,255,255,.12); background: #0f172a; color: #f8fafc;
  font-size: 16px; outline: none; transition: border-color .2s;
  -webkit-appearance: none;
}
input:focus { border-color: #22c55e; }
button {
  min-height: 48px; border: 0; border-radius: 14px;
  background: linear-gradient(135deg, #22c55e, #16a34a); color: #000;
  font-weight: 700; font-size: 16px; cursor: pointer; transition: all .2s;
  letter-spacing: .5px;
}
button:hover { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(34,197,94,.3); }
.error { margin: 0 0 14px; color: #fca5a5; font-size: 14px; text-align: center; }
.setup-note { margin-top: 16px; padding: 12px; background: rgba(234,179,8,.1); border: 1px solid rgba(234,179,8,.3); border-radius: 12px; font-size: 13px; color: #eab308; text-align: center; line-height: 1.5; }
"""


@app.get("/setup", response_class=HTMLResponse)
async def setup_page(error: str = ""):
    """Initial setup page — create first account."""
    if not _is_setup_needed():
        return RedirectResponse(url="/login", status_code=303)
    error_html = f'<p class="error">{error}</p>' if error else ""
    return HTMLResponse(f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Configuration · Picsou</title><link rel="icon" type="image/svg+xml" href="/static/icons/icon.svg">
<link rel="apple-touch-icon" href="/static/icons/icon-192.png"><style>{LOGIN_PAGE_CSS}</style></head><body>
<main class="card"><div class="logo">🪙</div>
<h1>Créer votre compte</h1>
<p class="subtitle">Première visite ? Créez vos identifiants pour accéder au dashboard Picsou.</p>
{error_html}
<form method="post" action="/setup">
  <label>Identifiant
    <input type="text" name="username" autocomplete="username" required minlength=3 maxlength=32 placeholder="Choisissez un identifiant" />
  </label>
  <label>Mot de passe
    <input type="password" name="password" autocomplete="new-password" required minlength=6 placeholder="Minimum 6 caractères" />
  </label>
  <label>Confirmer le mot de passe
    <input type="password" name="password_confirm" autocomplete="new-password" required minlength=6 placeholder="Retapez le mot de passe" />
  </label>
  <button type="submit">Créer le compte</button>
</form>
<div class="setup-note">🔒 Ces identifiants sont stockés localement sur le serveur. Choisissez un mot de passe solide.</div>
</main></body></html>""")


@app.post("/setup")
async def setup_submit(request: Request):
    if not _is_setup_needed():
        return RedirectResponse(url="/login", status_code=303)
    form = await request.form()
    username = str(form.get("username", "")).strip()
    password = str(form.get("password", ""))
    password_confirm = str(form.get("password_confirm", ""))
    if len(username) < 3:
        return HTMLResponse(f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Configuration · Picsou</title><link rel="icon" type="image/svg+xml" href="/static/icons/icon.svg"><style>{LOGIN_PAGE_CSS}</style></head><body><main class="card"><div class="logo">🪙</div><h1>Créer votre compte</h1><p class="error">L'identifiant doit contenir au moins 3 caractères.</p><form method="post" action="/setup"><label>Identifiant<input type="text" name="username" value="{username}" autocomplete="username" required minlength=3 /></label><label>Mot de passe<input type="password" name="password" autocomplete="new-password" required minlength=6 /></label><label>Confirmer<input type="password" name="password_confirm" autocomplete="new-password" required minlength=6 /></label><button type="submit">Créer le compte</button></form></main></body></html>""")
    if len(password) < 6:
        error = "Le mot de passe doit contenir au moins 6 caractères."
    elif password != password_confirm:
        error = "Les mots de passe ne correspondent pas."
    else:
        auth = _load_auth()
        auth["users"][username] = {"password_hash": _hash_password(password), "created_at": datetime.now(timezone.utc).isoformat()}
        auth["setup_needed"] = False
        _save_auth(auth)
        token = _create_session_token(username)
        response = RedirectResponse(url="/?v=1", status_code=303)
        response.set_cookie(AUTH_COOKIE, token, max_age=SESSION_TTL, httponly=True, samesite="lax", path="/")
        return response
    error_html = f'<p class="error">{error}</p>'
    return HTMLResponse(f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Configuration · Picsou</title><link rel="icon" type="image/svg+xml" href="/static/icons/icon.svg"><style>{LOGIN_PAGE_CSS}</style></head><body><main class="card"><div class="logo">🪙</div><h1>Créer votre compte</h1>{error_html}<form method="post" action="/setup"><label>Identifiant<input type="text" name="username" value="{username}" autocomplete="username" required minlength=3 /></label><label>Mot de passe<input type="password" name="password" autocomplete="new-password" required minlength=6 /></label><label>Confirmer<input type="password" name="password_confirm" autocomplete="new-password" required minlength=6 /></label><button type="submit">Créer le compte</button></form></main></body></html>""")


@app.get("/login", response_class=HTMLResponse)
async def login_page(error: str = "", next_path: str = "/"):
    if _is_setup_needed():
        return RedirectResponse(url="/setup", status_code=303)
    error_html = f'<p class="error">{error}</p>' if error else ""
    next_value = quote(next_path, safe="/\\%?=&")
    return HTMLResponse(f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Connexion · Picsou</title><link rel="icon" type="image/svg+xml" href="/static/icons/icon.svg">
<link rel="apple-touch-icon" href="/static/icons/icon-192.png"><meta name="theme-color" content="#0a0e17">
<style>{LOGIN_PAGE_CSS}</style></head><body>
<main class="card"><div class="logo">🪙</div>
<h1>Connexion</h1>
<p class="subtitle">Entrez vos identifiants pour accéder au dashboard.</p>
{error_html}
<form method="post" action="/login">
  <input type="hidden" name="next" value="{next_value}" />
  <label>Identifiant
    <input type="text" name="username" autocomplete="username" required autofocus />
  </label>
  <label>Mot de passe
    <input type="password" name="password" autocomplete="current-password" required />
  </label>
  <button type="submit">Se connecter</button>
</form>
</main></body></html>""")


@app.post("/login")
async def login_submit(request: Request):
    if _is_setup_needed():
        return RedirectResponse(url="/setup", status_code=303)
    form = await request.form()
    username = str(form.get("username", "")).strip()
    password = str(form.get("password", ""))
    next_path = str(form.get("next", "/"))
    auth = _load_auth()
    user = auth.get("users", {}).get(username)
    if not user or not _verify_password(password, user.get("password_hash", "")):
        return HTMLResponse(f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Connexion · Picsou</title><link rel="icon" type="image/svg+xml" href="/static/icons/icon.svg"><style>{LOGIN_PAGE_CSS}</style></head><body><main class="card"><div class="logo">🪙</div><h1>Connexion</h1><p class="error">Identifiant ou mot de passe incorrect.</p><form method="post" action="/login"><input type="hidden" name="next" value="{quote(next_path, safe='/%?=&')}" /><label>Identifiant<input type="text" name="username" value="{username}" autocomplete="username" required /></label><label>Mot de passe<input type="password" name="password" autocomplete="current-password" required /></label><button type="submit">Se connecter</button></form></main></body></html>""")
    token = _create_session_token(username)
    response = RedirectResponse(url="/?v=1", status_code=303)
    response.set_cookie(AUTH_COOKIE, token, max_age=SESSION_TTL, httponly=True, samesite="lax", path="/")
    return response


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(AUTH_COOKIE)
    return response


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_json(filename: str, default: Any = None) -> Any:
    filepath = DATA_DIR / filename
    if not filepath.exists():
        return default if default is not None else {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return default if default is not None else {}


def _uptime_seconds() -> float:
    return time.time() - START_TIME


def _format_uptime(secs: float) -> str:
    hours = int(secs // 3600)
    minutes = int((secs % 3600) // 60)
    seconds = int(secs % 60)
    return f"{hours:02d}h{minutes:02d}m{seconds:02d}s"


def _get_systemd_status() -> dict:
    """Check if picsou-v4 systemd service is active."""
    import subprocess
    result = {"active": False, "status": "unknown", "name": "picsou-v4"}
    try:
        proc = subprocess.run(
            ["systemctl", "is-active", "picsou-v4"],
            capture_output=True, text=True, timeout=5
        )
        status_text = proc.stdout.strip()
        result["status"] = status_text
        result["active"] = status_text == "active"
    except Exception:
        pass
    return result


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = TEMPLATE_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Dashboard template not found</h1>", status_code=404)
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    response = HTMLResponse(content)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/api/status")
async def get_status():
    portfolio = get_v4_portfolio()
    v4_health = await get_v4_status()

    # Merge v4 health data into status
    phase = v4_health.get("status", portfolio.get("phase", "UNKNOWN"))
    last_cycle = v4_health.get("last_heartbeat", portfolio.get("last_cycle"))
    balance = v4_health.get("balance", portfolio.get("balance", 0))
    pnl_pct = v4_health.get("pnl_pct", portfolio.get("pnl_pct", 0))

    # Systemd service status
    systemd = _get_systemd_status()

    return {
        "phase": phase,
        "uptime": _uptime_seconds(),
        "uptime_formatted": _format_uptime(_uptime_seconds()),
        "last_cycle": last_cycle,
        "kill_switch_active": _kill_switch_active,
        "v4_health": v4_health,
        "systemd": systemd,
        "balance": balance,
        "pnl_pct": pnl_pct,
        "started_at": v4_health.get("started_at"),
        "pid": v4_health.get("pid"),
        "cycle": v4_health.get("cycle"),
    }


@app.get("/api/portfolio")
async def get_portfolio():
    portfolio = get_v4_portfolio()
    return portfolio


@app.get("/api/decisions")
async def get_decisions():
    """Return recent trades as decisions."""
    portfolio = get_v4_portfolio()
    trades = portfolio.get("trades", [])
    return {"decisions": trades, "count": len(trades)}


@app.get("/api/strategies")
async def get_strategies():
    strategies = get_v4_strategies()
    return {"strategies": strategies}


@app.get("/api/learning")
async def get_learning():
    """Return learning progress from v4 data (observations + lessons + strategies)."""
    strategies = get_v4_strategies()
    observations = get_v4_observations()
    lessons = get_v4_lessons()

    # Calculate learning progress from trades in DB
    try:
        conn = _get_db()
        total_trades_row = conn.execute("SELECT COUNT(*) as cnt FROM trades WHERE status='closed'").fetchone()
        total_trades = total_trades_row["cnt"] if total_trades_row else 0
        conn.close()
    except Exception:
        total_trades = 0

    trades_required = 100  # Default graduation threshold
    progress_pct = round((total_trades / trades_required) * 100, 1) if trades_required else 0

    # Determine phase based on progress
    phase = "LEARNING"
    if total_trades >= trades_required:
        phase = "TRADING"

    # Build learning object compatible with v3 dashboard
    learning = {
        "phase": phase,
        "trades_done": total_trades,
        "trades_required": trades_required,
        "progress_pct": progress_pct,
        "graduation_threshold": trades_required,
        "started_at": None,
        "observations": observations,
        "lessons": lessons,
        "scores": strategies,
    }
    return learning


@app.get("/api/learning/strategies")
async def get_learning_strategies():
    """Return a summary per strategy from v4 SQLite."""
    strategies = get_v4_strategies()
    result = []
    for name, s in strategies.items():
        result.append({
            "name": s.get("name", name),
            "active": s.get("active", False),
            "probation": s.get("probation", False),
            "probation_trades": s.get("probation_trades", 0),
            "win_rate": round(s.get("win_rate", 0), 4),
            "avg_profit": round(s.get("avg_profit", 0), 6),
            "total_pnl": round(s.get("total_pnl", 0), 4),
            "weight": round(s.get("weight", 0), 4),
            "total_trades": s.get("total_trades", 0),
            "winning_trades": s.get("winning_trades", 0),
            "losing_trades": s.get("losing_trades", 0),
            "max_drawdown": round(s.get("max_drawdown", 0), 4),
            "sharpe_ratio": round(s.get("sharpe_ratio", 0), 4),
        })
    return {"strategies": result}


@app.get("/api/learning/backtest/{strategy}")
async def backtest_strategy(strategy: str):
    """Backtest not supported in v4 yet — return placeholder."""
    return {"error": "Backtest not yet implemented in v4", "strategy": strategy}


@app.get("/api/learning/history")
async def get_learning_history():
    """Return the evolution of strategy weights/win_rates from v4 SQLite."""
    strategies = get_v4_strategies()
    observations = get_v4_observations()

    current_scores = {}
    for name, s in strategies.items():
        current_scores[name] = {
            "weight": round(s.get("weight", 0), 4),
            "win_rate": round(s.get("win_rate", 0), 4),
            "active": s.get("active", False),
            "probation": s.get("probation", False),
            "total_trades": s.get("total_trades", 0),
        }

    # Build timeline from trades
    try:
        conn = _get_db()
        trades = conn.execute("SELECT timestamp, strategy, symbol, side, status FROM trades ORDER BY timestamp ASC").fetchall()
        conn.close()
        strategy_timeline = {}
        for t in trades:
            strat = t["strategy"] or "unknown"
            if strat not in strategy_timeline:
                strategy_timeline[strat] = []
            strategy_timeline[strat].append({
                "timestamp": t["timestamp"],
                "action": t["side"],
                "symbol": t["symbol"],
            })
    except Exception:
        strategy_timeline = {}

    result = {
        "current_scores": current_scores,
        "observations_count": len(observations),
        "journal_timeline": strategy_timeline,
    }
    return result


@app.get("/api/observations")
async def get_observations():
    """Return observations from v4 SQLite."""
    return {"observations": get_v4_observations(), "count": len(get_v4_observations())}


@app.get("/api/lessons")
async def get_lessons():
    """Return lessons from v4 SQLite."""
    return {"lessons": get_v4_lessons(), "count": len(get_v4_lessons())}


@app.get("/api/market")
async def get_market():
    """Fetch live market data from OKX public API."""
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    market = {}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for symbol in symbols:
                try:
                    resp = await client.get(
                        f"https://www.okx.com/api/v5/market/ticker",
                        params={"instId": symbol}
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("data") and len(data["data"]) > 0:
                            t = data["data"][0]
                            last_price = float(t.get("last", 0))
                            open_price = float(t.get("open24h", 0))
                            change_24h = ((last_price - open_price) / open_price * 100) if open_price > 0 else 0
                            market[symbol] = {
                                "price": last_price,
                                "change_24h": round(change_24h, 2),
                                "volume_24h": float(t.get("vol24h", 0)),
                                "high_24h": float(t.get("high24h", 0)),
                                "low_24h": float(t.get("low24h", 0)),
                            }
                except Exception:
                    pass
    except Exception:
        pass
    # Return in format expected by dashboard: {exchange: {symbol: priceData}}
    # All symbols from OKX, grouped under "okx"
    okx_market = {}
    for sym, info in market.items():
        okx_market[sym] = info
    if okx_market:
        return {"okx": okx_market}
    return {}


@app.post("/api/kill")
async def kill_switch():
    global _kill_switch_active
    _kill_switch_active = True
    kill_file = DATA_DIR / "kill_switch.json"
    kill_file.parent.mkdir(parents=True, exist_ok=True)
    with open(kill_file, "w", encoding="utf-8") as f:
        json.dump({"activated": True, "timestamp": datetime.now(timezone.utc).isoformat()}, f)
    return {"status": "KILLED", "message": "Arrêt d'urgence activé", "timestamp": datetime.now(timezone.utc).isoformat()}


# ─── Brain/LLM Config API ────────────────────────────────────────────────────

LLM_CONFIG_FILE = DATA_DIR / "llm_config.json"

MISTRAL_MODELS = [
    {"id": "mistral-small-latest", "name": "Mistral Small", "provider": "mistral"},
    {"id": "mistral-medium-latest", "name": "Mistral Medium", "provider": "mistral"},
    {"id": "mistral-large-latest", "name": "Mistral Large", "provider": "mistral"},
    {"id": "magistral-small-latest", "name": "Magistral Small (raisonnement)", "provider": "mistral"},
    {"id": "magistral-medium-latest", "name": "Magistral Medium (raisonnement)", "provider": "mistral"},
]
OLLAMA_MODELS = [
    {"id": "kimi-k2.6:cloud", "name": "Kimi K2.6", "provider": "ollama"},
    {"id": "kimi-k2.5:cloud", "name": "Kimi K2.5", "provider": "ollama"},
    {"id": "deepseek-v4-flash:cloud", "name": "DeepSeek V4 Flash", "provider": "ollama"},
    {"id": "qwen3.5:cloud", "name": "Qwen 3.5", "provider": "ollama"},
    {"id": "minimax-m2.7:cloud", "name": "MiniMax M2.7", "provider": "ollama"},
    {"id": "glm-5.1:cloud", "name": "GLM 5.1", "provider": "ollama"},
]
AVAILABLE_MODELS = MISTRAL_MODELS + OLLAMA_MODELS

MISTRAL_API_URL = "https://api.mistral.ai/v1"


def _load_llm_config() -> dict:
    """Load LLM config from llm_config.json, with defaults."""
    default = {
        "llm_model": "mistral-small-latest",
        "llm_url": "https://api.mistral.ai/v1",
        "llm_temperature": 0.3,
        "llm_max_tokens": 4096,
        "available_models": AVAILABLE_MODELS,
    }
    if not LLM_CONFIG_FILE.exists():
        return default
    try:
        with open(LLM_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in default.items():
            data.setdefault(k, v)
        return data
    except (json.JSONDecodeError, IOError):
        return default


def _save_llm_config(config: dict) -> dict:
    """Save LLM config to llm_config.json."""
    current = _load_llm_config()
    valid_model_ids = [m["id"] if isinstance(m, dict) else m for m in AVAILABLE_MODELS]
    # Only update allowed fields
    if "llm_model" in config:
        model = config["llm_model"]
        if model in valid_model_ids:
            current["llm_model"] = model
            # Auto-set URL based on provider
            model_info = next((m for m in AVAILABLE_MODELS if (m["id"] if isinstance(m, dict) else m) == model), None)
            if isinstance(model_info, dict) and model_info.get("provider") == "ollama":
                current["llm_url"] = "http://127.0.0.1:11434/v1"
            else:
                current["llm_url"] = MISTRAL_API_URL
        else:
            return {"error": f"Invalid model. Must be one of: {valid_model_ids}"}
    if "llm_temperature" in config:
        current["llm_temperature"] = max(0.0, min(2.0, float(config["llm_temperature"])))
    if "llm_max_tokens" in config:
        current["llm_max_tokens"] = max(100, min(8000, int(config["llm_max_tokens"])))
    # Always keep available_models list current
    current["available_models"] = AVAILABLE_MODELS

    LLM_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LLM_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)
    return current


@app.get("/api/brain/config")
async def get_brain_config():
    """Get current LLM brain configuration and status."""
    config = _load_llm_config()
    # Check if config file exists (agent will read it next cycle)
    config["config_file_exists"] = LLM_CONFIG_FILE.exists()
    config["message"] = "Config will take effect on next agent cycle (no restart needed)"
    return config


@app.put("/api/brain/config")
async def update_brain_config(request: Request):
    """Update LLM brain configuration (takes effect on next cycle, no restart needed)."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    result = _save_llm_config(body)
    if "error" in result:
        return JSONResponse(result, status_code=400)
    result["message"] = "Config updated. Will take effect on next agent cycle."
    return result


# ─── Agent Log Streaming ────────────────────────────────────────────────────

LOG_FILE = Path("/root/PROJECTS/picsou/v4/data/picsou_v4.log")


@app.get("/api/logs")
async def get_logs(limit: int = 100):
    """Return the last N lines of the agent log file."""
    if not LOG_FILE.exists():
        return {"lines": [], "error": "Log file not found"}
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        lines = [l.rstrip("\n") for l in all_lines[-limit:]]
        return {"lines": lines, "count": len(lines)}
    except Exception as e:
        return {"lines": [], "error": str(e)}


@app.get("/api/logs/stream")
async def stream_logs(request: Request):
    """SSE endpoint for real-time log streaming."""
    from starlette.responses import StreamingResponse
    import asyncio

    async def event_generator():
        last_pos = 0
        # Start from end of file
        if LOG_FILE.exists():
            last_pos = LOG_FILE.stat().st_size
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break
            if LOG_FILE.exists():
                try:
                    current_size = LOG_FILE.stat().st_size
                    if current_size < last_pos:
                        # File was rotated/truncated
                        last_pos = 0
                    if current_size > last_pos:
                        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                            f.seek(last_pos)
                            new_data = f.read()
                            last_pos = f.tell()
                        for line in new_data.splitlines():
                            if line.strip():
                                yield f"data: {line}\n\n"
                except Exception:
                    pass
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            portfolio = get_v4_portfolio()
            v4_health = await get_v4_status()
            phase = v4_health.get("status", portfolio.get("phase", "UNKNOWN"))
            positions = portfolio.get("positions", [])
            pos_count = len(positions)
            balance = float(portfolio.get("balance", 0.0))
            starting_capital = float(portfolio.get("starting_capital", 10000.0))
            total_pnl = float(portfolio.get("pnl", 0.0))
            payload = {
                "type": "update",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": {
                    "phase": phase,
                    "uptime": _uptime_seconds(),
                    "uptime_formatted": _format_uptime(_uptime_seconds()),
                    "kill_switch_active": _kill_switch_active,
                },
                "portfolio_summary": {
                    "balance": balance,
                    "pnl": total_pnl,
                    "pnl_pct": (total_pnl / starting_capital * 100) if starting_capital else 0.0,
                    "position_count": pos_count,
                },
                "v4_health": v4_health,
            }
            await websocket.send_json(payload)
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3037)