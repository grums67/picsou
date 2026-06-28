#!/bin/bash
set -e

PROJECT_DIR="/root/PROJECTS/picsou"
VENV_DIR="$PROJECT_DIR/venv"

cd "$PROJECT_DIR"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "🔧 Création de l'environnement virtuel..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Install requirements
echo "📦 Installation des dépendances..."
pip install -q -r requirements.txt

# Ensure data directory exists
mkdir -p "$PROJECT_DIR/data"

# Graceful shutdown handler
PIDS=()
shutdown() {
    echo ""
    echo "🛑 Arrêt de Picsou..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "✅ Picsou arrêté proprement."
    exit 0
}
trap shutdown SIGINT SIGTERM

# Start dashboard
echo "🚀 Démarrage du dashboard sur le port 3037..."
uvicorn dashboard.app:app --host 127.0.0.1 --port 3037 &
PIDS+=($!)

# Start picsou agent cycle (if module exists)
if [ -d "$PROJECT_DIR/src" ] || [ -f "$PROJECT_DIR/src/picsou.py" ]; then
    echo "🤖 Démarrage de l'agent Picsou..."
    python -m src.picsou &
    PIDS+=($!)
else
    echo "⚠️  Module src.picsou non trouvé, agent non démarré (dashboard uniquement)"
fi

echo ""
echo "✨ Picsou est en marche !"
echo "   Dashboard: http://localhost:3037"
echo ""

# Wait for any process to exit
wait -n 2>/dev/null || wait