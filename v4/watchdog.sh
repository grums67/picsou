#!/bin/bash
# Picsou v4 watchdog — restarts if crashed, checks health every 60s
# This script runs OUTSIDE the agent and cannot be modified by it.

PIDFILE=/root/PROJECTS/picsou/v4/data/picsou.pid
LOGFILE=/root/PROJECTS/picsou/v4/data/picsou_v4.log
RESTART_COUNT_FILE=/root/PROJECTS/picsou/v4/data/restart_count
MAX_RESTARTS_PER_HOUR=6
HEALTH_URL="http://localhost:3035/api/health"

mkdir -p /root/PROJECTS/picsou/v4/data

# Initialize restart counter
echo "0" > "$RESTART_COUNT_FILE.$$" 2>/dev/null

check_health() {
    # Try HTTP health check first
    response=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null)
    if [ "$response" = "200" ] || [ "$response" = "401" ]; then
        return 0  # Healthy (401 = auth required = server up)
    fi

    # Fallback: check if process exists
    if [ -f "$PIDFILE" ]; then
        pid=$(cat "$PIDFILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1  # Unhealthy
}

restart_agent() {
    echo "[$(date -Iseconds)] Restarting Picsou v4..." >> "$LOGFILE"
    
    # Kill old process if any
    if [ -f "$PIDFILE" ]; then
        old_pid=$(cat "$PIDFILE")
        kill "$old_pid" 2>/dev/null
        sleep 2
        kill -9 "$old_pid" 2>/dev/null
    fi
    
    # Start fresh
    cd /root/PROJECTS/picsou/v4
    export PICSOU_LLM_KEY="${PICSOU_LLM_KEY:-}"
    
    # Source .env
    if [ -f /root/PROJECTS/picsou/.env ]; then
        set -a
        source /root/PROJECTS/picsou/.env 2>/dev/null
        set +a
    fi
    
    nohup python3 run.py >> "$LOGFILE" 2>&1 &
    new_pid=$!
    echo "$new_pid" > "$PIDFILE"
    echo "[$(date -Iseconds)] Started Picsou v4 (PID=$new_pid)" >> "$LOGFILE"
}

# Main loop
while true; do
    if ! check_health; then
        echo "[$(date -Iseconds)] Health check FAILED — restarting" >> "$LOGFILE"
        restart_agent
        sleep 30  # Give it time to start
    fi
    sleep 60
done