#!/usr/bin/env python3
"""Clean Telegram pending updates before starting the bot."""
import urllib.request
import json
import os

# Load token from .env
token = os.environ.get("PICSOU_TELEGRAM_TOKEN", "")
if not token:
    with open("/root/PROJECTS/picsou/.env") as f:
        for line in f:
            line = line.strip()
            if line.startswith("PICSOU_TELEGRAM_TOKEN"):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

if not token:
    print("ERROR: No token found")
    exit(1)

print(f"Bot token: {token[:10]}...{token[-4:]}")

# 1. Delete webhook + drop pending
url1 = f"https://api.telegram.org/bot{token}/deleteWebhook?drop_pending_updates=true"
resp = urllib.request.urlopen(url1, timeout=10)
print(f"deleteWebhook: {json.loads(resp.read())}")

# 2. Get and clear pending updates  
url2 = f"https://api.telegram.org/bot{token}/getUpdates?timeout=1"
resp2 = urllib.request.urlopen(url2, timeout=5)
data = json.loads(resp2.read())
updates = data.get("result", [])
print(f"Pending updates: {len(updates)}")
if updates:
    max_id = max(u["update_id"] for u in updates)
    url3 = f"https://api.telegram.org/bot{token}/getUpdates?offset={max_id+1}&timeout=1"
    urllib.request.urlopen(url3, timeout=5)
    print(f"Cleared updates up to {max_id}")

print("Telegram queue cleared - ready to start bot")