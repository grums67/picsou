# Picsou v4 — Stratégie de démarrage simple
# L'agent part de zéro. Cette stratégie minimaliste sert de point de départ.
# Le LLM créera les vraies stratégies.

def signal(market_data, portfolio, memory):
    """Stratégie EMA simple — signal d'entrée basé sur croisement EMA9/EMA21.
    
    Point de départ minimal. L'agent créera de meilleures stratégies.
    """
    # Prendre le premier symbole disponible
    symbol = None
    candles = []
    for key, md in market_data.items():
        candles = md.get("candles", [])
        if len(candles) >= 21:
            symbol = key
            break
    
    if not symbol or len(candles) < 21:
        return {"action": "hold", "reasoning": "Pas assez de données"}
    
    # Calculer EMA9 et EMA21
    closes = [c["close"] for c in candles]
    
    def ema(data, period):
        multiplier = 2.0 / (period + 1)
        ema_val = sum(data[:period]) / period
        for price in data[period:]:
            ema_val = (price - ema_val) * multiplier + ema_val
        return ema_val
    
    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    prev_ema9 = ema(closes[:-1], 9) if len(closes) > 22 else ema9
    prev_ema21 = ema(closes[:-1], 21) if len(closes) > 22 else ema21
    
    current_price = closes[-1]
    
    # Croisement haussier → buy
    if prev_ema9 <= prev_ema21 and ema9 > ema21:
        return {
            "action": "buy",
            "symbol": symbol.split(":")[-1].split("-")[0],
            "confidence": 0.5,
            "size_pct": 0.05,
            "strategy": "ema_crossover_v1",
            "reasoning": f"Croisement EMA9/EMA21 haussier sur {symbol}"
        }
    
    # Croisement baissier → sell si position ouverte
    if prev_ema9 >= prev_ema21 and ema9 < ema21:
        return {
            "action": "sell",
            "symbol": symbol.split(":")[-1].split("-")[0],
            "confidence": 0.5,
            "size_pct": 0.05,
            "strategy": "ema_crossover_v1",
            "reasoning": f"Croisement EMA9/EMA21 baissier sur {symbol}"
        }
    
    return {"action": "hold", "reasoning": "Pas de signal EMA"}


def metadata():
    return {
        "name": "ema_crossover_v1",
        "version": "0.1",
        "type": "momentum",
        "description": "Croisement EMA9/EMA21 — stratégie de démarrage",
        "created_by": "human",
    }