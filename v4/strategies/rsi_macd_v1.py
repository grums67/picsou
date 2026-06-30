# Picsou v4 — Stratégie RSI + MACD
# Détection de retournements et momentum avec confirmation multiple

def signal(market_data, portfolio, memory):
    """Stratégie combinant RSI et MACD pour des signaux plus fiables.
    
    - RSI < 30 + MACD haussier → achat (oversold reversal)
    - RSI > 70 + MACD baissier → vente (overbought reversal)
    - MACD crossover seul → signal modéré
    """
    # Chercher le premier symbole avec assez de données
    symbol = None
    candles = []
    for key, md in market_data.items():
        cs = md.get("candles", [])
        if len(cs) >= 50:
            symbol = key
            candles = cs
            break
    
    if not symbol or len(candles) < 50:
        return {"action": "hold", "reasoning": "Pas assez de données (besoin 50 bougies)"}
    
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    volumes = [c.get("volume", 0) for c in candles]
    
    current_price = closes[-1]
    current_volume = volumes[-1]
    avg_volume = sum(volumes[-20:]) / 20
    
    # --- Calcul RSI (période 14) ---
    def calc_rsi(prices, period=14):
        gains, losses = 0, 0
        for i in range(1, period + 1):
            diff = prices[-i] - prices[-i-1]
            if diff > 0:
                gains += diff
            else:
                losses += abs(diff)
        avg_gain = gains / period
        avg_loss = losses / period
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    rsi = calc_rsi(closes)
    
    # --- Calcul MACD (12, 26, 9) ---
    def ema(data, period):
        multiplier = 2.0 / (period + 1)
        ema_val = sum(data[:period]) / period
        for price in data[period:]:
            ema_val = (price - ema_val) * multiplier + ema_val
        return ema_val
    
    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    macd_line = ema12 - ema26
    
    # Signal line = EMA9 of MACD line
    # On recalcule la MACD line sur toute la série pour avoir l'historique
    macd_values = []
    for i in range(26, len(closes)):
        e12 = ema(closes[:i+1], 12)
        e26 = ema(closes[:i+1], 26)
        macd_values.append(e12 - e26)
    
    if len(macd_values) >= 9:
        signal_line = ema(macd_values, 9)
        macd_histogram = macd_values[-1] - signal_line
        prev_macd = macd_values[-2] if len(macd_values) > 1 else macd_values[-1]
        prev_signal = ema(macd_values[:-1], 9) if len(macd_values) > 10 else signal_line
        macd_cross = macd_values[-1] > signal_line and prev_macd <= prev_signal
        macd_cross_bear = macd_values[-1] < signal_line and prev_macd >= prev_signal
    else:
        signal_line = 0
        macd_histogram = 0
        macd_cross = False
        macd_cross_bear = False
    
    # --- Vérifier les positions existantes ---
    base_symbol = symbol.split(":")[-1].split("-")[0]
    has_position = False
    for pos_id, pos in portfolio.get("positions", {}).items():
        if base_symbol in pos.get("symbol", ""):
            has_position = True
            break
    
    # --- Logique de décision ---
    volume_ok = current_volume > avg_volume * 0.5  # Pas de volume anormalement bas
    
    # SIGNAL ACHAT : RSI oversold + MACD haussier
    if rsi < 35 and macd_cross and volume_ok and not has_position:
        confidence = 0.6 if rsi < 30 else 0.5
        return {
            "action": "buy",
            "symbol": base_symbol,
            "confidence": confidence,
            "size_pct": 0.05,
            "strategy": "rsi_macd_v1",
            "reasoning": f"RSI({rsi:.1f}) oversold + MACD haussier sur {symbol}"
        }
    
    # SIGNAL ACHAT : MACD crossover seul (momentum)
    if macd_cross and volume_ok and not has_position and rsi < 60:
        return {
            "action": "buy",
            "symbol": base_symbol,
            "confidence": 0.45,
            "size_pct": 0.04,
            "strategy": "rsi_macd_v1",
            "reasoning": f"MACD crossover haussier sur {symbol} (RSI={rsi:.1f})"
        }
    
    # SIGNAL VENTE : RSI overbought + MACD baissier
    if rsi > 70 and macd_cross_bear and has_position:
        return {
            "action": "sell",
            "symbol": base_symbol,
            "confidence": 0.65,
            "size_pct": 0.05,
            "strategy": "rsi_macd_v1",
            "reasoning": f"RSI({rsi:.1f}) overbought + MACD baissier sur {symbol}"
        }
    
    # SIGNAL VENTE : MACD baissier seul
    if macd_cross_bear and has_position and rsi > 50:
        return {
            "action": "sell",
            "symbol": base_symbol,
            "confidence": 0.5,
            "size_pct": 0.05,
            "strategy": "rsi_macd_v1",
            "reasoning": f"MACD baissier sur {symbol} (RSI={rsi:.1f})"
        }
    
    return {"action": "hold", "reasoning": f"RSI={rsi:.1f}, MACD histo={macd_histogram:.2f}, pas de signal clair"}


def metadata():
    return {
        "name": "rsi_macd_v1",
        "version": "1.0",
        "type": "mean_reversion_momentum",
        "description": "RSI + MACD combinés pour signaux de retournement et momentum",
        "created_by": "picsou_agent",
    }
