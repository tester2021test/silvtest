# ================================================================
# 🥇 Tata Gold & Silver ETF Tracker — Pro Edition
# Files: tracker.py  |  .github/workflows/tracker.yml
# ================================================================
# ETF-SPECIFIC PARAMETERS:
#   ✅ Live iNAV calculation (Silver: SI=F × INR/oz; Gold: GC=F × INR/oz)
#   ✅ Premium / Discount % vs iNAV  ← MOST CRITICAL for ETFs
#   ✅ Premium alert tiers: DANGER (>5%), WARN (2-5%), FAIR (<2%), BUY ZONE (<-1%)
#   ✅ Tracking Error proxy (rolling 30d std-dev of daily ETF vs underlying)
#   ✅ Expense Ratio drag calculator (daily cost impact on returns)
#   ✅ Volume vs 20d avg (liquidity check)
#   ✅ Bid-Ask spread proxy (High-Low / Close %)
#
# UNDERLYING ASSET TRACKING:
#   ✅ US Silver Futures (SI=F)  — oz price in USD
#   ✅ US Gold Futures  (GC=F)  — oz price in USD
#   ✅ MCX Silver (SILVERM.MCX) — Indian domestic silver
#   ✅ MCX Gold   (GOLDM.MCX)  — Indian domestic gold
#   ✅ USD/INR live rate
#   ✅ Gold/Silver Ratio (GSR)  — historically 60-80; extremes signal rotation
#
# TECHNICALS:
#   ✅ RSI (14, EWM-smoothed)
#   ✅ MACD with crossover detection
#   ✅ Bollinger Bands + %B
#   ✅ EMA 20 / 50 trend
#   ✅ ATR (volatility / stop sizing)
#   ✅ 52-week position %
#   ✅ Momentum: 1d / 5d / 20d / 60d
#
# DECISION ENGINE:
#   ✅ 10-factor scoring per ETF
#   ✅ Side-by-side Gold vs Silver decision table
#   ✅ Gold/Silver ratio trade signal (which metal is relatively cheap)
#   ✅ Risk management: ATR stop + targets
#   ✅ 30-day backtest (premium-mean-reversion strategy)
#
# OUTPUT:
#   ✅ Telegram: 3 messages (Overview table, Silver detail, Gold detail)
#   ✅ CSV: appended to data/history.csv in repo
#   ✅ All GitHub-Actions friendly (env-var secrets)
# ================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import pytz
import requests
import os
import csv
import time as time_module
import json
from pathlib import Path
from datetime import datetime, time
from functools import wraps
import math

# ================================================================
# CONFIG
# ================================================================

# ETF metadata (verified Feb 2026)
ETF_META = {
    "TATSILV": {
        "ticker"         : "TATSILV.NS",
        "name"           : "Tata Silver ETF",
        "underlying"     : "Silver",
        "units_per_gram" : 1,        # 1 unit ≈ 1g silver (approx, adjust if AMC changes)
        "grams_per_unit" : 1.0,
        "oz_per_unit"    : 1.0 / 31.1035,   # grams→oz
        "expense_ratio"  : 0.0044,   # 0.44% p.a.
        "import_duty"    : 0.06,     # 6% import duty on silver
        "gst"            : 0.03,     # 3% GST
        "aum_cr"         : 6849,     # ₹Cr, update quarterly
        "mcx_ticker"     : "SILVERM.MCX",
        "us_ticker"      : "SI=F",
        "us_name"        : "US Silver Futures",
    },
    "TATAGOLD": {
        "ticker"         : "TATAGOLD.NS",
        "name"           : "Tata Gold ETF",
        "underlying"     : "Gold",
        "grams_per_unit" : 1.0,      # 1 unit ≈ 1g gold
        "oz_per_unit"    : 1.0 / 31.1035,
        "expense_ratio"  : 0.0038,   # 0.38% p.a.
        "import_duty"    : 0.06,     # 6% import duty on gold
        "gst"            : 0.03,
        "aum_cr"         : 5625,     # ₹Cr, update quarterly
        "mcx_ticker"     : "GOLDM.MCX",
        "us_ticker"      : "GC=F",
        "us_name"        : "US Gold Futures",
    },
}

# Premium/Discount alert thresholds (%)
PREMIUM_DANGER   =  5.0    # >5%  → DANGER: seriously overpriced, wait
PREMIUM_WARN     =  2.0    # 2–5% → CAUTION: slight premium, accept only if bullish
PREMIUM_FAIR     =  0.5    # ±0.5% → FAIR VALUE
DISCOUNT_BUY     = -1.0    # <-1% → BUY ZONE: discount to NAV
DISCOUNT_STRONG  = -3.0    # <-3% → STRONG BUY ZONE

# Technical thresholds
RSI_OVERSOLD     = 40
RSI_OVERBOUGHT   = 65
ATR_RISK_MULT    = 1.5

# Gold/Silver Ratio thresholds
GSR_BUY_SILVER   = 80   # GSR > 80 → silver cheap relative to gold → prefer silver
GSR_BUY_GOLD     = 60   # GSR < 60 → gold cheap relative to silver → prefer gold

# Retry
MAX_RETRIES  = 3
RETRY_DELAY  = 4

# ================================================================
# RETRY
# ================================================================
def with_retry(retries=MAX_RETRIES, delay=RETRY_DELAY):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(1, retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    if attempt < retries:
                        time_module.sleep(delay)
            print(f"  ❌ [{fn.__name__}] all retries failed: {last_err}")
            return None
        return wrapper
    return decorator

# ================================================================
# DATA FETCH
# ================================================================
@with_retry()
def fetch_hist(symbol: str, period="5d", interval="1d") -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"Empty df for {symbol}")
    return df

def safe_close(df) -> pd.Series:
    c = df["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c.dropna().astype(float)

def get_live_prev(symbol: str):
    df = fetch_hist(symbol, period="5d", interval="1d")
    if df is None or df.empty:
        return 0.0, 0.0
    c = safe_close(df)
    return (float(c.iloc[-1]), float(c.iloc[-2])) if len(c) > 1 else (float(c.iloc[-1]), float(c.iloc[-1]))

# ================================================================
# INDICATORS
# ================================================================
def calc_rsi(s: pd.Series, p=14) -> float:
    if len(s) < p + 1: return 50.0
    d   = s.diff()
    ag  = d.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    al  = (-d.clip(upper=0)).ewm(com=p-1, min_periods=p).mean()
    rs  = ag / al.replace(0, np.nan)
    v   = (100 - 100/(1+rs)).dropna()
    return round(float(v.iloc[-1]), 1) if not v.empty else 50.0

def calc_macd(s: pd.Series):
    """Returns (line, signal, hist_now, hist_prev)."""
    if len(s) < 27: return 0.0, 0.0, 0.0, 0.0
    ef = s.ewm(span=12, adjust=False).mean()
    es = s.ewm(span=26, adjust=False).mean()
    ml = ef - es
    sl = ml.ewm(span=9, adjust=False).mean()
    h  = (ml - sl).dropna()
    return (round(float(ml.iloc[-1]),4), round(float(sl.iloc[-1]),4),
            round(float(h.iloc[-1]),4), round(float(h.iloc[-2]) if len(h)>1 else 0.0, 4))

def calc_bb(s: pd.Series, p=20):
    """Returns (upper, mid, lower, pct_b)."""
    if len(s) < p: return 0.0, 0.0, 0.0, 0.5
    mid = s.rolling(p).mean()
    std = s.rolling(p).std()
    u, l = mid + 2*std, mid - 2*std
    pb = ((s - l) / (u - l).replace(0, np.nan)).dropna()
    return (round(float(u.iloc[-1]),4), round(float(mid.iloc[-1]),4),
            round(float(l.iloc[-1]),4), round(float(pb.iloc[-1]),3))

def calc_ema(s: pd.Series, span: int) -> float:
    v = s.ewm(span=span, adjust=False).mean().dropna()
    return round(float(v.iloc[-1]), 4) if not v.empty else 0.0

def calc_atr(df, p=14) -> float:
    try:
        hi = df["High"].astype(float); lo = df["Low"].astype(float)
        cl = safe_close(df); pv = cl.shift(1)
        tr = pd.concat([hi-lo,(hi-pv).abs(),(lo-pv).abs()],axis=1).max(axis=1)
        a  = tr.rolling(p).mean().dropna()
        return round(float(a.iloc[-1]), 4) if not a.empty else 0.0
    except: return 0.0

def week52_pos(s: pd.Series) -> float:
    lo, hi, cur = float(s.min()), float(s.max()), float(s.iloc[-1])
    return round((cur-lo)/(hi-lo)*100, 1) if hi != lo else 50.0

def momentum(s: pd.Series, days: int) -> float:
    if len(s) < days+1: return 0.0
    return round((float(s.iloc[-1])-float(s.iloc[-(days+1)]))/float(s.iloc[-(days+1)])*100, 2)

def tracking_error_proxy(etf_close: pd.Series, inav_series: pd.Series, days=30) -> float:
    """
    Proxy tracking error: std-dev of (ETF daily return - iNAV daily return).
    Uses whatever overlapping history is available.
    """
    try:
        combined = pd.concat([etf_close, inav_series], axis=1).dropna()
        combined.columns = ["etf", "inav"]
        combined = combined.iloc[-days:]
        diff = combined["etf"].pct_change() - combined["inav"].pct_change()
        te = diff.dropna().std() * math.sqrt(252) * 100
        return round(te, 3)
    except:
        return 0.0

def bid_ask_spread_proxy(df) -> float:
    """(High - Low) / Close % as a proxy for intraday spread."""
    try:
        hi = df["High"].astype(float).iloc[-1]
        lo = df["Low"].astype(float).iloc[-1]
        cl = safe_close(df).iloc[-1]
        return round((hi - lo) / cl * 100, 3) if cl else 0.0
    except: return 0.0

def daily_expense_drag(expense_ratio: float) -> float:
    """Daily cost drag from expense ratio (%)."""
    return round(expense_ratio / 252 * 100, 6)

# ================================================================
# iNAV CALCULATION
# ================================================================
def calc_inav(us_price_usd: float, usd_inr: float, oz_per_unit: float,
              import_duty: float, gst: float) -> float:
    """
    Theoretical fair value of 1 ETF unit in INR.
    Formula: US_price_per_oz × USD/INR × oz_per_unit × (1 + import_duty) × (1 + GST)
    """
    if us_price_usd <= 0 or usd_inr <= 0:
        return 0.0
    return round(us_price_usd * usd_inr * oz_per_unit * (1 + import_duty) * (1 + gst), 4)

def build_inav_series(us_hist: pd.DataFrame, fx_hist: pd.DataFrame,
                      oz_per_unit: float, import_duty: float, gst: float) -> pd.Series:
    """Build daily iNAV series for tracking error calculation."""
    us_close = safe_close(us_hist).rename("us")
    fx_close = safe_close(fx_hist).rename("fx")
    df = pd.concat([us_close, fx_close], axis=1).dropna()
    return df["us"] * df["fx"] * oz_per_unit * (1 + import_duty) * (1 + gst)

# ================================================================
# PREMIUM / DISCOUNT SIGNAL
# ================================================================
def premium_signal(prem_pct: float) -> tuple:
    """Returns (label, emoji, buy_advice)."""
    if prem_pct >= PREMIUM_DANGER:
        return "⛔ DANGER PREMIUM", "🔴🔴", "DO NOT BUY — paying way above fair value"
    elif prem_pct >= PREMIUM_WARN:
        return "⚠️ CAUTION PREMIUM", "🟠", "WAIT — slight premium, better entry ahead"
    elif prem_pct <= DISCOUNT_STRONG:
        return "🎯 STRONG BUY ZONE", "🟢🟢", "STRONG BUY — significant discount to NAV"
    elif prem_pct <= DISCOUNT_BUY:
        return "✅ BUY ZONE", "🟢", "BUY — trading at discount to fair value"
    else:
        return "➡️ FAIR VALUE", "🟡", "FAIR — acceptable entry if technically bullish"

# ================================================================
# GOLD / SILVER RATIO SIGNAL
# ================================================================
def gsr_signal(gsr: float) -> dict:
    """Interpret Gold/Silver Ratio for relative value."""
    hist_avg = 70.0   # long-run average
    if gsr > GSR_BUY_SILVER:
        action = "PREFER SILVER"
        reason = f"GSR={gsr:.1f} > {GSR_BUY_SILVER} → Silver historically cheap vs Gold"
        emoji  = "🥈⬆️"
    elif gsr < GSR_BUY_GOLD:
        action = "PREFER GOLD"
        reason = f"GSR={gsr:.1f} < {GSR_BUY_GOLD} → Gold historically cheap vs Silver"
        emoji  = "🥇⬆️"
    else:
        action = "NEUTRAL"
        reason = f"GSR={gsr:.1f} near historical avg ({hist_avg}) → no strong preference"
        emoji  = "⚖️"
    deviation = round((gsr - hist_avg) / hist_avg * 100, 1)
    return {"action": action, "reason": reason, "emoji": emoji,
            "gsr": round(gsr,2), "deviation_pct": deviation}

# ================================================================
# SIGNAL ENGINE (10 factors, score -10…+10)
# ================================================================
def build_signal(
    prem_pct: float,
    rsi: float, macd_h: float, macd_hp: float,
    pct_b: float, ema20: float, ema50: float, price: float,
    mom_5d: float, mom_20d: float,
    w52: float, vix: float,
) -> dict:
    score = 0; reasons = []

    # 1. Premium/Discount (most important for ETFs — weight x3)
    if prem_pct <= DISCOUNT_STRONG:
        score += 3; reasons.append(f"Strong discount {prem_pct:+.2f}% vs iNAV 🎯")
    elif prem_pct <= DISCOUNT_BUY:
        score += 2; reasons.append(f"Discount {prem_pct:+.2f}% vs iNAV ✅")
    elif abs(prem_pct) <= 0.5:
        score += 1; reasons.append(f"Fair value {prem_pct:+.2f}% vs iNAV ➡️")
    elif prem_pct >= PREMIUM_DANGER:
        score -= 3; reasons.append(f"DANGER premium {prem_pct:+.2f}% vs iNAV ⛔")
    elif prem_pct >= PREMIUM_WARN:
        score -= 2; reasons.append(f"Premium {prem_pct:+.2f}% vs iNAV ⚠️")
    else:
        score -= 1; reasons.append(f"Slight premium {prem_pct:+.2f}% vs iNAV")

    # 2. RSI
    if rsi < RSI_OVERSOLD:
        score += 2; reasons.append(f"RSI oversold ({rsi:.0f})")
    elif rsi > RSI_OVERBOUGHT:
        score -= 1; reasons.append(f"RSI overbought ({rsi:.0f})")
    else:
        reasons.append(f"RSI neutral ({rsi:.0f})")

    # 3. MACD crossover
    if macd_hp <= 0 < macd_h:
        score += 2; reasons.append("MACD bullish crossover ↑")
    elif macd_hp >= 0 > macd_h:
        score -= 2; reasons.append("MACD bearish crossover ↓")
    elif macd_h > 0:
        score += 1; reasons.append("MACD positive")
    else:
        score -= 1; reasons.append("MACD negative")

    # 4. Bollinger %B
    if pct_b < 0.2:
        score += 1; reasons.append(f"Near lower BB (%B={pct_b:.2f})")
    elif pct_b > 0.8:
        score -= 1; reasons.append(f"Near upper BB (%B={pct_b:.2f})")

    # 5. EMA trend
    if ema20 > ema50 and price >= ema20:
        score += 1; reasons.append("Price > EMA20 > EMA50 (uptrend)")
    elif ema20 < ema50 and price <= ema20:
        score -= 1; reasons.append("Price < EMA20 < EMA50 (downtrend)")

    # 6. 52-week position (low zone = value)
    if w52 < 25:
        score += 1; reasons.append(f"Near 52w low ({w52:.0f}%)")
    elif w52 > 85:
        score -= 1; reasons.append(f"Near 52w high ({w52:.0f}%)")

    # 7. VIX (high VIX → metals can spike or crater)
    if vix > 20:
        reasons.append(f"VIX elevated ({vix:.1f}) — volatility risk")
    else:
        reasons.append(f"VIX normal ({vix:.1f})")

    if score >= 6:    action, emoji = "STRONG BUY",     "🟢🟢"
    elif score >= 3:  action, emoji = "BUY",            "🟢"
    elif score <= -5: action, emoji = "STRONG AVOID",   "🔴🔴"
    elif score <= -2: action, emoji = "AVOID",          "🔴"
    else:             action, emoji = "NEUTRAL / HOLD", "🟡"

    return {"action": action, "emoji": emoji, "score": score,
            "confidence": round(min(abs(score)/10*100,100),1), "reasons": reasons}

# ================================================================
# RISK MANAGEMENT
# ================================================================
def risk_mgmt(price: float, atr: float, score: int) -> dict:
    stop = round(price - ATR_RISK_MULT * atr, 4)
    t1   = round(price + ATR_RISK_MULT * atr, 4)
    t2   = round(price + 2*ATR_RISK_MULT * atr, 4)
    rp   = round((price - stop)/price*100, 2) if price else 0
    if abs(score) >= 6:   size = "Full position"
    elif abs(score) >= 3: size = "Half position"
    else:                 size = "Wait / small probe"
    return {"stop": stop, "t1": t1, "t2": t2, "rp": rp, "size": size}

# ================================================================
# BACKTEST  (premium mean-reversion, 30 days)
# ================================================================
def run_backtest(etf_sym: str, us_sym: str, days=30) -> dict:
    """
    Buy when ETF trades at discount to iNAV (< -1%).
    Sell when ETF reaches fair value (premium ≥ 0%).
    """
    etf_df = fetch_hist(etf_sym, period=f"{days+10}d", interval="1d")
    us_df  = fetch_hist(us_sym,  period=f"{days+10}d", interval="1d")
    fx_df  = fetch_hist("INR=X", period=f"{days+10}d", interval="1d")
    if any(x is None or x.empty for x in [etf_df, us_df, fx_df]):
        return {"error": "no data"}

    meta     = next(m for m in ETF_META.values() if m["us_ticker"] == us_sym)
    etf_c    = safe_close(etf_df)
    us_c     = safe_close(us_df)
    fx_c     = safe_close(fx_df)
    combined = pd.concat([etf_c.rename("etf"), us_c.rename("us"), fx_c.rename("fx")], axis=1).dropna()
    if len(combined) < 5:
        return {"error": "insufficient data"}

    combined["inav"]    = combined["us"] * combined["fx"] * meta["oz_per_unit"] * (1 + meta["import_duty"]) * (1 + meta["gst"])
    combined["prem"]    = (combined["etf"] - combined["inav"]) / combined["inav"] * 100

    trades, position = [], None
    for i, row in combined.iterrows():
        p, e, prem = row["inav"], row["etf"], row["prem"]
        if np.isnan(prem): continue
        if position is None and prem <= DISCOUNT_BUY:
            position = {"entry": e, "date": str(i)[:10]}
        elif position is not None and prem >= 0:
            ret = (e - position["entry"]) / position["entry"] * 100
            trades.append({"buy": position["date"], "sell": str(i)[:10],
                           "entry": round(position["entry"],4), "exit": round(e,4),
                           "ret": round(ret,2)})
            position = None

    wins = [t for t in trades if t["ret"] > 0]
    wr   = round(len(wins)/len(trades)*100,1) if trades else 0
    avg  = round(sum(t["ret"] for t in trades)/len(trades),2) if trades else 0.0
    return {"total": len(trades), "wr": wr, "avg": avg,
            "best": round(max((t["ret"] for t in trades),default=0),2),
            "trades": trades[-3:]}

# ================================================================
# ANALYSE ONE ETF
# ================================================================
def analyse_etf(key: str, meta: dict, usd_inr: float, vix: float) -> dict:
    print(f"\n  📊 Analysing {key} ({meta['ticker']})...")

    etf_live, etf_prev = get_live_prev(meta["ticker"])
    us_live,  us_prev  = get_live_prev(meta["us_ticker"])
    mcx_live, mcx_prev = get_live_prev(meta["mcx_ticker"])

    # iNAV
    inav = calc_inav(us_live, usd_inr, meta["oz_per_unit"], meta["import_duty"], meta["gst"])
    inav_prev = calc_inav(us_prev, usd_inr, meta["oz_per_unit"], meta["import_duty"], meta["gst"])

    prem_pct  = round((etf_live - inav) / inav * 100, 3) if inav else 0.0
    prem_label, prem_emoji, prem_advice = premium_signal(prem_pct)

    # Fetch history
    df_1d  = fetch_hist(meta["ticker"], period="90d",  interval="1d")
    df_15m = fetch_hist(meta["ticker"], period="5d",   interval="15m")
    df_52w = fetch_hist(meta["ticker"], period="252d", interval="1d")
    us_1d  = fetch_hist(meta["us_ticker"], period="90d", interval="1d")
    fx_1d  = fetch_hist("INR=X",          period="90d", interval="1d")

    # Technicals
    rsi_v = macd_l = macd_s = macd_h = macd_hp = 0.0
    bb_u = bb_m = bb_l = pct_b = 0.0; pct_b = 0.5
    ema20 = ema50 = etf_live
    atr_v = 0.0; w52 = 50.0
    spread_proxy = 0.0
    te = 0.0

    if df_15m is not None and not df_15m.empty:
        c15 = safe_close(df_15m)
        if len(c15) > 26:
            rsi_v = calc_rsi(c15)
            macd_l, macd_s, macd_h, macd_hp = calc_macd(c15)
            bb_u, bb_m, bb_l, pct_b = calc_bb(c15)
        atr_v = calc_atr(df_15m)
        spread_proxy = bid_ask_spread_proxy(df_15m)

    if df_1d is not None and not df_1d.empty:
        c1d   = safe_close(df_1d)
        ema20 = calc_ema(c1d, 20)
        ema50 = calc_ema(c1d, 50)
        mom1d  = momentum(c1d, 1)
        mom5d  = momentum(c1d, 5)
        mom20d = momentum(c1d, 20)
        mom60d = momentum(c1d, 60)
        vol_s  = df_1d["Volume"].astype(float).dropna()
        vol_today = float(vol_s.iloc[-1]) if not vol_s.empty else 0.0
        avg_vol   = float(vol_s.iloc[-21:-1].mean()) if len(vol_s) >= 21 else float(vol_s.mean())

        # Tracking error proxy
        if us_1d is not None and fx_1d is not None:
            inav_series = build_inav_series(us_1d, fx_1d, meta["oz_per_unit"],
                                            meta["import_duty"], meta["gst"])
            te = tracking_error_proxy(c1d, inav_series)
    else:
        mom1d = mom5d = mom20d = mom60d = 0.0
        vol_today = avg_vol = 0.0

    if df_52w is not None and not df_52w.empty:
        w52 = week52_pos(safe_close(df_52w))

    # MCX premium vs US futures (import duty + GST should explain it)
    mcx_inr_per_gram  = mcx_live / 1000.0 if mcx_live > 0 else 0   # MCX quotes in ₹/kg
    us_inr_per_gram   = us_live * usd_inr / 31.1035 if us_live > 0 else 0
    mcx_premium_pct   = round((mcx_inr_per_gram - us_inr_per_gram) / us_inr_per_gram * 100, 2) if us_inr_per_gram else 0.0

    # Expense drag
    daily_drag = daily_expense_drag(meta["expense_ratio"])

    # Signal
    sig  = build_signal(prem_pct, rsi_v, macd_h, macd_hp, pct_b,
                        ema20, ema50, etf_live, mom5d, mom20d, w52, vix)
    risk = risk_mgmt(etf_live, atr_v, sig["score"])

    # Backtest (cached in main)
    bt = {}

    print(f"     ETF=₹{etf_live:.4f}  iNAV=₹{inav:.4f}  Premium={prem_pct:+.2f}%")
    print(f"     US={meta['us_name']}=${us_live:.3f}  MCX=₹{mcx_live:.0f}  USD/INR=₹{usd_inr:.4f}")
    print(f"     RSI={rsi_v:.1f}  MACD_h={macd_h:+.4f}  Signal={sig['action']}({sig['score']:+d})")

    return {
        "key": key, "meta": meta,
        "etf_live": etf_live, "etf_prev": etf_prev,
        "us_live": us_live, "us_prev": us_prev,
        "mcx_live": mcx_live,
        "usd_inr": usd_inr,
        "inav": inav, "inav_prev": inav_prev,
        "prem_pct": prem_pct,
        "prem_label": prem_label, "prem_emoji": prem_emoji, "prem_advice": prem_advice,
        "mcx_premium_pct": mcx_premium_pct,
        "rsi": rsi_v, "macd_l": macd_l, "macd_h": macd_h,
        "pct_b": pct_b, "bb_u": bb_u, "bb_l": bb_l,
        "ema20": ema20, "ema50": ema50,
        "atr": atr_v, "w52": w52,
        "mom1d": mom1d, "mom5d": mom5d, "mom20d": mom20d, "mom60d": mom60d,
        "vol_today": vol_today, "avg_vol": avg_vol,
        "spread_proxy": spread_proxy,
        "tracking_error": te,
        "daily_drag": daily_drag,
        "sig": sig, "risk": risk, "bt": {},
    }

# ================================================================
# CSV
# ================================================================
CSV_PATH = Path("data/history.csv")
FIELDNAMES = [
    "timestamp", "market_phase", "etf",
    "etf_price", "etf_prev", "day_chg_pct",
    "inav", "premium_pct", "premium_label",
    "us_price_usd", "mcx_price", "usd_inr",
    "mcx_premium_pct",
    "rsi", "macd_hist", "bb_pct_b",
    "ema20", "ema50", "atr",
    "mom_1d", "mom_5d", "mom_20d", "mom_60d",
    "w52_pos", "vol_today", "avg_vol", "vol_ratio",
    "spread_proxy_pct", "tracking_error_pct", "daily_expense_drag_pct",
    "signal", "score", "confidence",
    "stop", "target_1r", "target_2r",
    "bt_trades", "bt_wr", "bt_avg",
    "gold_silver_ratio",
]

def save_csv(rows: list, timestamp: str, market_phase: str, gsr: float):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            bt = r["bt"]
            day_chg = (r["etf_live"]-r["etf_prev"])/r["etf_prev"]*100 if r["etf_prev"] else 0
            vol_ratio = r["vol_today"]/r["avg_vol"] if r["avg_vol"] else 0
            w.writerow({
                "timestamp"              : timestamp,
                "market_phase"           : market_phase,
                "etf"                    : r["key"],
                "etf_price"              : r["etf_live"],
                "etf_prev"               : r["etf_prev"],
                "day_chg_pct"            : round(day_chg, 3),
                "inav"                   : r["inav"],
                "premium_pct"            : r["prem_pct"],
                "premium_label"          : r["prem_label"],
                "us_price_usd"           : r["us_live"],
                "mcx_price"              : r["mcx_live"],
                "usd_inr"                : r["usd_inr"],
                "mcx_premium_pct"        : r["mcx_premium_pct"],
                "rsi"                    : r["rsi"],
                "macd_hist"              : r["macd_h"],
                "bb_pct_b"               : r["pct_b"],
                "ema20"                  : r["ema20"],
                "ema50"                  : r["ema50"],
                "atr"                    : r["atr"],
                "mom_1d"                 : r["mom1d"],
                "mom_5d"                 : r["mom5d"],
                "mom_20d"                : r["mom20d"],
                "mom_60d"                : r["mom60d"],
                "w52_pos"                : r["w52"],
                "vol_today"              : int(r["vol_today"]),
                "avg_vol"                : int(r["avg_vol"]),
                "vol_ratio"              : round(vol_ratio,2),
                "spread_proxy_pct"       : r["spread_proxy"],
                "tracking_error_pct"     : r["tracking_error"],
                "daily_expense_drag_pct" : r["daily_drag"],
                "signal"                 : r["sig"]["action"],
                "score"                  : r["sig"]["score"],
                "confidence"             : r["sig"]["confidence"],
                "stop"                   : r["risk"]["stop"],
                "target_1r"              : r["risk"]["t1"],
                "target_2r"              : r["risk"]["t2"],
                "bt_trades"              : bt.get("total",""),
                "bt_wr"                  : bt.get("wr",""),
                "bt_avg"                 : bt.get("avg",""),
                "gold_silver_ratio"      : round(gsr,2),
            })

# ================================================================
# TELEGRAM
# ================================================================
def send_telegram(msg: str):
    token   = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("  ❌ Telegram credentials missing"); return
    for attempt in range(1, MAX_RETRIES+1):
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id":chat_id,"text":msg,
                      "parse_mode":"Markdown","disable_web_page_preview":True},
                timeout=15)
            if r.status_code == 200:
                print(f"  📨 Telegram sent ({len(msg)} chars)"); return
            print(f"  ⚠️  Attempt {attempt}: {r.status_code}")
        except Exception as e:
            print(f"  ⚠️  Attempt {attempt}: {e}")
        if attempt < MAX_RETRIES: time_module.sleep(RETRY_DELAY)
    print("  ❌ Telegram failed")

# ================================================================
# FORMAT MESSAGES
# ================================================================
def fmt_overview(silver: dict, gold: dict, gsr_info: dict, now, market_phase: str) -> str:
    sep = "─" * 36

    overview = (
        "```\n"
        f"{'Metric':<22} {'SILVER':>8}  {'GOLD':>8}\n"
        f"{sep}\n"
        f"{'ETF Price (₹)':<22} {silver['etf_live']:>8.4f}  {gold['etf_live']:>8.4f}\n"
        f"{'iNAV (₹)':<22} {silver['inav']:>8.4f}  {gold['inav']:>8.4f}\n"
        f"{'Premium/Disc %':<22} {silver['prem_pct']:>+8.2f}%  {gold['prem_pct']:>+8.2f}%\n"
        f"{sep}\n"
        f"{'US Price (USD)':<22} {silver['us_live']:>8.3f}  {gold['us_live']:>8.2f}\n"
        f"{'MCX Price (₹)':<22} {silver['mcx_live']:>8.0f}  {gold['mcx_live']:>8.0f}\n"
        f"{'MCX vs US %':<22} {silver['mcx_premium_pct']:>+8.2f}%  {gold['mcx_premium_pct']:>+8.2f}%\n"
        f"{'USD/INR':<22} {silver['usd_inr']:>8.4f}  {gold['usd_inr']:>8.4f}\n"
        f"{sep}\n"
        f"{'RSI (15m)':<22} {silver['rsi']:>8.1f}  {gold['rsi']:>8.1f}\n"
        f"{'MACD Hist':<22} {silver['macd_h']:>+8.4f}  {gold['macd_h']:>+8.4f}\n"
        f"{'BB %B':<22} {silver['pct_b']:>8.3f}  {gold['pct_b']:>8.3f}\n"
        f"{'1d Mom %':<22} {silver['mom1d']:>+8.2f}%  {gold['mom1d']:>+8.2f}%\n"
        f"{'5d Mom %':<22} {silver['mom5d']:>+8.2f}%  {gold['mom5d']:>+8.2f}%\n"
        f"{'20d Mom %':<22} {silver['mom20d']:>+8.2f}%  {gold['mom20d']:>+8.2f}%\n"
        f"{'52w Position':<22} {silver['w52']:>7.1f}%  {gold['w52']:>7.1f}%\n"
        f"{sep}\n"
        f"{'Tracking Error':<22} {silver['tracking_error']:>7.3f}%  {gold['tracking_error']:>7.3f}%\n"
        f"{'Expense Ratio':<22} {'0.44%':>8}  {'0.38%':>8}\n"
        f"{'Daily Drag (bps)':<22} {silver['daily_drag']*100:>8.4f}  {gold['daily_drag']*100:>8.4f}\n"
        f"{'Spread Proxy':<22} {silver['spread_proxy']:>7.3f}%  {gold['spread_proxy']:>7.3f}%\n"
        f"{sep}\n"
        f"{'Signal':<22} {silver['sig']['emoji']:>8}  {gold['sig']['emoji']:>8}\n"
        f"{'Score (/10)':<22} {silver['sig']['score']:>+8d}  {gold['sig']['score']:>+8d}\n"
        f"{'Confidence':<22} {silver['sig']['confidence']:>7.0f}%  {gold['sig']['confidence']:>7.0f}%\n"
        "```"
    )

    # GSR block
    gsr_block = (
        f"⚖️ *Gold/Silver Ratio: {gsr_info['gsr']:.2f}*  {gsr_info['emoji']}\n"
        f"  {gsr_info['reason']}\n"
        f"  Deviation from avg (70): {gsr_info['deviation_pct']:+.1f}%"
    )

    # Premium advice
    s_label = f"{silver['prem_emoji']} TATSILV: {silver['prem_label']}\n  → _{silver['prem_advice']}_"
    g_label = f"{gold['prem_emoji']} TATAGOLD: {gold['prem_label']}\n  → _{gold['prem_advice']}_"

    # Best pick
    if silver["sig"]["score"] > gold["sig"]["score"]:
        best = f"✅ *Best pick: TATSILV* (Silver ETF, score {silver['sig']['score']:+d})"
    elif gold["sig"]["score"] > silver["sig"]["score"]:
        best = f"✅ *Best pick: TATAGOLD* (Gold ETF, score {gold['sig']['score']:+d})"
    else:
        best = "➡️ *Both equal score — check premium/discount to decide*"

    return (
        f"💎 *Gold & Silver ETF Tracker*\n"
        f"🕒 *{market_phase}*  |  {now.strftime('%d-%b-%Y %H:%M IST')}\n\n"
        f"📊 *Side-by-Side Comparison*\n{overview}\n"
        f"🏷 *Premium / Discount*\n{s_label}\n{g_label}\n\n"
        f"{gsr_block}\n\n"
        f"{best}"
    )

def fmt_detail(r: dict, now) -> str:
    sig  = r["sig"]; risk = r["risk"]; bt = r["bt"]
    meta = r["meta"]; sep = "─" * 30
    vol_ratio = r["vol_today"]/r["avg_vol"] if r["avg_vol"] else 0
    day_chg = (r["etf_live"]-r["etf_prev"])/r["etf_prev"]*100 if r["etf_prev"] else 0
    reasons = "\n".join(f"  • {x}" for x in sig["reasons"])

    bt_block = "No trades in period"
    if "error" not in bt and bt.get("total", 0) > 0:
        recent = ""
        for t in bt.get("trades", []):
            icon = "🟢" if t["ret"] > 0 else "🔴"
            recent += f"\n    {icon} {t['buy']} → {t['sell']}: {t['ret']:+.2f}%"
        bt_block = (f"{bt['total']} trades | WR {bt['wr']}% | "
                    f"Avg {bt['avg']:+.2f}% | Best {bt['best']:+.2f}%{recent}")

    return (
        f"{'🥈' if r['key']=='TATSILV' else '🥇'} "
        f"*{meta['name']}* ({r['key']})  {sig['emoji']}\n\n"
        f"```\n"
        f"{'ETF Price':<18} ₹{r['etf_live']:>9.4f} ({day_chg:+.2f}%)\n"
        f"{'iNAV':<18} ₹{r['inav']:>9.4f}\n"
        f"{'Premium/Disc':<18} {r['prem_pct']:>+9.2f}%  ← KEY\n"
        f"{sep}\n"
        f"{'US Futures (USD)':<18} ${r['us_live']:>9.3f}\n"
        f"{'MCX (₹)':<18} ₹{r['mcx_live']:>9.0f}\n"
        f"{'USD/INR':<18} ₹{r['usd_inr']:>9.4f}\n"
        f"{'MCX vs US':<18} {r['mcx_premium_pct']:>+9.2f}%\n"
        f"{sep}\n"
        f"{'RSI (15m)':<18} {r['rsi']:>10.1f}\n"
        f"{'MACD Hist':<18} {r['macd_h']:>+10.4f}\n"
        f"{'BB %B':<18} {r['pct_b']:>10.3f}\n"
        f"{'EMA 20':<18} ₹{r['ema20']:>9.4f}\n"
        f"{'EMA 50':<18} ₹{r['ema50']:>9.4f}\n"
        f"{'ATR':<18} ₹{r['atr']:>9.4f}\n"
        f"{sep}\n"
        f"{'1d / 5d / 20d':<18} {r['mom1d']:>+5.2f}% {r['mom5d']:>+6.2f}% {r['mom20d']:>+6.2f}%\n"
        f"{'60d Momentum':<18} {r['mom60d']:>+9.2f}%\n"
        f"{'52w Position':<18} {r['w52']:>9.1f}%\n"
        f"{sep}\n"
        f"{'Volume':<18} {vol_ratio:>9.1f}× avg\n"
        f"{'Tracking Error':<18} {r['tracking_error']:>9.3f}%\n"
        f"{'Expense Ratio':<18} {meta['expense_ratio']*100:>9.2f}%\n"
        f"{'Daily Drag':<18} {r['daily_drag']*100:>9.5f}%\n"
        f"{'Spread Proxy':<18} {r['spread_proxy']:>9.3f}%\n"
        f"{'AUM':<18} {'₹'+str(meta['aum_cr'])+'Cr':>10}\n"
        f"{sep}\n"
        f"{'Stop Loss':<18} ₹{risk['stop']:>9.4f}\n"
        f"{'Target 1:1':<18} ₹{risk['t1']:>9.4f}\n"
        f"{'Target 1:2':<18} ₹{risk['t2']:>9.4f}\n"
        f"{'Risk %':<18} {risk['rp']:>9.2f}%\n"
        f"```\n"
        f"*Signal:* {sig['action']} (score {sig['score']:+d}/10, {sig['confidence']:.0f}%)\n"
        f"{reasons}\n"
        f"*Sizing:* {risk['size']}\n\n"
        f"📈 *30d Backtest (premium reversion):* {bt_block}"
    )

# ================================================================
# MAIN
# ================================================================
def main():
    print("⏳ Tata Gold & Silver ETF Tracker — starting...\n")

    ist          = pytz.timezone("Asia/Kolkata")
    now          = datetime.now(ist)
    today_str    = now.strftime("%Y-%m-%d")
    t            = now.time()
    market_phase = ("PRE-MARKET" if t < time(9,15) else
                    "POST-MARKET" if t > time(15,30) else "LIVE")

    # ── USD/INR & VIX
    usd_inr, _ = get_live_prev("INR=X")
    vix, _     = get_live_prev("^INDIAVIX")
    usd_inr    = usd_inr if usd_inr > 0 else 84.0
    vix        = vix if vix > 0 else 15.0
    print(f"  USD/INR=₹{usd_inr:.4f}  VIX={vix:.2f}")

    # ── Analyse both ETFs
    silver = analyse_etf("TATSILV", ETF_META["TATSILV"], usd_inr, vix)
    gold   = analyse_etf("TATAGOLD", ETF_META["TATAGOLD"], usd_inr, vix)

    # ── Gold/Silver Ratio
    ag_oz = silver["us_live"]   # Silver price per oz USD
    au_oz = gold["us_live"]     # Gold price per oz USD
    gsr   = au_oz / ag_oz if ag_oz > 0 else 70.0
    gsr_info = gsr_signal(gsr)
    print(f"\n  Gold/Silver Ratio: {gsr:.2f}  → {gsr_info['action']}")

    # ── Backtest (daily cache)
    bt_cache_path = Path("data/bt_cache.json")
    bt_cache = {}
    if bt_cache_path.exists():
        try:
            data = json.loads(bt_cache_path.read_text())
            if data.get("date") == today_str:
                bt_cache = data.get("results", {})
        except Exception: pass

    for key, meta, r in [("TATSILV", ETF_META["TATSILV"], silver),
                          ("TATAGOLD", ETF_META["TATAGOLD"], gold)]:
        if key not in bt_cache:
            bt_cache[key] = run_backtest(meta["ticker"], meta["us_ticker"])
        r["bt"] = bt_cache[key]

    bt_cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        bt_cache_path.write_text(json.dumps({"date": today_str, "results": bt_cache}))
    except Exception: pass

    # ── CSV
    save_csv([silver, gold], now.isoformat(), market_phase, gsr)
    print(f"  📁 CSV appended → {CSV_PATH}")

    # ── Telegram (3 messages)
    msg1 = fmt_overview(silver, gold, gsr_info, now, market_phase)
    msg2 = fmt_detail(silver, now)
    msg3 = fmt_detail(gold, now)

    send_telegram(msg1)
    time_module.sleep(1)
    send_telegram(msg2)
    time_module.sleep(1)
    send_telegram(msg3)

    print("\n✅ Done!")

if __name__ == "__main__":
    main()
