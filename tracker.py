# ================================================================
# 💎 Tata Gold & Silver ETF Tracker — v4
# ================================================================
# TATSILV.NS  — Tata Silver ETF FoF
# TATAGOLD.NS — Tata Gold ETF
#
# iNAV CALCULATION (pure COMEX, no proxy ETFs needed):
#   1. Get live price USD/oz  from COMEX: SI=F (silver) / GC=F (gold)
#   2. Convert to ₹/gram: (USD/oz × USD/INR) / 31.1035
#   3. Apply Indian landed cost: × (1 + 6% import duty) × (1 + 3% GST)
#   4. Derive grams_per_unit = median(ETF_price / ₹_per_gram) over 30d
#      → self-calibrating; handles FoF unit sizes automatically
#   5. iNAV = ₹_per_gram × grams_per_unit
#   6. Premium/Discount = (ETF_live − iNAV) / iNAV × 100
#
# Removed: SILVERBEES.NS, GOLDBEES.NS, SILVERM.MCX, GOLDM.MCX
# Fixed:   pd.concat() sort=False (no more Pandas4Warning)
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
import math
from pathlib import Path
from datetime import datetime, time
from functools import wraps

# ── ETF CONFIG ──────────────────────────────────────────────────
ETF_META = {
    "TATSILV": {
        "ticker"       : "TATSILV.NS",
        "name"         : "Tata Silver ETF FoF",
        "underlying"   : "Silver",
        "comex_ticker" : "SI=F",
        "comex_name"   : "COMEX Silver",
        "expense_ratio": 0.0044,
        "aum_cr"       : 6849,
    },
    "TATAGOLD": {
        "ticker"       : "TATAGOLD.NS",
        "name"         : "Tata Gold ETF",
        "underlying"   : "Gold",
        "comex_ticker" : "GC=F",
        "comex_name"   : "COMEX Gold",
        "expense_ratio": 0.0038,
        "aum_cr"       : 5625,
    },
}

OZ_TO_GRAM      = 31.1035
IMPORT_DUTY     = 0.06
GST             = 0.03

PREMIUM_DANGER  =  5.0
PREMIUM_WARN    =  2.0
DISCOUNT_BUY    = -1.0
DISCOUNT_STRONG = -3.0
RSI_OVERSOLD    = 40
RSI_OVERBOUGHT  = 65
ATR_RISK_MULT   = 1.5
GSR_BUY_SILVER  = 80
GSR_BUY_GOLD    = 60
GSR_HIST_AVG    = 70.0
MAX_RETRIES     = 3
RETRY_DELAY     = 4

# ── RETRY ───────────────────────────────────────────────────────
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

# ── FETCH ───────────────────────────────────────────────────────
@with_retry()
def fetch_hist(symbol, period="5d", interval="1d"):
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"Empty df for {symbol}")
    return df

def safe_close(df):
    c = df["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c.dropna().astype(float)

def get_live_prev(symbol):
    df = fetch_hist(symbol, period="5d", interval="1d")
    if df is None or df.empty:
        return 0.0, 0.0
    c = safe_close(df)
    return (float(c.iloc[-1]), float(c.iloc[-2])) if len(c) > 1 else (float(c.iloc[-1]), float(c.iloc[-1]))

# ── iNAV (pure COMEX) ───────────────────────────────────────────
def comex_to_inr_per_gram(usd_per_oz, usd_inr):
    """USD/oz → ₹/gram including import duty + GST."""
    if usd_per_oz <= 0 or usd_inr <= 0:
        return 0.0
    return round(usd_per_oz * usd_inr / OZ_TO_GRAM * (1 + IMPORT_DUTY) * (1 + GST), 6)

def build_ipg_series(comex_df, fx_df):
    """Daily ₹/gram series from COMEX + FX history."""
    c = safe_close(comex_df).rename("comex")
    f = safe_close(fx_df).rename("fx")
    df = pd.concat([c, f], axis=1, sort=False).dropna()
    return df["comex"] * df["fx"] / OZ_TO_GRAM * (1 + IMPORT_DUTY) * (1 + GST)

def calc_grams_per_unit(etf_hist, ipg_series, days=30):
    """Dynamic grams/unit = median(ETF_price / ₹_per_gram) over last N days."""
    combined = pd.concat(
        [etf_hist.rename("etf"), ipg_series.rename("ipg")],
        axis=1, sort=False
    ).dropna().iloc[-days:]
    if len(combined) < 3:
        return 0.0
    ratio = (combined["etf"] / combined["ipg"]).replace([np.inf, -np.inf], np.nan).dropna()
    return float(ratio.median()) if not ratio.empty else 0.0

def tracking_error(etf_hist, ipg_series, days=30):
    """Annualised tracking error: std-dev of (ETF% − underlying%)."""
    try:
        c = pd.concat(
            [etf_hist.rename("e"), ipg_series.rename("p")],
            axis=1, sort=False
        ).dropna().iloc[-days:]
        diff = c["e"].pct_change() - c["p"].pct_change()
        return round(diff.dropna().std() * math.sqrt(252) * 100, 3)
    except:
        return 0.0

# ── INDICATORS ──────────────────────────────────────────────────
def calc_rsi(s, p=14):
    if len(s) < p + 1: return 50.0
    d  = s.diff()
    ag = d.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    al = (-d.clip(upper=0)).ewm(com=p-1, min_periods=p).mean()
    rs = ag / al.replace(0, np.nan)
    v  = (100 - 100 / (1 + rs)).dropna()
    return round(float(v.iloc[-1]), 1) if not v.empty else 50.0

def calc_macd(s):
    if len(s) < 27: return 0.0, 0.0, 0.0, 0.0
    ef = s.ewm(span=12, adjust=False).mean()
    es = s.ewm(span=26, adjust=False).mean()
    ml = ef - es
    sl = ml.ewm(span=9, adjust=False).mean()
    h  = (ml - sl).dropna()
    return (round(float(ml.iloc[-1]), 4), round(float(sl.iloc[-1]), 4),
            round(float(h.iloc[-1]), 4),
            round(float(h.iloc[-2]) if len(h) > 1 else 0.0, 4))

def calc_bb(s, p=20):
    if len(s) < p: return 0.0, 0.0, 0.0, 0.5
    mid = s.rolling(p).mean(); std = s.rolling(p).std()
    u = mid + 2*std; l = mid - 2*std
    pb = ((s - l) / (u - l).replace(0, np.nan)).dropna()
    return (round(float(u.iloc[-1]), 4), round(float(mid.iloc[-1]), 4),
            round(float(l.iloc[-1]), 4), round(float(pb.iloc[-1]), 3))

def calc_ema(s, span):
    v = s.ewm(span=span, adjust=False).mean().dropna()
    return round(float(v.iloc[-1]), 4) if not v.empty else 0.0

def calc_atr(df, p=14):
    try:
        hi = df["High"].astype(float); lo = df["Low"].astype(float)
        cl = safe_close(df); pv = cl.shift(1)
        tr = pd.concat([hi-lo,(hi-pv).abs(),(lo-pv).abs()], axis=1, sort=False).max(axis=1)
        a  = tr.rolling(p).mean().dropna()
        return round(float(a.iloc[-1]), 4) if not a.empty else 0.0
    except: return 0.0

def week52_pos(s):
    lo, hi, cur = float(s.min()), float(s.max()), float(s.iloc[-1])
    return round((cur-lo)/(hi-lo)*100, 1) if hi != lo else 50.0

def momentum(s, days):
    if len(s) < days+1: return 0.0
    base = float(s.iloc[-(days+1)])
    return round((float(s.iloc[-1])-base)/base*100, 2) if base else 0.0

def spread_proxy(df):
    try:
        last = df.iloc[-1]
        hi, lo, cl = float(last["High"]), float(last["Low"]), float(last["Close"])
        return round((hi-lo)/cl*100, 3) if cl else 0.0
    except: return 0.0

def daily_expense_drag(er):
    return round(er / 252 * 100, 6)

# ── PREMIUM SIGNAL ──────────────────────────────────────────────
def premium_signal(p):
    if p >= PREMIUM_DANGER:    return "⛔ DANGER PREMIUM",  "🔴🔴","DO NOT BUY — far above fair value"
    elif p >= PREMIUM_WARN:    return "⚠️ CAUTION PREMIUM", "🟠",  "WAIT — premium likely to compress"
    elif p <= DISCOUNT_STRONG: return "🎯 STRONG BUY ZONE", "🟢🟢","STRONG BUY — significant discount"
    elif p <= DISCOUNT_BUY:    return "✅ BUY ZONE",         "🟢",  "BUY — trading below fair value"
    else:                      return "➡️ FAIR VALUE",        "🟡",  "FAIR — acceptable if technically bullish"

# ── GSR SIGNAL ──────────────────────────────────────────────────
def gsr_signal(gsr):
    if gsr > GSR_BUY_SILVER:
        a,r,e = "PREFER SILVER", f"GSR={gsr:.1f} > {GSR_BUY_SILVER} → Silver cheap vs Gold", "🥈⬆️"
    elif gsr < GSR_BUY_GOLD:
        a,r,e = "PREFER GOLD",   f"GSR={gsr:.1f} < {GSR_BUY_GOLD} → Gold cheap vs Silver",   "🥇⬆️"
    else:
        a,r,e = "NEUTRAL",       f"GSR={gsr:.1f} near historical avg ({GSR_HIST_AVG})",        "⚖️"
    return {"action":a,"reason":r,"emoji":e,"gsr":round(gsr,2),
            "deviation_pct":round((gsr-GSR_HIST_AVG)/GSR_HIST_AVG*100,1)}

# ── SIGNAL ENGINE ───────────────────────────────────────────────
def build_signal(prem,rsi,mh,mhp,pctb,e20,e50,price,m5,m20,w52,vix):
    score=0; reasons=[]
    if prem<=DISCOUNT_STRONG:   score+=3; reasons.append(f"Strong discount {prem:+.2f}% 🎯")
    elif prem<=DISCOUNT_BUY:    score+=2; reasons.append(f"Discount {prem:+.2f}% ✅")
    elif abs(prem)<=0.5:        score+=1; reasons.append(f"Fair value {prem:+.2f}% ➡️")
    elif prem>=PREMIUM_DANGER:  score-=3; reasons.append(f"DANGER premium {prem:+.2f}% ⛔")
    elif prem>=PREMIUM_WARN:    score-=2; reasons.append(f"Premium {prem:+.2f}% ⚠️")
    else:                       score-=1; reasons.append(f"Slight premium {prem:+.2f}%")
    if rsi<RSI_OVERSOLD:        score+=2; reasons.append(f"RSI oversold ({rsi:.0f})")
    elif rsi>RSI_OVERBOUGHT:    score-=1; reasons.append(f"RSI overbought ({rsi:.0f})")
    else:                                 reasons.append(f"RSI neutral ({rsi:.0f})")
    if mhp<=0<mh:               score+=2; reasons.append("MACD bullish crossover ↑")
    elif mhp>=0>mh:             score-=2; reasons.append("MACD bearish crossover ↓")
    elif mh>0:                  score+=1; reasons.append("MACD positive")
    else:                       score-=1; reasons.append("MACD negative")
    if pctb<0.20:               score+=1; reasons.append(f"Near lower BB (%B={pctb:.2f})")
    elif pctb>0.80:             score-=1; reasons.append(f"Near upper BB (%B={pctb:.2f})")
    if e20>e50 and price>=e20:  score+=1; reasons.append("Price > EMA20 > EMA50 ↑")
    elif e20<e50 and price<=e20:score-=1; reasons.append("Price < EMA20 < EMA50 ↓")
    if w52<25:                  score+=1; reasons.append(f"Near 52w low ({w52:.0f}%)")
    elif w52>85:                score-=1; reasons.append(f"Near 52w high ({w52:.0f}%)")
    if vix>20: reasons.append(f"VIX elevated ({vix:.1f}) ⚠️")
    else:      reasons.append(f"VIX normal ({vix:.1f})")
    if score>=6:    a,e="STRONG BUY","🟢🟢"
    elif score>=3:  a,e="BUY","🟢"
    elif score<=-5: a,e="STRONG AVOID","🔴🔴"
    elif score<=-2: a,e="AVOID","🔴"
    else:           a,e="NEUTRAL / HOLD","🟡"
    return {"action":a,"emoji":e,"score":score,
            "confidence":round(min(abs(score)/10*100,100),1),"reasons":reasons}

# ── RISK ────────────────────────────────────────────────────────
def risk_mgmt(price,atr,score):
    stop=round(price-ATR_RISK_MULT*atr,4)
    t1=round(price+ATR_RISK_MULT*atr,4); t2=round(price+2*ATR_RISK_MULT*atr,4)
    rp=round((price-stop)/price*100,2) if price else 0
    if abs(score)>=6:   size="Full position"
    elif abs(score)>=3: size="Half position"
    else:               size="Wait / small probe"
    return {"stop":stop,"t1":t1,"t2":t2,"rp":rp,"size":size}

# ── BACKTEST ────────────────────────────────────────────────────
def run_backtest(etf_sym, comex_sym, days=40):
    """Buy at discount ≤ -1%, sell when premium normalises ≥ 0%."""
    etf_df   = fetch_hist(etf_sym,   period=f"{days+10}d", interval="1d")
    comex_df = fetch_hist(comex_sym, period=f"{days+10}d", interval="1d")
    fx_df    = fetch_hist("INR=X",   period=f"{days+10}d", interval="1d")
    if any(x is None or x.empty for x in [etf_df, comex_df, fx_df]):
        return {"error":"no data"}
    etf_c   = safe_close(etf_df)
    ipg_s   = build_ipg_series(comex_df, fx_df)
    combined = pd.concat([etf_c.rename("etf"),ipg_s.rename("ipg")],axis=1,sort=False).dropna()
    if len(combined)<5: return {"error":"short"}
    half  = max(3, len(combined)//2)
    scale = float((combined["etf"]/combined["ipg"]).iloc[:half].median())
    combined["inav"] = combined["ipg"]*scale
    combined["prem"] = (combined["etf"]-combined["inav"])/combined["inav"]*100
    trades,position = [],None
    for i,row in combined.iterrows():
        if np.isnan(row["prem"]): continue
        if position is None and row["prem"]<=DISCOUNT_BUY:
            position={"entry":row["etf"],"date":str(i)[:10]}
        elif position is not None and row["prem"]>=0:
            ret=(row["etf"]-position["entry"])/position["entry"]*100
            trades.append({"buy":position["date"],"sell":str(i)[:10],
                           "entry":round(position["entry"],4),"exit":round(row["etf"],4),"ret":round(ret,2)})
            position=None
    if not trades: return {"total":0,"wr":0,"avg":0.0,"best":0.0,"trades":[]}
    wins=[t for t in trades if t["ret"]>0]
    wr=round(len(wins)/len(trades)*100,1); avg=round(sum(t["ret"] for t in trades)/len(trades),2)
    return {"total":len(trades),"wr":wr,"avg":avg,
            "best":round(max(t["ret"] for t in trades),2),"trades":trades[-3:]}

# ── ANALYSE ETF ─────────────────────────────────────────────────
def analyse_etf(key, meta, usd_inr, vix):
    print(f"\n  📊 {key} ({meta['ticker']})...")
    etf_live,etf_prev     = get_live_prev(meta["ticker"])
    comex_live,comex_prev = get_live_prev(meta["comex_ticker"])

    df_1d    = fetch_hist(meta["ticker"],    period="90d",  interval="1d")
    df_15m   = fetch_hist(meta["ticker"],    period="5d",   interval="15m")
    df_52w   = fetch_hist(meta["ticker"],    period="252d", interval="1d")
    comex_1d = fetch_hist(meta["comex_ticker"], period="90d", interval="1d")
    fx_1d    = fetch_hist("INR=X",           period="90d", interval="1d")

    # ₹/gram series from COMEX history
    ipg_series = pd.Series(dtype=float)
    if comex_1d is not None and not comex_1d.empty and fx_1d is not None and not fx_1d.empty:
        ipg_series = build_ipg_series(comex_1d, fx_1d)

    inr_per_gram_live = comex_to_inr_per_gram(comex_live, usd_inr)
    etf_hist = safe_close(df_1d) if df_1d is not None and not df_1d.empty else pd.Series([etf_live])

    grams_per_unit = calc_grams_per_unit(etf_hist, ipg_series) if not ipg_series.empty else 0.0
    inav           = round(inr_per_gram_live * grams_per_unit, 4) if grams_per_unit > 0 else 0.0
    prem_pct       = round((etf_live-inav)/inav*100, 2) if inav > 0 else 0.0
    pl,pe,pa       = premium_signal(prem_pct)
    te             = tracking_error(etf_hist, ipg_series) if not ipg_series.empty else 0.0

    rsi_v=macd_l=macd_s=macd_h=macd_hp=0.0
    bb_u=bb_m=bb_l=0.0; pct_b=0.5
    ema20=ema50=etf_live; atr_v=sp=0.0
    m1=m5=m20=m60=0.0; w52=50.0; vt=av=0.0

    if df_15m is not None and not df_15m.empty:
        c15=safe_close(df_15m)
        if len(c15)>26:
            rsi_v=calc_rsi(c15); macd_l,macd_s,macd_h,macd_hp=calc_macd(c15)
            bb_u,bb_m,bb_l,pct_b=calc_bb(c15)
        atr_v=calc_atr(df_15m); sp=spread_proxy(df_15m)

    if df_1d is not None and not df_1d.empty:
        c1d=safe_close(df_1d)
        ema20=calc_ema(c1d,20); ema50=calc_ema(c1d,50)
        m1=momentum(c1d,1); m5=momentum(c1d,5); m20=momentum(c1d,20); m60=momentum(c1d,60)
        vol_s=df_1d["Volume"].astype(float).dropna()
        vt=float(vol_s.iloc[-1]) if not vol_s.empty else 0.0
        av=float(vol_s.iloc[-21:-1].mean()) if len(vol_s)>=21 else float(vol_s.mean())

    if df_52w is not None and not df_52w.empty:
        w52=week52_pos(safe_close(df_52w))

    day_chg=(etf_live-etf_prev)/etf_prev*100 if etf_prev else 0.0
    sig=build_signal(prem_pct,rsi_v,macd_h,macd_hp,pct_b,ema20,ema50,etf_live,m5,m20,w52,vix)
    risk=risk_mgmt(etf_live,atr_v,sig["score"])

    print(f"     ETF=₹{etf_live:.4f}  iNAV=₹{inav:.4f}  Prem={prem_pct:+.2f}%  {pe}")
    print(f"     COMEX={meta['comex_name']} ${comex_live:.3f}/oz  ₹/g={inr_per_gram_live:.2f}  g/unit={grams_per_unit:.6f}")
    print(f"     RSI={rsi_v:.1f}  MACD_h={macd_h:+.4f}  {sig['action']}({sig['score']:+d})")

    return {"key":key,"meta":meta,
            "etf_live":etf_live,"etf_prev":etf_prev,"day_chg":round(day_chg,3),
            "comex_live":comex_live,"comex_prev":comex_prev,
            "inr_per_gram":inr_per_gram_live,"grams_per_unit":round(grams_per_unit,6),
            "usd_inr":usd_inr,"inav":inav,
            "prem_pct":prem_pct,"prem_label":pl,"prem_emoji":pe,"prem_advice":pa,
            "rsi":rsi_v,"macd_l":macd_l,"macd_h":macd_h,
            "pct_b":pct_b,"bb_u":bb_u,"bb_l":bb_l,
            "ema20":ema20,"ema50":ema50,"atr":atr_v,
            "m1":m1,"m5":m5,"m20":m20,"m60":m60,
            "w52":w52,"vt":vt,"av":av,"sp":sp,"te":te,
            "daily_drag":daily_expense_drag(meta["expense_ratio"]),
            "sig":sig,"risk":risk,"bt":{}}

# ── CSV ─────────────────────────────────────────────────────────
CSV_PATH=Path("data/history.csv")
FIELDNAMES=["timestamp","market_phase","etf","etf_price","etf_prev","day_chg_pct",
            "inav","premium_pct","premium_label",
            "comex_price_usd","inr_per_gram","grams_per_unit","usd_inr",
            "rsi","macd_hist","bb_pct_b","ema20","ema50","atr",
            "mom_1d","mom_5d","mom_20d","mom_60d","w52_pos",
            "vol_today","avg_vol","vol_ratio",
            "spread_proxy_pct","tracking_error_pct","daily_expense_drag_pct",
            "signal","score","confidence","stop","target_1r","target_2r",
            "bt_trades","bt_wr","bt_avg","gold_silver_ratio"]

def save_csv(rows,timestamp,market_phase,gsr):
    CSV_PATH.parent.mkdir(parents=True,exist_ok=True)
    wh=not CSV_PATH.exists()
    with CSV_PATH.open("a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDNAMES,extrasaction="ignore")
        if wh: w.writeheader()
        for r in rows:
            bt=r["bt"]; vr=r["vt"]/r["av"] if r["av"] else 0
            w.writerow({"timestamp":timestamp,"market_phase":market_phase,"etf":r["key"],
                "etf_price":r["etf_live"],"etf_prev":r["etf_prev"],"day_chg_pct":r["day_chg"],
                "inav":r["inav"],"premium_pct":r["prem_pct"],"premium_label":r["prem_label"],
                "comex_price_usd":r["comex_live"],"inr_per_gram":r["inr_per_gram"],
                "grams_per_unit":r["grams_per_unit"],"usd_inr":r["usd_inr"],
                "rsi":r["rsi"],"macd_hist":r["macd_h"],"bb_pct_b":r["pct_b"],
                "ema20":r["ema20"],"ema50":r["ema50"],"atr":r["atr"],
                "mom_1d":r["m1"],"mom_5d":r["m5"],"mom_20d":r["m20"],"mom_60d":r["m60"],
                "w52_pos":r["w52"],"vol_today":int(r["vt"]),"avg_vol":int(r["av"]),"vol_ratio":round(vr,2),
                "spread_proxy_pct":r["sp"],"tracking_error_pct":r["te"],
                "daily_expense_drag_pct":r["daily_drag"],
                "signal":r["sig"]["action"],"score":r["sig"]["score"],"confidence":r["sig"]["confidence"],
                "stop":r["risk"]["stop"],"target_1r":r["risk"]["t1"],"target_2r":r["risk"]["t2"],
                "bt_trades":bt.get("total",""),"bt_wr":bt.get("wr",""),"bt_avg":bt.get("avg",""),
                "gold_silver_ratio":round(gsr,2)})

# ── TELEGRAM ────────────────────────────────────────────────────
def send_telegram(msg):
    token=os.getenv("TELEGRAM_BOT_TOKEN"); chat_id=os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id: print("  ❌ Telegram credentials missing"); return
    for attempt in range(1,MAX_RETRIES+1):
        try:
            r=requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id":chat_id,"text":msg,"parse_mode":"Markdown","disable_web_page_preview":True},
                timeout=15)
            if r.status_code==200: print(f"  📨 Telegram sent ({len(msg)} chars)"); return
            print(f"  ⚠️  Attempt {attempt}: {r.status_code}")
        except Exception as e: print(f"  ⚠️  Attempt {attempt}: {e}")
        if attempt<MAX_RETRIES: time_module.sleep(RETRY_DELAY)
    print("  ❌ Telegram failed")

# ── FORMAT ──────────────────────────────────────────────────────
def fmt_overview(s,g,gsr_info,now,market_phase):
    sep="─"*38; sv,gv=s["sig"],g["sig"]
    table=(
        "```\n"
        f"{'Metric':<24} {'TATSILV':>8}  {'TATAGOLD':>8}\n{sep}\n"
        f"{'ETF Price (₹)':<24} {s['etf_live']:>8.4f}  {g['etf_live']:>8.4f}\n"
        f"{'iNAV (₹)':<24} {s['inav']:>8.4f}  {g['inav']:>8.4f}\n"
        f"{'Premium / Disc %':<24} {s['prem_pct']:>+8.2f}%  {g['prem_pct']:>+8.2f}%\n{sep}\n"
        f"{'COMEX (USD/oz)':<24} {s['comex_live']:>8.3f}  {g['comex_live']:>8.2f}\n"
        f"{'INR per gram':<24} {s['inr_per_gram']:>8.2f}  {g['inr_per_gram']:>8.2f}\n"
        f"{'Grams per unit':<24} {s['grams_per_unit']:>8.5f}  {g['grams_per_unit']:>8.5f}\n"
        f"{'USD/INR':<24} {s['usd_inr']:>8.4f}  {g['usd_inr']:>8.4f}\n{sep}\n"
        f"{'RSI (15m)':<24} {s['rsi']:>8.1f}  {g['rsi']:>8.1f}\n"
        f"{'MACD Hist':<24} {s['macd_h']:>+8.4f}  {g['macd_h']:>+8.4f}\n"
        f"{'BB %B':<24} {s['pct_b']:>8.3f}  {g['pct_b']:>8.3f}\n"
        f"{'1d/5d/20d (Silver)':<24} {s['m1']:>+5.1f}% {s['m5']:>+5.1f}% {s['m20']:>+5.1f}%\n"
        f"{'1d/5d/20d (Gold)':<24} {g['m1']:>+5.1f}% {g['m5']:>+5.1f}% {g['m20']:>+5.1f}%\n"
        f"{'52w Position':<24} {s['w52']:>7.1f}%   {g['w52']:>7.1f}%\n{sep}\n"
        f"{'Tracking Error %':<24} {s['te']:>8.3f}  {g['te']:>8.3f}\n"
        f"{'Expense Ratio':<24} {'0.44%':>8}  {'0.38%':>8}\n"
        f"{'Spread Proxy %':<24} {s['sp']:>8.3f}  {g['sp']:>8.3f}\n{sep}\n"
        f"{'Signal':<24} {sv['emoji']:>8}  {gv['emoji']:>8}\n"
        f"{'Score (/10)':<24} {sv['score']:>+8}  {gv['score']:>+8}\n"
        f"{'Confidence':<24} {sv['confidence']:>7.0f}%  {gv['confidence']:>7.0f}%\n```"
    )
    prem=(f"{s['prem_emoji']} *TATSILV:* {s['prem_label']}\n  → _{s['prem_advice']}_\n"
          f"{g['prem_emoji']} *TATAGOLD:* {g['prem_label']}\n  → _{g['prem_advice']}_")
    gsr_b=(f"⚖️ *Gold/Silver Ratio: {gsr_info['gsr']:.2f}*  {gsr_info['emoji']}\n"
           f"  {gsr_info['reason']}\n"
           f"  Deviation from avg (70): {gsr_info['deviation_pct']:+.1f}%")
    if sv["score"]>gv["score"]:   best=f"✅ *Best pick: TATSILV* (score {sv['score']:+d})"
    elif gv["score"]>sv["score"]: best=f"✅ *Best pick: TATAGOLD* (score {gv['score']:+d})"
    else:                         best="➡️ *Equal scores — premium/discount is tiebreaker*"
    return (f"💎 *Gold & Silver ETF Tracker*\n🕒 *{market_phase}*  |  {now.strftime('%d-%b-%Y %H:%M IST')}\n\n"
            f"📊 *Side-by-Side*\n{table}\n🏷 *Premium / Discount*\n{prem}\n\n{gsr_b}\n\n{best}")

def fmt_detail(r,now):
    sig=r["sig"]; risk=r["risk"]; bt=r["bt"]; meta=r["meta"]; sep="─"*30
    vr=r["vt"]/r["av"] if r["av"] else 0
    icon="🥈" if r["key"]=="TATSILV" else "🥇"
    bt_block="No premium-reversion trades in period"
    if "error" not in bt and bt.get("total",0)>0:
        recent="".join(f"\n    {'🟢' if t['ret']>0 else '🔴'} {t['buy']}→{t['sell']}: {t['ret']:+.2f}%"
                       for t in bt.get("trades",[]))
        bt_block=f"{bt['total']} trades | WR {bt['wr']}% | Avg {bt['avg']:+.2f}% | Best {bt['best']:+.2f}%{recent}"
    reasons="\n".join(f"  • {x}" for x in sig["reasons"])
    return (
        f"{icon} *{meta['name']}* ({r['key']})  {sig['emoji']}\n\n"
        "```\n"
        f"{'ETF Price':<22} ₹{r['etf_live']:>9.4f} ({r['day_chg']:>+.2f}%)\n"
        f"{'iNAV (COMEX-based)':<22} ₹{r['inav']:>9.4f}\n"
        f"{'Premium / Disc':<22}  {r['prem_pct']:>+9.2f}%  ← KEY\n{sep}\n"
        f"{'COMEX (USD/oz)':<22} ${r['comex_live']:>9.3f}\n"
        f"{'INR per gram':<22} ₹{r['inr_per_gram']:>9.2f}\n"
        f"{'Grams per unit':<22}  {r['grams_per_unit']:>9.6f}\n"
        f"{'USD/INR':<22} ₹{r['usd_inr']:>9.4f}\n{sep}\n"
        f"{'RSI (15m)':<22} {r['rsi']:>10.1f}\n"
        f"{'MACD Hist':<22} {r['macd_h']:>+10.4f}\n"
        f"{'BB %B':<22} {r['pct_b']:>10.3f}\n"
        f"{'EMA 20 / 50':<22} ₹{r['ema20']:.4f} / ₹{r['ema50']:.4f}\n"
        f"{'ATR':<22} ₹{r['atr']:>9.4f}\n{sep}\n"
        f"{'Mom 1d/5d/20d/60d':<22} {r['m1']:>+5.1f}% {r['m5']:>+5.1f}% {r['m20']:>+5.1f}% {r['m60']:>+5.1f}%\n"
        f"{'52w Position':<22} {r['w52']:>9.1f}%\n{sep}\n"
        f"{'Volume':<22} {vr:>9.1f}× avg\n"
        f"{'Tracking Error':<22} {r['te']:>9.3f}%\n"
        f"{'Expense Ratio':<22} {meta['expense_ratio']*100:>9.2f}%\n"
        f"{'Daily Drag':<22} {r['daily_drag']*100:>9.5f}%\n"
        f"{'Spread Proxy':<22} {r['sp']:>9.3f}%\n"
        f"{'AUM':<22} {'₹'+str(meta['aum_cr'])+'Cr':>10}\n{sep}\n"
        f"{'Stop Loss':<22} ₹{risk['stop']:>9.4f}\n"
        f"{'Target 1:1':<22} ₹{risk['t1']:>9.4f}\n"
        f"{'Target 1:2':<22} ₹{risk['t2']:>9.4f}\n"
        f"{'Risk %':<22} {risk['rp']:>9.2f}%\n```\n"
        f"*Signal:* {sig['action']} (score {sig['score']:+d}/10, {sig['confidence']:.0f}%)\n"
        f"{reasons}\n*Sizing:* {risk['size']}\n\n"
        f"📈 *30d Backtest (premium reversion)*\n{bt_block}"
    )

# ── MAIN ────────────────────────────────────────────────────────
def main():
    print("⏳ Tata Gold & Silver ETF Tracker v4\n")
    ist=pytz.timezone("Asia/Kolkata"); now=datetime.now(ist)
    today_str=now.strftime("%Y-%m-%d"); t=now.time()
    market_phase=("PRE-MARKET" if t<time(9,15) else "POST-MARKET" if t>time(15,30) else "LIVE")

    usd_inr,_=get_live_prev("INR=X"); vix,_=get_live_prev("^INDIAVIX")
    usd_inr=usd_inr if usd_inr>0 else 84.0; vix=vix if vix>0 else 15.0
    print(f"  USD/INR=₹{usd_inr:.4f}  VIX={vix:.2f}")

    silver=analyse_etf("TATSILV",  ETF_META["TATSILV"],  usd_inr,vix)
    gold  =analyse_etf("TATAGOLD", ETF_META["TATAGOLD"], usd_inr,vix)

    gsr=gold["comex_live"]/silver["comex_live"] if silver["comex_live"]>0 else GSR_HIST_AVG
    gsr_info=gsr_signal(gsr)
    print(f"\n  Gold/Silver Ratio: {gsr:.2f}  → {gsr_info['action']}")

    bt_cache_path=Path("data/bt_cache.json"); bt_cache={}
    if bt_cache_path.exists():
        try:
            data=json.loads(bt_cache_path.read_text())
            if data.get("date")==today_str: bt_cache=data.get("results",{}); print("  BT: using cache")
        except: pass

    for key,meta,r in [("TATSILV",ETF_META["TATSILV"],silver),("TATAGOLD",ETF_META["TATAGOLD"],gold)]:
        if key not in bt_cache: bt_cache[key]=run_backtest(meta["ticker"],meta["comex_ticker"])
        r["bt"]=bt_cache[key]

    bt_cache_path.parent.mkdir(parents=True,exist_ok=True)
    try: bt_cache_path.write_text(json.dumps({"date":today_str,"results":bt_cache}))
    except: pass

    save_csv([silver,gold],now.isoformat(),market_phase,gsr)
    print(f"  📁 CSV → {CSV_PATH}")

    send_telegram(fmt_overview(silver,gold,gsr_info,now,market_phase))
    time_module.sleep(1)
    send_telegram(fmt_detail(silver,now))
    time_module.sleep(1)
    send_telegram(fmt_detail(gold,now))
    print("\n✅ Done!")

if __name__=="__main__":
    main()
