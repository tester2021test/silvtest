# 💎 Tata Gold & Silver ETF Tracker

Automated tracker for **TATSILV.NS** (Tata Silver ETF) and **TATAGOLD.NS** (Tata Gold ETF).  
Runs every **5 minutes** via GitHub Actions during NSE market hours and sends rich Telegram alerts.

---

## 📁 Repository Structure

```
├── tracker.py                        # Main tracker script
├── requirements.txt                  # Python dependencies
├── .github/
│   └── workflows/
│       └── tracker.yml               # GitHub Actions workflow (runs every 5 min)
├── data/
│   ├── history.csv                   # Auto-updated price & signal history
│   └── bt_cache.json                 # Daily backtest cache (auto-generated)
└── README.md
```

---

## 🚀 Setup (5 minutes)

### Step 1 — Fork / create repo
Create a new GitHub repository and push all these files.

### Step 2 — Add Telegram Bot
1. Message **@BotFather** on Telegram → `/newbot` → copy the token
2. Get your Chat ID: message **@userinfobot** or send a message to your bot then visit:  
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`

### Step 3 — Add GitHub Secrets
Go to your repo → **Settings → Secrets and variables → Actions → New repository secret**

| Secret Name | Value |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Your bot token from BotFather |
| `TELEGRAM_CHAT_ID` | Your Telegram chat/user ID |

### Step 4 — Enable GitHub Actions
Go to **Actions** tab → enable workflows if prompted.

### Step 5 — Test manually
Go to **Actions → Gold & Silver ETF Tracker → Run workflow** to trigger immediately.

---

## 📊 What You Receive (3 Telegram Messages per run)

### Message 1 — Side-by-Side Overview
```
SILVER  vs  GOLD comparison table:
• ETF Price vs iNAV (calculated fair value)
• Premium / Discount % ← MOST IMPORTANT
• US Futures price (USD), MCX price (₹), USD/INR
• RSI, MACD Histogram, Bollinger %B
• 1d / 5d / 20d momentum
• 52-week range position
• Tracking Error, Expense Ratio, Daily drag
• Signal score & emoji
Gold/Silver Ratio with trade recommendation
Best pick verdict
```

### Message 2 — TATSILV Deep Dive
Full technicals + risk levels + 30-day backtest

### Message 3 — TATAGOLD Deep Dive
Full technicals + risk levels + 30-day backtest

---

## 🧠 How to Interpret Signals

### ⚡ Premium / Discount (MOST CRITICAL for ETFs)
| Premium | Label | Action |
|---|---|---|
| > +5% | ⛔ DANGER PREMIUM | **DO NOT BUY** — paying way above fair value |
| +2% to +5% | ⚠️ CAUTION | Wait for premium to compress |
| ±0.5% | ➡️ FAIR VALUE | Acceptable if technically bullish |
| < -1% | ✅ BUY ZONE | ETF cheaper than fair value — good entry |
| < -3% | 🎯 STRONG BUY ZONE | Significant discount — strong entry |

> **Why this matters:** Indian silver ETFs have traded at [15–20% premiums](https://www.businesstoday.in/mutual-funds/story/your-silver-etf-is-not-tracking-silver-497595-2025-10-09) during rallies.  
> Buying at a 15% premium means you need a 15% rally just to break even if the premium normalises.  
> **Always check premium before buying.**

### ⚖️ Gold/Silver Ratio (GSR)
| GSR | Signal |
|---|---|
| > 80 | Silver historically cheap → **PREFER SILVER** |
| 60–80 | Neutral zone |
| < 60 | Gold historically cheap → **PREFER GOLD** |

Historical average: ~70. The ratio tends to mean-revert.

### 📐 Signal Score (/10)
| Score | Action |
|---|---|
| +6 to +10 | STRONG BUY |
| +3 to +5 | BUY |
| -2 to +2 | NEUTRAL / HOLD |
| -3 to -5 | AVOID |
| -6 to -10 | STRONG AVOID |

### 📈 Tracking Error
- Lower is better (means ETF closely tracks underlying silver/gold)
- > 2% annually = notable deviation — check if MCX prices are diverging

### 💸 Expense Ratio Impact
- TATSILV: **0.44%/year** = ~0.00175%/day drag on returns
- TATAGOLD: **0.38%/year** = ~0.00151%/day drag on returns

---

## 📋 CSV Columns (data/history.csv)

| Column | Description |
|---|---|
| `premium_pct` | ETF price vs calculated iNAV (%) |
| `inav` | Calculated fair value based on US futures × USD/INR |
| `mcx_premium_pct` | MCX price vs US futures (includes import duty + GST) |
| `tracking_error_pct` | Rolling 30d annualised tracking error vs underlying |
| `daily_expense_drag_pct` | Daily cost drag from expense ratio |
| `spread_proxy_pct` | (High-Low)/Close — proxy for bid-ask spread |
| `gold_silver_ratio` | Gold price (oz) ÷ Silver price (oz) |
| `w52_pos` | Where current price sits in 52-week range (0–100%) |
| `mom_1d / 5d / 20d / 60d` | Price momentum over different periods |

---

## ⏱️ Schedule
- Runs **every 5 minutes**, Monday–Friday
- Active from **09:10 IST to 15:35 IST** (cron: 03:30–10:30 UTC)
- Each run appends one row per ETF to `data/history.csv`
- CSV is automatically committed back to the repo

---

## ⚠️ Disclaimer
This tool is for informational purposes only and does not constitute financial advice.
Always check iNAV on NSE website before buying any ETF. Past backtest results do not guarantee future returns.
