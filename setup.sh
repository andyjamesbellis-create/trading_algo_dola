#!/bin/bash
set -e

# Create all required files
cat > mt5_spread_calibration.json << 'EOF'
{
  "pairs": [
    {"Pair": "GBPUSD", "medSprd": "1.60", "mgn/.01": "33.33", "swapL": "-3.10", "swapS": "-4.20"},
    {"Pair": "EURUSD", "medSprd": "1.40", "mgn/.01": "28.76", "swapL": "-8.90", "swapS": "1.90"},
    {"Pair": "GBPJPY", "medSprd": "3.10", "mgn/.01": "33.33", "swapL": "7.60", "swapS": "-36.40"},
    {"Pair": "USDJPY", "medSprd": "1.50", "mgn/.01": "24.86", "swapL": "6.90", "swapS": "-27.95"},
    {"Pair": "USDCAD", "medSprd": "1.70", "mgn/.01": "24.86", "swapL": "2.25", "swapS": "-8.10"},
    {"Pair": "EURGBP", "medSprd": "1.30", "mgn/.01": "28.76", "swapL": "-8.23", "swapS": "1.34"},
    {"Pair": "NZDUSD", "medSprd": "2.70", "mgn/.01": "14.50", "swapL": "-3.65", "swapS": "0.50"}
  ]
}
EOF

cat > 1_fetch_data.py << 'EOF'
import os
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "backtest_data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "backtest_results")
COST_FILE = os.path.join(SCRIPT_DIR, "mt5_spread_calibration.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

if not os.path.exists(COST_FILE):
    print(f"❌ Missing: {COST_FILE}")
    exit()

with open(COST_FILE, "r") as f:
    raw_costs = json.load(f)

SYMBOL_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCAD": "USDCAD=X",
    "EURGBP": "EURGBP=X",
    "NZDUSD": "NZDUSD=X",
    "GBPJPY": "GBPJPY=X",
    "SPY.N": "SPY",
    "QQQ.O": "QQQ",
    "TLT.O": "TLT",
    "#Germany40": "^GDAXI",
    "#USNDAQ100": "^NDX"
}

COST_MODEL = {}
for entry in raw_costs.get("pairs", []):
    sym = entry["Pair"]
    if sym in SYMBOL_MAP:
        COST_MODEL[sym] = {
            "med_spread_pips": float(entry["medSprd"]),
            "margin_per_001": float(entry["mgn/.01"]),
            "swap_long_per_lot": float(entry["swapL"]),
            "swap_short_per_lot": float(entry["swapS"])
        }

COST_MODEL.update({
    "SPY.N": {"med_spread_pips": 0.0, "margin_per_001": 0.0, "swap_long_per_lot": 0.0, "swap_short_per_lot": 0.0},
    "QQQ.O": {"med_spread_pips": 0.0, "margin_per_001": 0.0, "swap_long_per_lot": 0.0, "swap_short_per_lot": 0.0},
    "TLT.O": {"med_spread_pips": 0.0, "margin_per_001": 0.0, "swap_long_per_lot": 0.0, "swap_short_per_lot": 0.0},
    "#Germany40": {"med_spread_pips": 0.0, "margin_per_001": 10.7, "swap_long_per_lot": 0.0, "swap_short_per_lot": 0.0},
    "#USNDAQ100": {"med_spread_pips": 0.0, "margin_per_001": 11.1, "swap_long_per_lot": 0.0, "swap_short_per_lot": 0.0}
})

end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

for symbol, yf_ticker in SYMBOL_MAP.items():
    print(f"Fetching {symbol}...")
    try:
        df = yf.download(yf_ticker, start=start_date, end=end_date, interval="1h", progress=False)
        if df.empty:
            print(f"No data for {symbol}")
            continue
        df = df[["Open", "High", "Low", "Close"]].copy()
        df.index.name = "time"
        df.reset_index(inplace=True)
        df["med_spread_pips"] = COST_MODEL[symbol]["med_spread_pips"]
        df["margin_per_001"] = COST_MODEL[symbol]["margin_per_001"]
        df["swap_long"] = COST_MODEL[symbol]["swap_long_per_lot"]
        df["swap_short"] = COST_MODEL[symbol]["swap_short_per_lot"]
        df.to_csv(os.path.join(DATA_DIR, f"{symbol}.csv"), index=False)
        print(f"Saved {len(df)} rows")
    except Exception as e:
        print(f"Error: {str(e)}")

print("\nData fetch complete.")
EOF

cat > 2_backtest_strategy.py << 'EOF'
import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "backtest_data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "backtest_results")

INITIAL_CAPITAL = 225.0
RISK_PER_TRADE_PCT = 0.5
MAX_MARGIN_PCT = 60.0
LOT_SIZE = 0.01
ATR_PERIOD = 14
EMA_FAST = 50
EMA_SLOW = 200
SL_MULT = 1.5
TP_MULT = 3.0
MAX_POSITIONS = 5

def calculate_indicators(df):
    df["ema_fast"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["tr"] = np.maximum(df["High"]-df["Low"], np.maximum(abs(df["High"]-df["Close"].shift(1)), abs(df["Low"]-df["Close"].shift(1))))
    df["atr"] = df["tr"].rolling(window=ATR_PERIOD).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta>0,0)
    loss = -delta.where(delta<0,0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100/(1+rs))
    return df.dropna()

def pip_value(symbol, price):
    if symbol in ["EURUSD","GBPUSD","AUDUSD","NZDUSD"]: return 0.10
    elif symbol in ["USDJPY","USDCAD","EURGBP"]: return 0.08
    return 0.10

results = []
all_trades = []

if not os.path.exists(DATA_DIR):
    print("Run 1_fetch_data.py first")
    exit()

for filename in os.listdir(DATA_DIR):
    if not filename.endswith(".csv"): continue
    symbol = filename.replace(".csv","")
    print(f"\nBacktesting {symbol}...")
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    df = calculate_indicators(df)
    if len(df) < 200: continue
    balance = INITIAL_CAPITAL
    open_positions = []
    trades = []
    for i in range(200, len(df)):
        row = df.iloc[i]
        close = row["Close"]
        atr = row["atr"]
        rsi = row["rsi"]
        spread_pips = row["med_spread_pips"]
        margin_req = row["margin_per_001"]
        total_margin = sum(p["margin"] for p in open_positions)
        if total_margin + margin_req > INITIAL_CAPITAL * MAX_MARGIN_PCT / 100: continue
        uptrend = row["ema_fast"] > row["ema_slow"]
        downtrend = row["ema_fast"] < row["ema_slow"]
        if len(open_positions) < MAX_POSITIONS:
            risk_amt = balance * RISK_PER_TRADE_PCT / 100
            sl_pips = atr * SL_MULT / 0.0001
            if sl_pips < 1: continue
            pip_val = pip_value(symbol, close)
            if sl_pips * pip_val > risk_amt: continue
            spread_cost = spread_pips * pip_val
            swap_rate = (row["swap_long"] if uptrend else row["swap_short"]) / 100
            if uptrend and 35 < rsi < 65:
                entry = close + (spread_pips * 0.0001)
                sl = entry - atr * SL_MULT
                tp = entry + atr * TP_MULT
                open_positions.append({"type":"LONG","entry":entry,"sl":sl,"tp":tp,"margin":margin_req,"time":row["time"],"pip_val":pip_val,"spread_cost":spread_cost,"swap_rate":swap_rate})
            elif downtrend and 35 < rsi < 65:
                entry = close - (spread_pips * 0.0001)
                sl = entry + atr * SL_MULT
                tp = entry - atr * TP_MULT
                open_positions.append({"type":"SHORT","entry":entry,"sl":sl,"tp":tp,"margin":margin_req,"time":row["time"],"pip_val":pip_val,"spread_cost":spread_cost,"swap_rate":swap_rate})
        closed = []
        for pos in open_positions:
            days = (pd.to_datetime(row["time"]) - pd.to_datetime(pos["time"])).days
            swap = pos["swap_rate"] * LOT_SIZE * days
            if pos["type"] == "LONG":
                if close <= pos["sl"]:
                    pnl = (pos["sl"] - pos["entry"])*100000*LOT_SIZE - pos["spread_cost"] + swap
                    trades.append({"time":row["time"],"symbol":symbol,"result":"LOSS","pnl":round(pnl,2)})
                    balance += pnl
                    closed.append(pos)
                elif close >= pos["tp"]:
                    pnl = (pos["tp"] - pos["entry"])*100000*LOT_SIZE - pos["spread_cost"] + swap
                    trades.append({"time":row["time"],"symbol":symbol,"result":"WIN","pnl":round(pnl,2)})
                    balance += pnl
                    closed.append(pos)
            else:
                if close >= pos["sl"]:
                    pnl = (pos["entry"] - pos["sl"])*100000*LOT_SIZE - pos["spread_cost"] + swap
                    trades.append({"time":row["time"],"symbol":symbol,"result":"LOSS","pnl":round(pnl,2)})
                    balance += pnl
                    closed.append(pos)
                elif close <= pos["tp"]:
                    pnl = (pos["entry"] - pos["tp"])*100000*LOT_SIZE - pos["spread_cost"] + swap
                    trades.append({"time":row["time"],"symbol":symbol,"result":"WIN","pnl":round(pnl,2)})
                    balance += pnl
                    closed.append(pos)
        open_positions = [p for p in open_positions if p not in closed]
    wins = sum(1 for t in trades if t["result"] == "WIN")
    total = len(trades)
    results.append({"symbol":symbol,"final_balance":round(balance,2),"net_profit":round(balance-INITIAL_CAPITAL,2),"total_trades":total,"win_rate_pct":round(wins/total*100,1) if total else 0})
    all_trades.extend(trades)

pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR,"summary.csv"),index=False)
pd.DataFrame(all_trades).to_csv(os.path.join(RESULTS_DIR,"all_trades.csv"),index=False)
print("\nBacktest done.")
EOF

cat > 3_generate_report.py << 'EOF'
import os
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "backtest_results")
summary_path = os.path.join(RESULTS_DIR, "summary.csv")
trades_path = os.path.join(RESULTS_DIR, "all_trades.csv")

if not os.path.exists(summary_path):
    print("Run backtest first")
    exit()

summary = pd.read_csv(summary_path)
all_trades = pd.read_csv(trades_path)

print("\n" + "="*70)
print("📊 DIVERSIFIED STRATEGY BACKTEST REPORT")
print(f"Initial Capital: £225 | Risk: 0.5% | Leverage: 1:30")
print("="*70)
print(summary.to_string(index=False))
print("-"*70)

total_net = summary["net_profit"].sum()
total_trades = summary["total_trades"].sum()
win_rate = round((sum(summary["win_rate_pct"] * summary["total_trades"]) / total_trades),1) if total_trades else 0

print(f"TOTAL NET PROFIT: £{round(total_net,2)}")
print(f"FINAL BALANCE: £{round(225 + total_net,2)}")
print(f"OVERALL WIN RATE: {win_rate}%")
print(f"TOTAL TRADES: {total_trades}")
print("="*70)

if len(all_trades) > 0:
    all_trades["time"] = pd.to_datetime(all_trades["time"])
    all_trades = all_trades.sort_values("time")
    all_trades["equity"] = 225 + all_trades["pnl"].cumsum()
    plt.figure(figsize=(10,5))
    plt.plot(all_trades["time"], all_trades["equity"], color="#2c3e50", linewidth=1.2)
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Balance (£)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,"equity_curve.png"))
    plt.show()
EOF

cat > requirements.txt << 'EOF'
pandas>=2.0
numpy>=1.24
yfinance>=0.2
matplotlib>=3.7
EOF

cat > run_all.sh << 'EOF'
#!/bin/bash
echo "=== Installing dependencies ==="
pip install -r requirements.txt
echo -e "\n=== Step 1/3: Fetching data ==="
python 1_fetch_data.py
echo -e "\n=== Step 2/3: Running backtest ==="
python 2_backtest_strategy.py
echo -e "\n=== Step 3/3: Generating report ==="
python 3_generate_report.py
echo -e "\n✅ All done! Check backtest_results/ for outputs."
EOF

chmod +x run_all.sh setup.sh
echo "✅ All files created successfully!"
