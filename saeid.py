import os
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
import time
import requests
import ccxt

FAST, SLOW = 12, 26
# تاریخ الگوی مرجع (روزانه) — DD/MM/YYYY → ISO
PATTERN_START, PATTERN_END = '2025-05-08', '2025-07-06'
REFERENCE_SYMBOL = 'XTZ-USD'   # نماد الگوی مرجع
SHOW_N = 10

# ---------- توابع ----------
def get_lbank_futures_symbols():
    exchange = ccxt.lbank({'options': {'defaultType': 'future'}})
    try:
        markets = exchange.load_markets()
    except Exception as e:
        print(f"❌ خطا در اتصال به LBank: {e}")
        return []

    base_list = []
    for symbol, market in markets.items():
        if not market.get('swap'):
            continue
        base = market.get('base')
        if not base or base.isdigit():
            continue
        base_list.append(base.upper())

    seen = set()
    unique_bases = []
    for b in base_list:
        if b not in seen:
            seen.add(b)
            unique_bases.append(b)
    print(f"✅ تعداد ارزهای پایه‌ی منحصربه‌فرد فیوچرز LBank: {len(unique_bases)}")
    return unique_bases

def get_daily_data(ticker):
    """داده‌ی روزانه برای الگوی مرجع"""
    df = yf.download(ticker, start='2024-01-01', interval='1d', progress=False, auto_adjust=False)
    if df.empty:
        return None
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    df.index = pd.to_datetime(df.index)
    df.columns = ['open', 'high', 'low', 'close']
    return df

def get_30m_data(ticker):
    df = yf.download(ticker, period='60d', interval='30m', progress=False, auto_adjust=False)
    if df.empty:
        return None
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    df.index = pd.to_datetime(df.index)
    df.columns = ['open', 'high', 'low', 'close']
    return df

def macd_line(close_series):
    ema_fast = close_series.ewm(span=FAST, adjust=False).mean()
    ema_slow = close_series.ewm(span=SLOW, adjust=False).mean()
    return ema_fast - ema_slow

def dtw_distance(x, y, window=None):
    n = len(x)
    dtw = np.full((n+1, n+1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n+1):
        if window is None:
            j_start, j_end = 1, n
        else:
            j_start = max(1, i - window)
            j_end = min(n, i + window)
        for j in range(j_start, j_end+1):
            cost = (x[i-1] - y[j-1]) ** 2
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    return np.sqrt(dtw[n, n])

def send_telegram_message(text):
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        print("❌ توکن یا chat_id تنظیم نشده است.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': text,
        'parse_mode': 'HTML'
    }
    try:
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            print(f"⚠️ خطا در ارسال پیام: {r.text}")
        else:
            print("✅ پیام با موفقیت به تلگرام ارسال شد.")
    except Exception as e:
        print(f"❌ خطا در ارسال به تلگرام: {e}")

# ---------- اجرای اصلی ----------
print(f"🔍 استخراج الگوی روزانه {REFERENCE_SYMBOL} ...")
ref_d = get_daily_data(REFERENCE_SYMBOL)
if ref_d is None:
    print(f"❌ خطا در دریافت داده‌های {REFERENCE_SYMBOL}")
    exit()

ref_macd = macd_line(ref_d['close']).dropna()
mask = (ref_macd.index >= PATTERN_START) & (ref_macd.index <= PATTERN_END)
pattern = ref_macd[mask].values
L = len(pattern)
print(f"✅ الگوی مرجع (روزانه) با {L} کندل")

if L < 2:
    print("❌ الگوی مرجع خیلی کوتاه است.")
    exit()

pattern_mean = np.mean(pattern)
pattern_std = np.std(pattern) + 1e-9
pat_norm = (pattern - pattern_mean) / pattern_std

WINDOW = max(1, int(0.5 * L))
print(f"🔹 پنجره DTW (Sakoe-Chiba): {WINDOW}")

print("\n📊 دریافت نمادهای فیوچرز LBank ...")
top_symbols = get_lbank_futures_symbols()

if len(top_symbols) == 0:
    print("❌ هیچ نمادی برای اسکن وجود ندارد!")
    exit()

results = []
for sym in tqdm(top_symbols, desc="اسکن ۳۰ دقیقه‌ای"):
    try:
        df_30m = get_30m_data(f"{sym}-USD")
        if df_30m is None or len(df_30m) < L + 30:
            continue

        macd = macd_line(df_30m['close']).dropna()
        if len(macd) < L:
            continue

        current = macd.iloc[-L:].values
        cur_mean = np.mean(current)
        cur_std = np.std(current) + 1e-9
        cur_norm = (current - cur_mean) / cur_std

        dist_dtw = dtw_distance(pat_norm, cur_norm, window=WINDOW)

        last_time = macd.index[-1].strftime('%Y-%m-%d %H:%M')
        results.append({
            'symbol': sym,
            'dist_dtw': dist_dtw,
            'last_30m': last_time
        })
        time.sleep(0.3)
    except Exception:
        continue

if results:
    df_res = pd.DataFrame(results)

    # رتبه‌بندی فقط بر اساس DTW (کمترین = بهترین)
    df_top = df_res.sort_values('dist_dtw').head(SHOW_N)

    # ساخت پیام متنی برای تلگرام
    message_lines = []
    message_lines.append(f"🏆 <b>برترین ارزهای مشابه الگوی روزانه {REFERENCE_SYMBOL} (فقط DTW)</b>\n")
    message_lines.append(f"(MACD ۳۰ دقیقه‌ای در برابر الگوی روزانه {PATTERN_START} تا {PATTERN_END} با محدودیت Sakoe-Chiba)\n")
    for idx, row in df_top.iterrows():
        line = (
            f"🔸 <b>{row['symbol']}</b>\n"
            f"   DTW: {row['dist_dtw']:.4f} | بروزرسانی: {row['last_30m']}\n"
        )
        message_lines.append(line)
    message_lines.append(f"\n📅 تعداد کل ارزهای اسکن‌شده: {len(top_symbols)}")
    message = "\n".join(message_lines)

    # ارسال به تلگرام
    send_telegram_message(message)

    # چاپ در لاگ هم برای بررسی
    print("\n" + message)
else:
    print("\n❌ نتیجه‌ای یافت نشد.")
    send_telegram_message("❌ در اسکن امروز هیچ نتیجه‌ای یافت نشد.")

print("\n✅ اسکن کامل شد!")
