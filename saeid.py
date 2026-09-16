import os
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
import time
import requests
import ccxt

SHOW_N = 20

# ---------- پارامترهای تشخیص چکش معکوس سبز ----------
BODY_MAX_RATIO     = 0.35   # بدنه حداکثر ۳۵٪ دامنه
UPPER_MIN_RATIO    = 0.50   # سایه بالایی حداقل ۵۰٪ دامنه
UPPER_BODY_MULT    = 2.0    # سایه بالایی حداقل ۲ برابر بدنه
LOWER_MAX_RATIO    = 0.15   # سایه پایینی حداکثر ۱۵٪ دامنه

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

    seen, unique_bases = set(), []
    for b in base_list:
        if b not in seen:
            seen.add(b)
            unique_bases.append(b)
    print(f"✅ تعداد ارزهای پایه‌ی منحصربه‌فرد فیوچرز LBank: {len(unique_bases)}")
    return unique_bases

def get_4h_data(ticker):
    df = yf.download(ticker, period='60d', interval='1h',
                     progress=False, auto_adjust=False)
    if df.empty:
        return None
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    df.index = pd.to_datetime(df.index)
    df.columns = ['open', 'high', 'low', 'close']

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df_4h = df.resample('4h').agg({
        'open':  'first',
        'high':  'max',
        'low':   'min',
        'close': 'last',
    }).dropna()
    return df_4h

def detect_green_inverted_hammer(o, h, l, c):
    """
    اگر کندل «چکش معکوس سبز» بود، دیکشنری نسبت‌ها برمی‌گرداند، وگرنه None.
    """
    rng = float(h - l)
    if rng <= 0 or not np.isfinite(rng):
        return None

    # 🟢 سبز بودن
    if c <= o:
        return None

    body       = c - o
    upper_wick = h - c
    lower_wick = o - l

    body_r  = body / rng
    upper_r = upper_wick / rng
    lower_r = lower_wick / rng

    if body_r  > BODY_MAX_RATIO:                    return None
    if upper_r < UPPER_MIN_RATIO:                   return None
    if upper_wick < UPPER_BODY_MULT * body:         return None
    if lower_r > LOWER_MAX_RATIO:                   return None
    if lower_wick > body:                           return None

    # امتیاز کیفیت: هرچه سایه بالایی بلندتر و بدنه کوچک‌تر => بهتر
    score = upper_r - body_r
    return {
        'body_ratio':  body_r,
        'upper_ratio': upper_r,
        'lower_ratio': lower_r,
        'score':       score,
    }

def send_telegram_message(text):
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        print("❌ توکن یا chat_id تنظیم نشده است.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML'}
    try:
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            print(f"⚠️ خطا در ارسال پیام: {r.text}")
        else:
            print("✅ پیام با موفقیت به تلگرام ارسال شد.")
    except Exception as e:
        print(f"❌ خطا در ارسال به تلگرام: {e}")

# ---------- اجرای اصلی ----------
print("🔎 جستجوی الگوی «چکش معکوس سبز» در کندل ۴ ساعته (قبلی)...")

print("\n📊 دریافت نمادهای فیوچرز LBank ...")
symbols = get_lbank_futures_symbols()
if not symbols:
    print("❌ هیچ نمادی برای اسکن وجود ندارد!")
    exit()

results = []
for sym in tqdm(symbols, desc="اسکن کندل ۴ ساعته (قبلی)"):
    try:
        df_4h = get_4h_data(f"{sym}-USD")
        if df_4h is None or len(df_4h) < 3:
            continue

        prev = df_4h.iloc[-2]     # کندل بسته‌شده‌ی قبلی
        o, h, l, c = (float(prev['open']),  float(prev['high']),
                      float(prev['low']),   float(prev['close']))

        info = detect_green_inverted_hammer(o, h, l, c)
        if info is None:
            continue

        results.append({
            'symbol':   sym,
            'score':    info['score'],
            'body_r':   info['body_ratio'],
            'upper_r':  info['upper_ratio'],
            'lower_r':  info['lower_ratio'],
            'time_4h':  df_4h.index[-2].strftime('%Y-%m-%d %H:%M'),
            'o': o, 'h': h, 'l': l, 'c': c,
        })
        time.sleep(0.3)
    except Exception:
        continue

if results:
    df_res = pd.DataFrame(results).sort_values('score', ascending=False).head(SHOW_N)

    lines = []
    lines.append(f"🕯 <b>کندل‌های ۴ ساعته‌ی «چکش معکوس سبز» (کندل قبلی)</b>\n")
    lines.append(
        f"معیار: بدنه≤{BODY_MAX_RATIO:.0%} | سایه‌بالا≥{UPPER_MIN_RATIO:.0%} و "
        f"≥{UPPER_BODY_MULT:g}× بدنه | سایه‌پایین≤{LOWER_MAX_RATIO:.0%}\n"
    )
    for _, row in df_res.iterrows():
        lines.append(
            f"🟢 <b>{row['symbol']}</b>  (امتیاز: {row['score']:.3f})\n"
            f"   زمان ۴h: {row['time_4h']}\n"
            f"   O={row['o']:.6g} H={row['h']:.6g} L={row['l']:.6g} C={row['c']:.6g}\n"
            f"   بدنه={row['body_r']:.2f}  سایه‌بالا={row['upper_r']:.2f}  "
            f"سایه‌پایین={row['lower_r']:.2f}"
        )
    lines.append(f"\n📅 تعداد ارزهای اسکن‌شده: {len(symbols)}")
    message = "\n".join(lines)

    send_telegram_message(message)
    print("\n" + message)
else:
    print("\n❌ هیچ چکش معکوس سبزی یافت نشد.")
    send_telegram_message("❌ در اسکن امروز هیچ «چکش معکوس سبزی» یافت نشد.")

print("\n✅ اسکن کامل شد!")
