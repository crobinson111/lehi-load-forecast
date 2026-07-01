"""
Fetch UAMPS Real-time Schedulers Log for a rolling date range and write to CSV.
Usage: python fetch_uamps.py <output_csv> [days_back] [days_forward]
Reads UAMPS_USER_ID and UAMPS_PASSWORD from environment (or .env file).
"""
import os, re, sys, time
from datetime import date, timedelta

import pandas as pd
import requests
import urllib3
from dotenv import load_dotenv

load_dotenv()
urllib3.disable_warnings()

output       = sys.argv[1]
days_back    = int(sys.argv[2]) if len(sys.argv) > 2 else 30
days_forward = int(sys.argv[3]) if len(sys.argv) > 3 else 16

user_id  = os.environ.get("UAMPS_USER_ID", "")
password = os.environ.get("UAMPS_PASSWORD", "")
if not user_id or not password:
    print("ERROR: UAMPS_USER_ID or UAMPS_PASSWORD not set in .env")
    sys.exit(1)

BASE = "https://px.uamps.com/cgi-bin/wwiz.asp"

sess = requests.Session()
sess.headers["User-Agent"] = "Mozilla/5.0"
sess.verify = False  # city SSL proxy workaround

# Login
r = sess.post(BASE, data={
    "wwizmstr": "WEB.LOGIN", "WWIZ_FORMNO": "0",
    "user": user_id, "pwd": password, "Submit": "Submit",
}, timeout=30)
if "logoff" not in r.text.lower():
    print("ERROR: UAMPS login failed — check credentials")
    sys.exit(1)
print("UAMPS login OK")

today = date.today()
start = today - timedelta(days=days_back)
end   = today + timedelta(days=days_forward)
print(f"Fetching {start} to {end} ({(end - start).days + 1} days)...")

rows = []
d = start
while d <= end:
    for attempt in range(3):
        try:
            r2 = sess.post(BASE, data={
                "wwizmstr": "WEB.SCHED.LOG.FOR.MBRS", "WWIZ_FORMNO": "0",
                "Destination": "S",
                "Year":   str(d.year)[-2:],
                "Month":  str(d.month),
                "Day":    str(d.day),
                "Submit": "Run Report Now",
            }, timeout=30)
            break
        except Exception as exc:
            if attempt < 2:
                print(f"  {d}: error ({exc}), retrying...")
                time.sleep(5)
            else:
                print(f"  {d}: failed after 3 attempts, skipping")
                r2 = None

    if r2 is None:
        d += timedelta(days=1)
        continue

    m = re.search(r'<pre>(.*?)</pre>', r2.text, re.S | re.I)
    if not m:
        print(f"  {d}: no data")
        d += timedelta(days=1)
        time.sleep(0.3)
        continue

    pre_text   = re.sub(r'<[^>]+>', '\n', m.group(1))
    kw_section = re.split(r'-{20,}', pre_text)[0]
    day_rows   = 0
    for line in kw_section.splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        hr = int(parts[0])
        if hr < 1 or hr > 24 or len(parts) < 16:
            continue
        rows.append({
            "date":      d.isoformat(),
            "hr":        hr,
            "crsp":      int(parts[5]),
            "provo_riv": int(parts[8]),
            "veyo":      int(parts[15]),
        })
        day_rows += 1

    print(f"  {d}: {day_rows} hours")
    d += timedelta(days=1)
    time.sleep(0.4)  # be polite to the server

df = pd.DataFrame(rows, columns=["date", "hr", "crsp", "provo_riv", "veyo"])
df.to_csv(output, index=False)
print(f"Wrote {len(df)} rows to {output}  ({df['date'].min()} to {df['date'].max()})")
