"""
Fetch today's UAMPS hourly meter load and save to data/realtime_load.json.
Run hourly at :15 past via Windows Task Scheduler.
"""
import os, sys, json
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import requests, urllib3

load_dotenv()
urllib3.disable_warnings()

uid = os.environ.get("UAMPS_USER_ID", "").strip()
pwd = os.environ.get("UAMPS_PASSWORD", "").strip()
if not uid or not pwd:
    print("ERROR: UAMPS_USER_ID or UAMPS_PASSWORD not set in .env")
    sys.exit(1)

sess = requests.Session()
sess.headers["User-Agent"] = "Mozilla/5.0"
sess.verify = False

r = sess.post("https://px.uamps.com/cgi-bin/wwiz.asp", data={
    "wwizmstr": "WEB.LOGIN", "WWIZ_FORMNO": "0",
    "user": uid, "pwd": pwd, "Submit": "Submit",
}, timeout=30)
if "logoff" not in r.text.lower():
    print("ERROR: UAMPS login failed")
    sys.exit(1)

r2 = sess.get("https://px.uamps.com/members/lehi/hourlylog.xls", timeout=30)
r2.raise_for_status()

hours = {}
lines = r2.content.decode("utf-8", errors="replace").splitlines()
header_idx = None
for i, line in enumerate(lines):
    cols = [c.strip().upper() for c in line.split("\t")]
    if len(cols) >= 2 and cols[0] == "HOUR" and cols[1] == "METERS":
        header_idx = i
        break

if header_idx is not None:
    for line in lines[header_idx + 1:]:
        cols = [c.strip() for c in line.split("\t")]
        if not cols or cols[0].upper() in ("", "TOTAL", "TAGS", "UAMPS", "HOUR"):
            break
        try:
            hr = int(cols[0])
            if not (1 <= hr <= 24):
                continue
            meters = int(cols[1]) if cols[1] else 0
            if meters > 0:
                hours[str(hr)] = meters
        except (ValueError, IndexError):
            continue

tz = ZoneInfo("America/Denver")
now = datetime.now(tz)
output = {
    "date":    now.strftime("%Y-%m-%d"),
    "updated": now.strftime("%Y-%m-%dT%H:%M:%S"),
    "hours":   hours,
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "realtime_load.json")
with open(out_path, "w") as f:
    json.dump(output, f)

print(f"Wrote {len(hours)} hours to {out_path}")
print(f"Hours with data: {sorted(int(k) for k in hours)}")
