import sys, os, shutil, tempfile, json, time
import urllib.request
import urllib.parse
import pandas as pd
from datetime import datetime

excel_path         = sys.argv[1]
output_load        = sys.argv[2]
output_solar       = sys.argv[3]
output_load_wx     = sys.argv[4]
output_solar_wx    = sys.argv[5]
output_steele_a    = sys.argv[6]
output_steele_a_wx = sys.argv[7]
output_supply      = sys.argv[8]

# ── Excel export ─────────────────────────────────────────────────────────────
try:
    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    tmp.close()
    shutil.copy2(excel_path, tmp.name)
except Exception as exc:
    print(f"ERROR: Could not copy spreadsheet: {exc}", file=sys.stderr)
    sys.exit(1)

try:
    df = pd.read_excel(tmp.name, sheet_name="Sch Log Data (2)", engine="openpyxl")
finally:
    os.unlink(tmp.name)

df.columns = df.columns.str.strip()

required = {"Date", "Hr", "Total Meters plus Gens"}
missing = required - set(df.columns)
if missing:
    print(f"ERROR: Missing columns: {missing}", file=sys.stderr)
    sys.exit(1)

df_load = df.dropna(subset=list(required)).copy()
df_load["date"] = (
    df_load["Date"].astype(int).astype(str).str.zfill(6)
    .pipe(lambda s: pd.to_datetime(s, format="%y%m%d"))
    .dt.strftime("%Y-%m-%d")
)
df_load["hr"]   = df_load["Hr"].astype(int)
df_load["load"] = pd.to_numeric(df_load["Total Meters plus Gens"], errors="coerce")
df_load = df_load[(df_load["hr"] >= 1) & (df_load["hr"] <= 24) & (df_load["load"] > 0)].dropna(subset=["load"])
df_load[["date", "hr", "load"]].to_csv(output_load, index=False)
print(f"Wrote {len(df_load)} load rows to {output_load}")

if "RED MESA" not in df.columns:
    print("WARNING: RED MESA column not found — skipping solar export", file=sys.stderr)
    sys.exit(0)

df_solar = df.dropna(subset=["Date", "Hr"]).copy()
df_solar["date"] = (
    df_solar["Date"].astype(int).astype(str).str.zfill(6)
    .pipe(lambda s: pd.to_datetime(s, format="%y%m%d"))
    .dt.strftime("%Y-%m-%d")
)
df_solar["hr"]  = df_solar["Hr"].astype(int)
df_solar["kwh"] = pd.to_numeric(df_solar["RED MESA"], errors="coerce").fillna(0)
df_solar = df_solar[(df_solar["hr"] >= 1) & (df_solar["hr"] <= 24)]
df_solar[["date", "hr", "kwh"]].to_csv(output_solar, index=False)
print(f"Wrote {len(df_solar)} solar rows to {output_solar}")

# ── Fetch training weather (runs on local machine — your IP, not Render's) ───

def fetch_archive(lat, lon, start, end, variables):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ",".join(variables),
        "temperature_unit": "fahrenheit",
        "timezone": "America/Denver",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    print(f"  GET {url[:80]}...")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.loads(r.read())
        except Exception as exc:
            if attempt < 3:
                wait = 15 * (attempt + 1)
                print(f"  Error: {exc}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

# Load weather — last 2 years for Lehi, UT
load_end   = df_load["date"].max()
load_start = (pd.to_datetime(load_end) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
print(f"\nFetching load weather {load_start} to {load_end} for Lehi, UT...")
wx = fetch_archive(40.3916, -111.8508, load_start, load_end,
                   ["temperature_2m", "apparent_temperature"])
rows = []
for t, temp, app in zip(wx["hourly"]["time"],
                        wx["hourly"]["temperature_2m"],
                        wx["hourly"]["apparent_temperature"]):
    if temp is None:
        continue
    dt = datetime.fromisoformat(t)
    rows.append({
        "date": dt.strftime("%Y-%m-%d"),
        "hr": dt.hour + 1,
        "temp_f": round(float(temp), 2),
        "apparent_f": round(float(app) if app is not None else float(temp), 2),
    })
pd.DataFrame(rows).to_csv(output_load_wx, index=False)
print(f"Wrote {len(rows)} load weather rows to {output_load_wx}")

time.sleep(5)

# Solar weather — last 3 years for Bluff, UT
solar_end   = df_solar["date"].max()
solar_start = (pd.to_datetime(solar_end) - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
print(f"\nFetching solar weather {solar_start} to {solar_end} for Bluff, UT...")
wx_s = fetch_archive(37.2879, -109.5512, solar_start, solar_end,
                     ["shortwave_radiation", "temperature_2m"])
rows_s = []
for t, ghi, temp in zip(wx_s["hourly"]["time"],
                        wx_s["hourly"]["shortwave_radiation"],
                        wx_s["hourly"]["temperature_2m"]):
    if ghi is None:
        continue
    dt = datetime.fromisoformat(t)
    rows_s.append({
        "date": dt.strftime("%Y-%m-%d"),
        "hr": dt.hour + 1,
        "ghi": round(float(ghi), 2),
        "temp_f": round(float(temp) if temp is not None else 70.0, 2),
    })
pd.DataFrame(rows_s).to_csv(output_solar_wx, index=False)
print(f"Wrote {len(rows_s)} solar weather rows to {output_solar_wx}")

time.sleep(5)

# Steele A history — "Steel A" column
df_steele = df.dropna(subset=["Date", "Hr"]).copy()
df_steele["date"] = (
    df_steele["Date"].astype(int).astype(str).str.zfill(6)
    .pipe(lambda s: pd.to_datetime(s, format="%y%m%d"))
    .dt.strftime("%Y-%m-%d")
)
df_steele["hr"]  = df_steele["Hr"].astype(int)
df_steele["kwh"] = pd.to_numeric(df_steele["Steel A"], errors="coerce").fillna(0)
df_steele = df_steele[(df_steele["hr"] >= 1) & (df_steele["hr"] <= 24)]
df_steele[["date", "hr", "kwh"]].to_csv(output_steele_a, index=False)
print(f"Wrote {len(df_steele)} Steele A rows to {output_steele_a}")

# Steele A weather — last 3 years for Plymouth, UT
steele_end   = df_steele["date"].max()
steele_start = (pd.to_datetime(steele_end) - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
print(f"\nFetching Steele A weather {steele_start} to {steele_end} for Plymouth, UT...")
wx_sa = fetch_archive(41.878, -112.148, steele_start, steele_end,
                      ["shortwave_radiation", "temperature_2m"])
rows_sa = []
for t, ghi, temp in zip(wx_sa["hourly"]["time"],
                        wx_sa["hourly"]["shortwave_radiation"],
                        wx_sa["hourly"]["temperature_2m"]):
    if ghi is None:
        continue
    dt = datetime.fromisoformat(t)
    rows_sa.append({
        "date": dt.strftime("%Y-%m-%d"),
        "hr": dt.hour + 1,
        "ghi": round(float(ghi), 2),
        "temp_f": round(float(temp) if temp is not None else 70.0, 2),
    })
pd.DataFrame(rows_sa).to_csv(output_steele_a_wx, index=False)
print(f"Wrote {len(rows_sa)} Steele A weather rows to {output_steele_a_wx}")

# Supply portfolio history — Nebo, H Butte, PX, OS columns
df_supply = df.dropna(subset=["Date", "Hr"]).copy()
df_supply["date"] = (
    df_supply["Date"].astype(int).astype(str).str.zfill(6)
    .pipe(lambda s: pd.to_datetime(s, format="%y%m%d"))
    .dt.strftime("%Y-%m-%d")
)
df_supply["hr"]      = df_supply["Hr"].astype(int)
df_supply["nebo"]    = pd.to_numeric(df_supply["NEBO"]    if "NEBO"    in df_supply.columns else 0, errors="coerce").fillna(0)
df_supply["h_butte"] = pd.to_numeric(df_supply["H BUTTE"] if "H BUTTE" in df_supply.columns else 0, errors="coerce").fillna(0)
df_supply["px"]      = pd.to_numeric(df_supply["PX"]      if "PX"      in df_supply.columns else 0, errors="coerce").fillna(0)
df_supply["os"]      = pd.to_numeric(df_supply["OS"]      if "OS"      in df_supply.columns else 0, errors="coerce").fillna(0)
df_supply = df_supply[(df_supply["hr"] >= 1) & (df_supply["hr"] <= 24)]
df_supply[["date", "hr", "nebo", "h_butte", "px", "os"]].to_csv(output_supply, index=False)
print(f"Wrote {len(df_supply)} supply history rows to {output_supply}")

print("\nDone.")
