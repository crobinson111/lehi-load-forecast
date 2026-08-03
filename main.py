import csv
import io
import json
import os
import re
import time
import asyncio
import logging
import shutil
import tempfile
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import httpx
import pandas as pd
from datetime import datetime, timedelta, date, timezone
from zoneinfo import ZoneInfo
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
import holidays as holidays_lib
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Utah state holidays (includes federal + Pioneer Day Jul 24) through 2040
_HOLIDAYS = holidays_lib.US(state="UT", years=range(2018, 2041))

# NERC/WECC holidays: New Year's, Memorial Day, Independence Day, Labor Day, Thanksgiving, Christmas
_WECC_HOLIDAY_NAMES = {
    "New Year's Day", "Memorial Day", "Independence Day",
    "Labor Day", "Thanksgiving", "Christmas Day",
}
_us_holidays = holidays_lib.US(years=range(2018, 2041))
_WECC_HOLIDAYS: set = {d for d, name in _us_holidays.items() if any(w in name for w in _WECC_HOLIDAY_NAMES)}

def _load_school_calendar() -> set:
    """Returns a set of date objects on which school is out."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "school_calendar.csv")
    out: set = set()
    try:
        df = pd.read_csv(path, parse_dates=["start_date", "end_date"])
        for _, row in df.iterrows():
            d = row["start_date"].date()
            while d <= row["end_date"].date():
                out.add(d)
                d += timedelta(days=1)
        logging.getLogger(__name__).info(f"Loaded school calendar: {len(out)} school-out days")
    except FileNotFoundError:
        logging.getLogger(__name__).warning("school_calendar.csv not found — school-out feature will be 0")
    return out

_SCHOOL_OUT: set = _load_school_calendar()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PACIFIC_TZ = ZoneInfo("America/Los_Angeles")

# Persistent storage directory — set PERSIST_DIR env var to the Render disk mount path
_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR  = os.environ.get("PERSIST_DIR", os.path.join(_BASE_DIR, "data"))
os.makedirs(PERSIST_DIR, exist_ok=True)
CAISO_NODE = "ELAP_PACE-APND"

LEHI_LAT          = 40.3916
LEHI_LON          = -111.8508
BLUFF_LAT         = 37.2879   # Red Mesa solar plant — Bluff, UT
BLUFF_LON         = -109.5512
PLYMOUTH_LAT      = 41.878    # Steele A solar plant — Plymouth, UT
PLYMOUTH_LON      = -112.148
IDAHO_FALLS_LAT   = 43.4917   # Horse Butte wind farm — Idaho Falls, ID
IDAHO_FALLS_LON   = -112.0341

SOLAR_PLANTS: dict = {
    "red-mesa": {
        "label":       "Red Mesa",
        "lat":         BLUFF_LAT,
        "lon":         BLUFF_LON,
        "column":      "RED MESA",
        "csv_env":     "RED_MESA_CSV_URL",
        "weather_env": "SOLAR_WEATHER_CSV_URL",
    },
    "steele-a": {
        "label":       "Steele A",
        "lat":         PLYMOUTH_LAT,
        "lon":         PLYMOUTH_LON,
        "column":      "Steel A",
        "csv_env":     "STEELE_A_CSV_URL",
        "weather_env": "STEELE_A_WEATHER_CSV_URL",
    },
}

EXCEL_PATH = os.environ.get(
    "EXCEL_PATH",
    r"C:\Users\crobinson\OneDrive - Lehi City\Scheduling - Documents\SchLogData.xlsx",
)


def _load_weather_csv(url_env_key: str) -> dict:
    """Load pre-fetched training weather from a CSV URL. Returns {} if not configured."""
    url = os.environ.get(url_env_key)
    if not url:
        return {}
    try:
        resp = httpx.get(url, timeout=30.0)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        result: dict = {}
        for _, row in df.iterrows():
            om_hour = int(row["hr"]) - 1  # hr 1-24 → 0-23
            entry = {k: float(row[k]) for k in df.columns if k not in ("date", "hr")}
            result.setdefault(str(row["date"]), {})[om_hour] = entry
        logger.info(f"Loaded training weather from {url_env_key}: {len(df)} rows")
        return result
    except Exception as exc:
        logger.warning(f"Could not load training weather CSV ({url_env_key}): {exc}")
        return {}

model_state: dict = {
    "model": None,
    "history": {},      # {date_str: {om_hour_0_23: load}}
    "trained_at": None,
    "record_count": 0,
    "date_range": None,
    "r2": None,
    "training": False,
    "error": None,
}

solar_states: dict = {
    key: {
        "model": None,
        "history": {},
        "trained_at": None,
        "record_count": 0,
        "date_range": None,
        "r2": None,
        "training": False,
        "error": None,
    }
    for key in SOLAR_PLANTS
}

WIND_PLANTS: dict = {
    "horse-butte": {
        "label":       "Horse Butte",
        "lat":         IDAHO_FALLS_LAT,
        "lon":         IDAHO_FALLS_LON,
        "column":      "H BUTTE",
        "supply_col":  "h_butte",   # column name in supply_history CSV
        "weather_env": "H_BUTTE_WEATHER_CSV_URL",
    },
}

wind_states: dict = {
    key: {
        "model": None,
        "history": {},
        "trained_at": None,
        "record_count": 0,
        "date_range": None,
        "r2": None,
        "training": False,
        "error": None,
    }
    for key in WIND_PLANTS
}

supply_history: dict = {}  # {date_str: {om_hour (0-23): {nebo, h_butte, px, os}}}
uamps_schedule: dict = {}  # {date_str: {hr_1_24: {crsp, provo_riv, veyo}}} loaded from CSV

_uamps_cache: dict = {}
_UAMPS_CACHE_TTL = 1800  # 30 minutes (for today/future dates — live fetch fallback)

_UAMPS_COLS = {
    # 0-based index after splitting a data line by whitespace
    # Header: HOUR METERS RESTOTAL DIFF IPP CRSP HNTR SAN_JUAN PROVO_RIV MR OS PX PV_WIND NEBO H_BUTTE VEYO OLMSTED RED_MESA SUNNYSIDE STEEL_A
    "crsp":      5,
    "hunter":    6,
    "provo_riv": 8,
    "mr":        9,   # "MR" in log — likely Jordanelle hydro
    "pv_wind":   12,
    "veyo":      15,
    "olmsted":   16,
}


def _uamps_fetch_day_sync(user_id: str, password: str, dt: date) -> dict:
    """Login + fetch one day of UAMPS scheduler log. Returns {hr_1_24: {crsp, provo_riv, veyo}}."""
    import requests as _req
    import urllib3 as _u3
    _u3.disable_warnings()

    BASE = "https://px.uamps.com/cgi-bin/wwiz.asp"
    sess = _req.Session()
    sess.headers["User-Agent"] = "Mozilla/5.0"
    sess.verify = False  # city proxy intercepts SSL; Render won't need this

    r = sess.post(BASE, data={
        "wwizmstr": "WEB.LOGIN", "WWIZ_FORMNO": "0",
        "user": user_id, "pwd": password, "Submit": "Submit",
    }, timeout=30)
    if "logoff" not in r.text.lower():
        raise RuntimeError("UAMPS login failed — check UAMPS_USER_ID / UAMPS_PASSWORD")

    r2 = sess.post(BASE, data={
        "wwizmstr": "WEB.SCHED.LOG.FOR.MBRS", "WWIZ_FORMNO": "0",
        "Destination": "S",
        "Year":   str(dt.year)[-2:],
        "Month":  str(dt.month),
        "Day":    str(dt.day),
        "Submit": "Run Report Now",
    }, timeout=30)

    m = re.search(r'<pre>(.*?)</pre>', r2.text, re.S | re.I)
    if not m:
        raise RuntimeError("No <pre> block in UAMPS response")

    pre_text = re.sub(r'<[^>]+>', '\n', m.group(1))
    # Split at the first long separator (---...) to isolate the kW section only.
    # The <pre> block also contains an MW fractions section which has the same
    # hour numbers but floats, causing int() conversion errors.
    kw_section = re.split(r'-{20,}', pre_text)[0]
    result: dict = {}
    for line in kw_section.splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        hr = int(parts[0])
        if hr < 1 or hr > 24 or len(parts) < 16:
            continue
        result[hr] = {key: int(parts[idx]) for key, idx in _UAMPS_COLS.items()}
    return result


def load_uamps_schedule() -> dict:
    """Load UAMPS schedule CSV. Returns {date_str: {hr_1_24: {crsp, provo_riv, veyo}}}."""
    url = (os.environ.get("UAMPS_SCHEDULE_CSV_URL") or "").strip()
    if not url:
        return {}
    try:
        resp = httpx.get(url, timeout=30.0)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df["date"] = df["date"].astype(str)
        df["hr"]   = df["hr"].astype(int)
        _uamps_resource_cols = ("crsp", "hunter", "provo_riv", "mr", "pv_wind", "veyo", "olmsted")
        for col in _uamps_resource_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            else:
                df[col] = 0
        result: dict = {}
        for _, row in df.iterrows():
            result.setdefault(str(row["date"]), {})[int(row["hr"])] = {
                col: int(row[col]) for col in _uamps_resource_cols
            }
        logger.info(f"Loaded UAMPS schedule: {len(df)} rows ({df['date'].min()} to {df['date'].max()})")
        return result
    except Exception as exc:
        logger.warning(f"Could not load UAMPS schedule CSV: {exc}")
        return {}


async def _uamps_get_day(dt: date) -> dict | None:
    """Returns {hr_1_24: {crsp, provo_riv, veyo}} or None if unavailable.
    Prefers the pre-committed CSV; falls back to live fetch if credentials present."""
    date_str = dt.isoformat()

    # 1. CSV (always available on Render)
    if date_str in uamps_schedule:
        return uamps_schedule[date_str]

    # 2. Live fetch (works locally, blocked on Render)
    user_id  = os.getenv("UAMPS_USER_ID", "")
    password = os.getenv("UAMPS_PASSWORD", "")
    if not user_id or not password:
        return None

    today  = date.today()
    cached = _uamps_cache.get(date_str)
    if cached is not None:
        ts, data = cached
        if dt < today or time.monotonic() - ts < _UAMPS_CACHE_TTL:
            return data

    try:
        data = await asyncio.to_thread(_uamps_fetch_day_sync, user_id, password, dt)
        _uamps_cache[date_str] = (time.monotonic(), data)
        logger.info(f"UAMPS live fetch {date_str}: {len(data)} hours")
        return data
    except Exception as exc:
        logger.warning(f"UAMPS live fetch failed for {date_str}: {exc}")
        return None


def load_plant_data(plant_key: str) -> pd.DataFrame:
    plant = SOLAR_PLANTS[plant_key]
    url = os.environ.get(plant["csv_env"])
    if url:
        resp = httpx.get(url, timeout=30.0)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df["date"] = df["date"].astype(str)
        df["hr"]   = df["hr"].astype(int)
        df["kwh"]  = pd.to_numeric(df["kwh"], errors="coerce").fillna(0)
        return df[["date", "hr", "kwh"]]

    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        tmp.close()
        shutil.copy2(EXCEL_PATH, tmp.name)
    except Exception as exc:
        raise RuntimeError(f"Could not copy spreadsheet: {exc}")

    try:
        df = pd.read_excel(tmp.name, sheet_name="Sch Log Data (2)", engine="openpyxl")
    finally:
        os.unlink(tmp.name)

    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["Date", "Hr"])
    df["date"] = (
        df["Date"].astype(int).astype(str).str.zfill(6)
        .pipe(lambda s: pd.to_datetime(s, format="%y%m%d"))
        .dt.strftime("%Y-%m-%d")
    )
    df["hr"]  = df["Hr"].astype(int)
    df["kwh"] = pd.to_numeric(df[plant["column"]], errors="coerce").fillna(0)
    df = df[(df["hr"] >= 1) & (df["hr"] <= 24)]
    return df[["date", "hr", "kwh"]]


def _wecc_holidays(year: int) -> set:
    """Return WECC holiday date strings (YYYY-MM-DD) for the given year."""
    from datetime import date as _d
    h = {f"{year}-01-01", f"{year}-07-04", f"{year}-12-25"}
    d = _d(year, 5, 31)                          # Memorial Day: last Monday of May
    while d.weekday() != 0:
        d = _d(d.year, d.month, d.day - 1)
    h.add(d.isoformat())
    d = _d(year, 9, 1)                           # Labor Day: first Monday of September
    while d.weekday() != 0:
        d = _d(d.year, d.month, d.day + 1)
    h.add(d.isoformat())
    d, n = _d(year, 11, 1), 0                    # Thanksgiving: fourth Thursday of November
    while n < 4:
        if d.weekday() == 3:
            n += 1
            if n == 4:
                break
        d = _d(d.year, d.month, d.day + 1)
    h.add(d.isoformat())
    return h


def load_supply_history() -> dict:
    """Load Nebo, H Butte, PX, OS hourly data. Returns {date: {om_hour: {nebo, h_butte, px, os}}}."""
    url = os.environ.get("SUPPLY_HISTORY_CSV_URL")
    if url:
        resp = httpx.get(url, timeout=30.0)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
    else:
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
            tmp.close()
            shutil.copy2(EXCEL_PATH, tmp.name)
        except Exception as exc:
            raise RuntimeError(f"Could not copy spreadsheet: {exc}")
        try:
            df_raw = pd.read_excel(tmp.name, sheet_name="Sch Log Data (2)", engine="openpyxl")
        finally:
            os.unlink(tmp.name)
        df_raw.columns = df_raw.columns.str.strip()
        df_raw = df_raw.dropna(subset=["Date", "Hr"])
        df_raw["date"] = (
            df_raw["Date"].astype(int).astype(str).str.zfill(6)
            .pipe(lambda s: pd.to_datetime(s, format="%y%m%d"))
            .dt.strftime("%Y-%m-%d")
        )
        df_raw["hr"]      = df_raw["Hr"].astype(int)
        df_raw["nebo"]    = pd.to_numeric(df_raw["NEBO"]    if "NEBO"    in df_raw.columns else 0, errors="coerce").fillna(0)
        df_raw["h_butte"] = pd.to_numeric(df_raw["H BUTTE"] if "H BUTTE" in df_raw.columns else 0, errors="coerce").fillna(0)
        df_raw["px"]      = pd.to_numeric(df_raw["PX"]      if "PX"      in df_raw.columns else 0, errors="coerce").fillna(0)
        df_raw["os"]      = pd.to_numeric(df_raw["OS"]      if "OS"      in df_raw.columns else 0, errors="coerce").fillna(0)
        df = df_raw[["date", "hr", "nebo", "h_butte", "px", "os"]].copy()

    df["date"] = df["date"].astype(str)
    df["hr"]   = df["hr"].astype(int)
    for col in ["nebo", "h_butte", "px", "os"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    result: dict = {}
    for _, row in df.iterrows():
        om_hour = int(row["hr"]) - 1
        if not (0 <= om_hour <= 23):
            continue
        result.setdefault(str(row["date"]), {})[om_hour] = {
            "nebo":    round(float(row["nebo"]),    1),
            "h_butte": round(float(row["h_butte"]), 1),
            "px":      round(float(row["px"]),      1),
            "os":      round(float(row["os"]),      1),
        }
    return result


def load_excel_data() -> pd.DataFrame:
    data_csv_url = os.environ.get("DATA_CSV_URL")
    if data_csv_url:
        resp = httpx.get(data_csv_url, timeout=30.0)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df["date"] = df["date"].astype(str)
        df["hr"] = df["hr"].astype(int)
        df["load"] = pd.to_numeric(df["load"], errors="coerce")
        df = df[(df["hr"] >= 1) & (df["hr"] <= 24) & (df["load"] > 0)].dropna(subset=["load"])
        return df[["date", "hr", "load"]]

    # Copy to a temp file first so we can read it even if Excel has it open
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        tmp.close()
        shutil.copy2(EXCEL_PATH, tmp.name)
    except Exception as exc:
        raise RuntimeError(f"Could not copy spreadsheet: {exc}")

    try:
        df = pd.read_excel(tmp.name, sheet_name="Sch Log Data (2)", engine="openpyxl")
    finally:
        os.unlink(tmp.name)

    df.columns = df.columns.str.strip()

    required = {"Date", "Hr", "Total Meters plus Gens"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in spreadsheet: {missing}")

    df = df.dropna(subset=list(required))

    # Date is stored as YYMMDD integer (e.g. 181001 → 2018-10-01)
    df["date"] = (
        df["Date"].astype(int).astype(str).str.zfill(6)
        .pipe(lambda s: pd.to_datetime(s, format="%y%m%d"))
        .dt.strftime("%Y-%m-%d")
    )

    df["hr"] = df["Hr"].astype(int)           # 1–24
    df["load"] = pd.to_numeric(df["Total Meters plus Gens"], errors="coerce")

    df = df[(df["hr"] >= 1) & (df["hr"] <= 24) & (df["load"] > 0)].dropna(subset=["load"])
    return df[["date", "hr", "load"]]


_weather_cache: dict = {}
_weather_day_cache: dict = {}   # (date_str, use_forecast_api) → (ts, day_data)
_WEATHER_CACHE_TTL = 7200  # 2 hours
_openmeteo_semaphore: asyncio.Semaphore | None = None  # created after event loop starts


def _get_openmeteo_sem() -> asyncio.Semaphore:
    global _openmeteo_semaphore
    if _openmeteo_semaphore is None:
        _openmeteo_semaphore = asyncio.Semaphore(2)
    return _openmeteo_semaphore

async def fetch_weather(start_date: str, end_date: str, use_forecast_api: bool = False) -> dict:
    """Returns {date_str: {openmeteo_hour_0_to_23: {temp_f, apparent_f}}}"""
    # Fast path: single-day lookup — check per-day cache first
    if start_date == end_date:
        day_key = (start_date, use_forecast_api)
        cached_day = _weather_day_cache.get(day_key)
        if cached_day and time.monotonic() - cached_day[0] < _WEATHER_CACHE_TTL:
            return {start_date: cached_day[1]}

    cache_key = (start_date, end_date, use_forecast_api)
    cached = _weather_cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < _WEATHER_CACHE_TTL:
        return cached[1]

    base_url = (
        "https://api.open-meteo.com/v1/forecast"
        if use_forecast_api
        else "https://archive-api.open-meteo.com/v1/archive"
    )
    params = {
        "latitude": LEHI_LAT,
        "longitude": LEHI_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,apparent_temperature",
        "temperature_unit": "fahrenheit",
        "timezone": "America/Denver",
    }
    resp = None
    for attempt in range(3):
        async with _get_openmeteo_sem():
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(base_url, params=params)
        if resp.status_code == 429:
            wait = 15 * (attempt + 1)
            logger.warning(f"Open-Meteo rate limit hit, retrying in {wait}s...")
            await asyncio.sleep(wait)
            continue
        resp.raise_for_status()
        break
    else:
        resp.raise_for_status()

    result: dict = {}
    data = resp.json()
    times      = data["hourly"]["time"]
    temps      = data["hourly"]["temperature_2m"]
    app_temps  = data["hourly"]["apparent_temperature"]
    for t, temp, app in zip(times, temps, app_temps):
        if temp is None:
            continue
        dt = datetime.fromisoformat(t)
        result.setdefault(dt.strftime("%Y-%m-%d"), {})[dt.hour] = {
            "temp_f": temp,
            "apparent_f": app if app is not None else temp,
        }
    ts = time.monotonic()
    _weather_cache[cache_key] = (ts, result)
    # Populate per-day cache so pre-warmed ranges satisfy single-day lookups
    for date_str, day_data in result.items():
        _weather_day_cache[(date_str, use_forecast_api)] = (ts, day_data)
    return result


async def fetch_solar_weather(start_date: str, end_date: str, lat: float, lon: float,
                              use_forecast_api: bool = False) -> dict:
    """Returns {date_str: {hour_0_23: {"ghi": float, "temp_f": float}}}."""
    cache_key = ("solar", lat, lon, start_date, end_date, use_forecast_api)
    cached = _weather_cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < _WEATHER_CACHE_TTL:
        return cached[1]

    base_url = (
        "https://api.open-meteo.com/v1/forecast"
        if use_forecast_api
        else "https://archive-api.open-meteo.com/v1/archive"
    )
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "shortwave_radiation,temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "America/Denver",
    }
    resp = None
    for attempt in range(3):
        async with _get_openmeteo_sem():
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(base_url, params=params)
        if resp.status_code == 429:
            wait = 15 * (attempt + 1)
            logger.warning(f"Open-Meteo rate limit (solar), retrying in {wait}s...")
            await asyncio.sleep(wait)
            continue
        resp.raise_for_status()
        break
    else:
        resp.raise_for_status()

    result: dict = {}
    data = resp.json()
    times = data["hourly"]["time"]
    ghis  = data["hourly"]["shortwave_radiation"]
    temps = data["hourly"]["temperature_2m"]
    for t, ghi, temp in zip(times, ghis, temps):
        if ghi is None:
            continue
        dt = datetime.fromisoformat(t)
        result.setdefault(dt.strftime("%Y-%m-%d"), {})[dt.hour] = {
            "ghi": ghi,
            "temp_f": temp if temp is not None else 70.0,
        }
    _weather_cache[cache_key] = (time.monotonic(), result)
    return result


async def fetch_wind_weather(start_date: str, end_date: str, lat: float, lon: float,
                             use_forecast_api: bool = False) -> dict:
    """Returns {date_str: {hour_0_23: {"wind_speed": float, "wind_dir": float}}}."""
    cache_key = ("wind", lat, lon, start_date, end_date, use_forecast_api)
    cached = _weather_cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < _WEATHER_CACHE_TTL:
        return cached[1]

    base_url = (
        "https://api.open-meteo.com/v1/forecast"
        if use_forecast_api
        else "https://archive-api.open-meteo.com/v1/archive"
    )
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": "wind_speed_100m,wind_direction_100m",
        "timezone": "America/Denver",
    }
    resp = None
    for attempt in range(3):
        async with _get_openmeteo_sem():
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(base_url, params=params)
        if resp.status_code == 429:
            wait = 15 * (attempt + 1)
            logger.warning(f"Open-Meteo rate limit (wind), retrying in {wait}s...")
            await asyncio.sleep(wait)
            continue
        resp.raise_for_status()
        break
    else:
        resp.raise_for_status()

    result: dict = {}
    data = resp.json()
    times  = data["hourly"]["time"]
    speeds = data["hourly"]["wind_speed_100m"]
    dirs   = data["hourly"]["wind_direction_100m"]
    for t, spd, d in zip(times, speeds, dirs):
        if spd is None:
            continue
        dt = datetime.fromisoformat(t)
        result.setdefault(dt.strftime("%Y-%m-%d"), {})[dt.hour] = {
            "wind_speed": spd,
            "wind_dir":   d if d is not None else 0.0,
        }
    _weather_cache[cache_key] = (time.monotonic(), result)
    return result


def build_solar_features(hr_1_to_24: int, ghi: float, temp_f: float, date_str: str) -> np.ndarray:
    angle   = 2 * np.pi * hr_1_to_24 / 24
    sin_hr  = np.sin(angle);     cos_hr  = np.cos(angle)
    sin_hr2 = np.sin(2*angle);   cos_hr2 = np.cos(2*angle)

    d   = datetime.strptime(date_str, "%Y-%m-%d").date()
    doy = d.timetuple().tm_yday
    sin_doy = np.sin(2*np.pi*doy/365.25)
    cos_doy = np.cos(2*np.pi*doy/365.25)

    return np.array([[
        ghi, ghi**2,
        ghi * sin_doy, ghi * cos_doy,  # seasonal efficiency variation
        sin_hr, cos_hr,
        sin_hr2, cos_hr2,
        sin_doy, cos_doy,
        temp_f,
    ]])


def build_wind_features(hr_1_to_24: int, wind_speed: float, wind_dir_deg: float, date_str: str) -> np.ndarray:
    angle   = 2 * np.pi * hr_1_to_24 / 24
    sin_hr  = np.sin(angle);   cos_hr  = np.cos(angle)
    sin_hr2 = np.sin(2*angle); cos_hr2 = np.cos(2*angle)

    d   = datetime.strptime(date_str, "%Y-%m-%d").date()
    doy = d.timetuple().tm_yday
    sin_doy = np.sin(2*np.pi*doy/365.25)
    cos_doy = np.cos(2*np.pi*doy/365.25)

    wind_dir_rad = np.deg2rad(wind_dir_deg)
    sin_dir = np.sin(wind_dir_rad)
    cos_dir = np.cos(wind_dir_rad)

    ws2 = wind_speed ** 2
    ws3 = wind_speed ** 3

    return np.array([[
        wind_speed, ws2, ws3,
        wind_speed * sin_doy, wind_speed * cos_doy,  # seasonal wind-power interaction
        sin_dir, cos_dir,
        sin_hr, cos_hr,
        sin_hr2, cos_hr2,
        sin_doy, cos_doy,
    ]])


SOLAR_TRAINING_YEARS = 3


async def train_plant_model(plant_key: str) -> None:
    plant = SOLAR_PLANTS[plant_key]
    state = solar_states[plant_key]
    state["training"] = True
    state["error"]    = None
    try:
        logger.info(f"Loading {plant['label']} data...")
        df = load_plant_data(plant_key)
        df = df[df["kwh"] >= 0]

        cutoff = (pd.to_datetime(df["date"].max()) - pd.DateOffset(years=SOLAR_TRAINING_YEARS)).strftime("%Y-%m-%d")
        train_df = df[df["date"] >= cutoff].copy()
        logger.info(f"{plant['label']} training on {len(train_df)} rows from {train_df['date'].min()} to {train_df['date'].max()}")

        weather = _load_weather_csv(plant["weather_env"])
        if not weather:
            logger.info(f"Fetching solar weather for {plant['label']}...")
            weather = await fetch_solar_weather(
                train_df["date"].min(), train_df["date"].max(),
                plant["lat"], plant["lon"],
            )
        else:
            logger.info(f"Using pre-loaded training weather for {plant['label']}")

        train_df["om_hour"] = train_df["hr"] - 1
        train_df["ghi"]    = train_df.apply(lambda r: (weather.get(r["date"], {}).get(r["om_hour"]) or {}).get("ghi"),    axis=1)
        train_df["temp_f"] = train_df.apply(lambda r: (weather.get(r["date"], {}).get(r["om_hour"]) or {}).get("temp_f"), axis=1)
        train_df = train_df.dropna(subset=["ghi", "temp_f"])
        train_df = train_df[train_df["ghi"] > 10]

        if len(train_df) < 24:
            raise RuntimeError(f"Only {len(train_df)} daytime rows after joining weather data.")

        X = np.vstack(train_df.apply(
            lambda r: build_solar_features(r["hr"], r["ghi"], r["temp_f"], r["date"])[0], axis=1
        ).values)
        y = train_df["kwh"].values

        mdl = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        mdl.fit(X, y)
        r2 = float(mdl.score(X, y))

        history: dict = {}
        for _, row in df.iterrows():
            om_hour = int(row["hr"]) - 1
            history.setdefault(row["date"], {})[om_hour] = round(float(row["kwh"]), 1)

        state.update({
            "model": mdl,
            "history": history,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "record_count": int(len(y)),
            "date_range": [train_df["date"].min(), train_df["date"].max()],
            "r2": round(r2, 3),
        })
        logger.info(f"{plant['label']} model ready — {len(y):,} daytime samples, R²={r2:.3f}")

    except Exception as exc:
        logger.error(f"{plant['label']} training failed: {exc}")
        state["error"] = str(exc)
    finally:
        state["training"] = False


def _nebo_scheduled() -> float:
    """Nebo fixed schedule: 10,000 kWh every hour."""
    return 10000.0


def load_wind_data(plant_key: str) -> pd.DataFrame:
    """Load wind plant history from supply_history CSV (reuses already-fetched file)."""
    plant = WIND_PLANTS[plant_key]
    supply_url = os.environ.get("SUPPLY_HISTORY_CSV_URL")
    if supply_url:
        resp = httpx.get(supply_url, timeout=30.0)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df["kwh"] = pd.to_numeric(df[plant["supply_col"]], errors="coerce").fillna(0)
        df["date"] = df["date"].astype(str)
        df["hr"]   = df["hr"].astype(int)
        return df[["date", "hr", "kwh"]]

    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        tmp.close()
        shutil.copy2(EXCEL_PATH, tmp.name)
    except Exception as exc:
        raise RuntimeError(f"Could not copy spreadsheet: {exc}")
    try:
        df = pd.read_excel(tmp.name, sheet_name="Sch Log Data (2)", engine="openpyxl")
    finally:
        os.unlink(tmp.name)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["Date", "Hr"])
    df["date"] = (
        df["Date"].astype(int).astype(str).str.zfill(6)
        .pipe(lambda s: pd.to_datetime(s, format="%y%m%d"))
        .dt.strftime("%Y-%m-%d")
    )
    df["hr"]  = df["Hr"].astype(int)
    df["kwh"] = pd.to_numeric(df[plant["column"]], errors="coerce").fillna(0)
    df = df[(df["hr"] >= 1) & (df["hr"] <= 24)]
    return df[["date", "hr", "kwh"]]


WIND_TRAINING_YEARS = 3


async def train_wind_model(plant_key: str) -> None:
    plant = WIND_PLANTS[plant_key]
    state = wind_states[plant_key]
    state["training"] = True
    state["error"]    = None
    try:
        logger.info(f"Loading {plant['label']} wind data...")
        df = load_wind_data(plant_key)
        df = df[df["kwh"] >= 0]

        cutoff = (pd.to_datetime(df["date"].max()) - pd.DateOffset(years=WIND_TRAINING_YEARS)).strftime("%Y-%m-%d")
        train_df = df[df["date"] >= cutoff].copy()
        logger.info(f"{plant['label']} training on {len(train_df)} rows from {train_df['date'].min()} to {train_df['date'].max()}")

        weather = _load_weather_csv(plant["weather_env"])
        if not weather:
            logger.info(f"Fetching wind weather for {plant['label']}...")
            weather = await fetch_wind_weather(
                train_df["date"].min(), train_df["date"].max(),
                plant["lat"], plant["lon"],
            )
        else:
            logger.info(f"Using pre-loaded training weather for {plant['label']}")

        train_df["om_hour"]    = train_df["hr"] - 1
        train_df["wind_speed"] = train_df.apply(lambda r: (weather.get(r["date"], {}).get(r["om_hour"]) or {}).get("wind_speed"), axis=1)
        train_df["wind_dir"]   = train_df.apply(lambda r: (weather.get(r["date"], {}).get(r["om_hour"]) or {}).get("wind_dir"),   axis=1)
        train_df = train_df.dropna(subset=["wind_speed", "wind_dir"])

        if len(train_df) < 24:
            raise RuntimeError(f"Only {len(train_df)} rows after joining weather data.")

        X = np.vstack(train_df.apply(
            lambda r: build_wind_features(r["hr"], r["wind_speed"], r["wind_dir"], r["date"])[0], axis=1
        ).values)
        y = train_df["kwh"].values

        mdl = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        mdl.fit(X, y)
        r2 = float(mdl.score(X, y))

        history: dict = {}
        for _, row in df.iterrows():
            om_hour = int(row["hr"]) - 1
            history.setdefault(row["date"], {})[om_hour] = round(float(row["kwh"]), 1)

        state.update({
            "model": mdl,
            "history": history,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "record_count": int(len(y)),
            "date_range": [train_df["date"].min(), train_df["date"].max()],
            "r2": round(r2, 3),
        })
        logger.info(f"{plant['label']} wind model ready — {len(y):,} samples, R²={r2:.3f}")

    except Exception as exc:
        logger.error(f"{plant['label']} wind training failed: {exc}")
        state["error"] = str(exc)
    finally:
        state["training"] = False


def _px_scheduled(om_hour: int, dt: date) -> float:
    """PX fixed schedule: Mon-Sat in July only, excluding WECC holidays."""
    if dt in _WECC_HOLIDAYS or dt.month != 7 or dt.weekday() == 6:
        return 0.0
    if 7 <= om_hour <= 12:
        return 38210.0
    if 13 <= om_hour <= 20:
        return 47754.0
    if 21 <= om_hour <= 22:
        return 38210.0
    return 0.0


def _os_scheduled(om_hour: int, dt: date) -> float:
    """OS fixed schedule: Mon-Sat 7am-10pm, excluding Sundays and WECC holidays."""
    if dt in _WECC_HOLIDAYS or dt.weekday() == 6:
        return 0.0
    return 20000.0 if 7 <= om_hour <= 22 else 0.0


async def _plant_hourly_kwh(plant_key: str, target_date: str, dt: date) -> dict:
    """Returns {om_hour: kwh} for a solar plant. Returns {} if model not ready."""
    state = solar_states[plant_key]
    cfg = SOLAR_PLANTS[plant_key]
    if state["model"] is None:
        if not state["training"] and state["error"] is None:
            asyncio.create_task(train_plant_model(plant_key))
        return {}
    if target_date in state["history"]:
        return state["history"][target_date]
    use_forecast_api = dt >= date.today() - timedelta(days=7)
    try:
        weather = await fetch_solar_weather(
            target_date, target_date, cfg["lat"], cfg["lon"],
            use_forecast_api=use_forecast_api,
        )
        day_wx = weather.get(target_date, {})
    except Exception:
        return {}
    mdl = state["model"]
    result: dict = {}
    for om_hour in range(24):
        wx = day_wx.get(om_hour)
        if wx is None or wx["ghi"] <= 10:
            result[om_hour] = 0.0
        else:
            pred = float(max(mdl.predict(build_solar_features(om_hour + 1, wx["ghi"], wx["temp_f"], target_date))[0], 0))
            result[om_hour] = round(pred, 1)
    return result


async def _wind_plant_hourly_kwh(plant_key: str, target_date: str, dt: date) -> dict:
    """Returns {om_hour: kwh} for a wind plant. Returns {} if model not ready."""
    state = wind_states[plant_key]
    cfg   = WIND_PLANTS[plant_key]
    if state["model"] is None:
        if not state["training"] and state["error"] is None:
            asyncio.create_task(train_wind_model(plant_key))
        return {}
    if target_date in state["history"]:
        return state["history"][target_date]
    use_forecast_api = dt >= date.today() - timedelta(days=7)
    try:
        weather = await fetch_wind_weather(
            target_date, target_date, cfg["lat"], cfg["lon"],
            use_forecast_api=use_forecast_api,
        )
        day_wx = weather.get(target_date, {})
    except Exception:
        return {}
    mdl = state["model"]
    result: dict = {}
    for om_hour in range(24):
        wx = day_wx.get(om_hour)
        if wx is None:
            result[om_hour] = 0.0
        else:
            pred = float(max(mdl.predict(build_wind_features(om_hour + 1, wx["wind_speed"], wx["wind_dir"], target_date))[0], 0))
            result[om_hour] = round(pred, 1)
    return result


TRAINING_YEARS = 2  # only use this many recent years for fitting

def build_features(hr_1_to_24: int, temp_f: float, apparent_f: float, date_str: str) -> np.ndarray:
    """hr_1_to_24 matches the spreadsheet convention (1=midnight, 24=11pm)."""
    # Three harmonics of the daily cycle — fundamental + 2nd + 3rd
    # This allows the model to represent the asymmetric M-shaped load curve
    # (sharp morning commercial ramp, midday plateau, evening decline) rather
    # than the smooth single-peak shape that one harmonic produces.
    angle = 2 * np.pi * hr_1_to_24 / 24
    sin_hr  = np.sin(angle);      cos_hr  = np.cos(angle)
    sin_hr2 = np.sin(2 * angle);  cos_hr2 = np.cos(2 * angle)
    sin_hr3 = np.sin(3 * angle);  cos_hr3 = np.cos(3 * angle)

    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    dow = d.weekday()  # 0=Mon … 5=Sat, 6=Sun
    is_saturday = 1.0 if dow == 5 else 0.0
    is_sunday   = 1.0 if dow == 6 else 0.0

    doy = d.timetuple().tm_yday
    sin_doy = np.sin(2 * np.pi * doy / 365.25)
    cos_doy = np.cos(2 * np.pi * doy / 365.25)

    heat_index = apparent_f - temp_f  # humidity/wind effect independent of raw temp

    # Holiday features
    is_holiday = 1.0 if d in _HOLIDAYS else 0.0
    week_mon = d - timedelta(days=dow)
    is_holiday_week = 1.0 if any((week_mon + timedelta(days=i)) in _HOLIDAYS for i in range(7)) else 0.0

    # School calendar — interacted with season so the model learns that
    # summer school-out (kids home + AC) differs from winter break (lower load).
    is_school_out = 1.0 if d in _SCHOOL_OUT else 0.0
    school_x_sin_doy = is_school_out * sin_doy
    school_x_cos_doy = is_school_out * cos_doy

    # Year-over-year load growth trend. Without this, Ridge averages 2024/2025/2026
    # observations together and systematically underpredicts the current year.
    # Expressed as decimal years since 2024 so the coefficient is interpretable
    # (~kW per year of city load growth).
    year_trend = d.year + d.timetuple().tm_yday / 365.25 - 2024.0

    return np.array([[
        sin_hr,  cos_hr,
        sin_hr2, cos_hr2,
        sin_hr3, cos_hr3,
        temp_f, temp_f ** 2,
        apparent_f,
        heat_index,
        is_saturday, is_sunday,
        sin_doy, cos_doy,
        is_holiday,
        is_holiday_week,
        is_school_out,
        school_x_sin_doy,   # summer school-out vs winter break behave differently
        school_x_cos_doy,
        year_trend,
    ]])


async def train_model() -> None:
    model_state["training"] = True
    model_state["error"] = None
    try:
        logger.info("Reading spreadsheet...")
        df = load_excel_data()
        logger.info(f"Loaded {len(df)} rows from {df['date'].min()} to {df['date'].max()}")

        # Trim to most recent TRAINING_YEARS so the model reflects current load levels
        cutoff = (pd.to_datetime(df["date"].max()) - pd.DateOffset(years=TRAINING_YEARS)).strftime("%Y-%m-%d")
        train_df = df[df["date"] >= cutoff].copy()
        logger.info(f"Training on {len(train_df)} rows from {train_df['date'].min()} to {train_df['date'].max()}")

        weather = _load_weather_csv("LOAD_WEATHER_CSV_URL")
        if not weather:
            logger.info("Fetching historical weather from Open-Meteo...")
            weather = await fetch_weather(train_df["date"].min(), train_df["date"].max())
        else:
            logger.info("Using pre-loaded training weather (no API call needed)")

        # Spreadsheet hr 1–24 maps to Open-Meteo hour 0–23 (hr - 1)
        train_df["om_hour"] = train_df["hr"] - 1
        train_df["temp_f"]     = train_df.apply(lambda r: (weather.get(r["date"], {}).get(r["om_hour"]) or {}).get("temp_f"),     axis=1)
        train_df["apparent_f"] = train_df.apply(lambda r: (weather.get(r["date"], {}).get(r["om_hour"]) or {}).get("apparent_f"), axis=1)
        train_df = train_df.dropna(subset=["temp_f", "apparent_f"])

        if len(train_df) < 24:
            raise RuntimeError(f"Only {len(train_df)} matched rows after joining with weather data.")

        X = np.vstack(train_df.apply(lambda r: build_features(r["hr"], r["temp_f"], r["apparent_f"], r["date"])[0], axis=1).values)
        y = train_df["load"].values

        # Exponential decay with 14-day half-life so the model is anchored to
        # recent actuals rather than a 2-year average.
        max_date = pd.to_datetime(train_df["date"].max())
        days_ago = (max_date - pd.to_datetime(train_df["date"])).dt.days.values
        time_weights = np.exp(-days_ago / 14.0)

        # Same-day-type multiplier: weekday rows get 5× weight when the most
        # recent training day is a weekday (and vice versa for weekends).
        # This anchors weekday forecasts to recent weekday actuals.
        max_dow = max_date.dayofweek  # 0=Mon … 6=Sun
        max_is_weekday = max_dow < 5
        train_dow = pd.to_datetime(train_df["date"]).dt.dayofweek.values
        same_type = (train_dow < 5) == max_is_weekday
        day_type_mult = np.where(same_type, 5.0, 1.0)

        sample_weights = time_weights * day_type_mult

        mdl = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        mdl.fit(X, y, ridge__sample_weight=sample_weights)
        r2 = float(mdl.score(X, y))

        # Build historical lookup: {date: {om_hour (0-23): load}}
        raw_df = load_excel_data()
        history: dict = {}
        for _, row in raw_df.iterrows():
            om_hour = int(row["hr"]) - 1   # convert spreadsheet hr 1-24 → 0-23
            history.setdefault(row["date"], {})[om_hour] = round(float(row["load"]), 1)

        model_state.update({
            "model": mdl,
            "history": history,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "record_count": int(len(y)),
            "date_range": [train_df["date"].min(), train_df["date"].max()],
            "r2": round(r2, 3),
        })
        logger.info(f"Model ready — {len(y):,} samples, R²={r2:.3f}")

    except Exception as exc:
        logger.error(f"Training failed: {exc}")
        model_state["error"] = str(exc)
    finally:
        model_state["training"] = False


async def _init_supply() -> None:
    global supply_history, uamps_schedule
    try:
        supply_history = await asyncio.to_thread(load_supply_history)
        logger.info(f"Supply history loaded: {len(supply_history)} dates")
    except Exception as exc:
        logger.error(f"Failed to load supply history: {exc}")
    try:
        uamps_schedule = await asyncio.to_thread(load_uamps_schedule)
    except Exception as exc:
        logger.error(f"Failed to load UAMPS schedule: {exc}")


async def _prewarm_weather_cache() -> None:
    """Fetch today + 8 days in one API call so per-day lookups hit the cache."""
    await asyncio.sleep(5)  # let other startup tasks settle first
    today_str = date.today().strftime("%Y-%m-%d")
    end_str   = (date.today() + timedelta(days=8)).strftime("%Y-%m-%d")
    try:
        await fetch_weather(today_str, end_str, use_forecast_api=True)
        logger.info(f"Load weather cache pre-warmed: {today_str} → {end_str}")
    except Exception as exc:
        logger.warning(f"Weather cache pre-warm failed (will retry on first request): {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(train_model())
    asyncio.create_task(_init_supply())
    asyncio.create_task(_prewarm_weather_cache())
    # Solar models train on demand (first solar request) — not at startup
    yield


app = FastAPI(title="Lehi Load Forecast", lifespan=lifespan)


@app.get("/api/status")
async def status():
    return {
        "trained": model_state["model"] is not None,
        "training": model_state["training"],
        "trained_at": model_state["trained_at"],
        "record_count": model_state["record_count"],
        "date_range": model_state["date_range"],
        "r2": model_state["r2"],
        "error": model_state["error"],
    }


@app.post("/api/train")
async def retrain(background_tasks: BackgroundTasks):
    if model_state["training"]:
        return {"message": "Training already in progress."}
    background_tasks.add_task(train_model)
    return {"message": "Retraining started."}


def _hourly_to_csv(hourly: list) -> Response:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Hour", "Temp (°F)", "Load (kW)"])
    for h in hourly:
        writer.writerow([h["hour"], h["temp_f"], h["load"]])
    return Response(content=buf.getvalue(), media_type="text/csv")


def _solar_to_csv(hourly: list) -> Response:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Hour", "GHI (W/m²)", "Generation (kWh)"])
    for h in hourly:
        writer.writerow([h["hour"], h["ghi"], h["kwh"]])
    return Response(content=buf.getvalue(), media_type="text/csv")


@app.get("/api/forecast")
async def forecast(
    target_date: str = Query(..., alias="date", description="YYYY-MM-DD"),
    fmt: str = Query("json", alias="format", description="Response format: json or csv"),
):
    if model_state["model"] is None:
        detail = (
            "Model is still training, please wait..."
            if model_state["training"]
            else f"Model not ready: {model_state['error']}"
        )
        raise HTTPException(status_code=503, detail=detail)

    try:
        dt = datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # --- Historical: date exists in the spreadsheet ---
    if target_date in model_state["history"]:
        day_hist = model_state["history"][target_date]

        try:
            weather = await fetch_weather(target_date, target_date, use_forecast_api=False)
            day_temps = weather.get(target_date, {})
        except Exception:
            day_temps = {}

        hourly = []
        for om_hour in range(24):
            load = day_hist.get(om_hour)
            wx = day_temps.get(om_hour)
            hourly.append({
                "hour": om_hour,
                "temp_f": round(wx["temp_f"], 1) if wx else None,
                "load": load,
            })

        loads = [h["load"] for h in hourly if h["load"] is not None]
        peak = max(loads) if loads else None
        summary = {
            "peak_load": peak,
            "peak_hour": next((h["hour"] for h in hourly if h["load"] == peak), None),
            "min_load": min(loads) if loads else None,
            "avg_load": round(sum(loads) / len(loads), 1) if loads else None,
        }
        if fmt == "csv":
            return _hourly_to_csv(hourly)
        return {"date": target_date, "type": "historical", "hourly": hourly, "summary": summary}

    # --- Forecast: future or recent date not yet in spreadsheet ---
    today = date.today()
    if dt > today + timedelta(days=16):
        raise HTTPException(status_code=400, detail="Open-Meteo only forecasts up to 16 days ahead.")

    use_forecast_api = dt >= today - timedelta(days=7)
    try:
        weather = await fetch_weather(target_date, target_date, use_forecast_api=use_forecast_api)
        day_temps = weather.get(target_date, {})
    except Exception as exc:
        if "429" in str(exc):
            raise HTTPException(status_code=503, detail="Open-Meteo is rate-limiting this server — please try again in a minute.")
        raise HTTPException(status_code=502, detail=f"Weather API error: {exc}")

    if not day_temps:
        raise HTTPException(status_code=404, detail=f"No weather data available for {target_date}.")

    mdl = model_state["model"]
    hourly = []
    for om_hour in range(24):
        dataset_hr = om_hour + 1
        wx = day_temps.get(om_hour)
        if wx is None:
            hourly.append({"hour": om_hour, "temp_f": None, "load": None})
            continue
        temp     = wx["temp_f"]
        apparent = wx["apparent_f"]
        pred = float(max(mdl.predict(build_features(dataset_hr, temp, apparent, target_date))[0], 0))
        hourly.append({"hour": om_hour, "temp_f": round(temp, 1), "load": round(pred, 1)})

    # ── Intraday bias correction ───────────────────────────────────────────
    # When forecasting today, compare the last ≤3 completed hours of actuals
    # against what the model predicted, then shift all remaining hours by that
    # average error so the forecast tracks what's actually happening.
    intraday_bias = 0.0
    intraday_hours_used = 0
    if dt == today:
        rt_path = os.path.join(PERSIST_DIR, "realtime_load.json")
        if not os.path.exists(rt_path):
            rt_path = os.path.join(_BASE_DIR, "data", "realtime_load.json")
        try:
            with open(rt_path) as _f:
                rt = json.load(_f)
            if rt.get("date") == target_date:
                now_mdt = datetime.now(ZoneInfo("America/Denver"))
                # Current MDT hour (0-23); hours strictly before this are completed
                current_om_hour = now_mdt.hour
                # Collect up to last 3 completed hours that have actuals
                bias_samples = []
                for h in hourly:
                    om_h = h["hour"]
                    if om_h >= current_om_hour:
                        continue  # not yet completed
                    uamps_hr = str(om_h + 1)  # UAMPS hour-ending: om_hour 0 → "1"
                    actual = rt["hours"].get(uamps_hr)
                    if actual is not None and h["load"] is not None:
                        bias_samples.append(actual - h["load"])
                # Use only the last 3
                bias_samples = bias_samples[-3:]
                if bias_samples:
                    intraday_bias = round(sum(bias_samples) / len(bias_samples), 1)
                    intraday_hours_used = len(bias_samples)
                    # Apply bias to all future (not-yet-completed) hours
                    for h in hourly:
                        if h["hour"] >= current_om_hour and h["load"] is not None:
                            h["load"] = round(h["load"] + intraday_bias, 1)
        except Exception:
            pass  # don't break forecast if realtime data is unavailable
    # ── End intraday bias correction ──────────────────────────────────────

    loads = [h["load"] for h in hourly if h["load"] is not None]
    peak = max(loads) if loads else None
    summary = {
        "peak_load": peak,
        "peak_hour": next((h["hour"] for h in hourly if h["load"] == peak), None),
        "min_load": min(loads) if loads else None,
        "avg_load": round(sum(loads) / len(loads), 1) if loads else None,
    }
    if fmt == "csv":
        return _hourly_to_csv(hourly)
    return {
        "date": target_date,
        "type": "forecast",
        "hourly": hourly,
        "summary": summary,
        "intraday_bias": intraday_bias,
        "intraday_hours_used": intraday_hours_used,
    }


@app.get("/api/accuracy")
async def accuracy(target_date: str = Query(..., alias="date", description="YYYY-MM-DD past date")):
    if model_state["model"] is None:
        detail = "Model is still training, please wait..." if model_state["training"] else f"Model not ready: {model_state['error']}"
        raise HTTPException(status_code=503, detail=detail)
    try:
        datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    if target_date not in model_state["history"]:
        raise HTTPException(status_code=404, detail=f"No recorded actuals for {target_date}. Choose a date that is already in the load history.")

    day_hist = model_state["history"][target_date]
    try:
        weather = await fetch_weather(target_date, target_date, use_forecast_api=False)
        day_temps = weather.get(target_date, {})
    except Exception as exc:
        if "429" in str(exc):
            raise HTTPException(status_code=503, detail="Open-Meteo is rate-limiting this server — please try again in a minute.")
        raise HTTPException(status_code=502, detail=f"Weather API error: {exc}")

    mdl = model_state["model"]
    hourly = []
    for om_hour in range(24):
        actual = day_hist.get(om_hour)
        wx = day_temps.get(om_hour)
        forecast_val = None
        if wx is not None:
            pred = mdl.predict(build_features(om_hour + 1, wx["temp_f"], wx["apparent_f"], target_date))[0]
            forecast_val = round(float(max(pred, 0)), 1)
        diff     = round(actual - forecast_val, 1)       if actual is not None and forecast_val is not None else None
        diff_pct = round((actual - forecast_val) / actual * 100, 1) if actual and forecast_val else None
        hourly.append({"hour": om_hour, "temp_f": round(wx["temp_f"], 1) if wx else None,
                        "actual": actual, "forecast": forecast_val, "diff": diff, "diff_pct": diff_pct})

    paired = [(h["actual"], h["forecast"]) for h in hourly if h["actual"] is not None and h["forecast"] is not None]
    mae  = round(sum(abs(a - f) for a, f in paired) / len(paired), 1) if paired else None
    mape = round(sum(abs(a - f) / a * 100 for a, f in paired if a) / len(paired), 2) if paired else None
    actuals   = [h["actual"]   for h in hourly if h["actual"]   is not None]
    forecasts = [h["forecast"] for h in hourly if h["forecast"] is not None]
    actual_peak   = max(actuals)   if actuals   else None
    forecast_peak = max(forecasts) if forecasts else None
    return {
        "date": target_date,
        "hourly": hourly,
        "metrics": {
            "mae": mae, "mape": mape,
            "actual_peak": actual_peak,
            "actual_peak_hour": next((h["hour"] for h in hourly if h["actual"] == actual_peak), None),
            "forecast_peak": forecast_peak,
            "forecast_peak_hour": next((h["hour"] for h in hourly if h["forecast"] == forecast_peak), None),
        },
    }


def _get_plant_or_404(plant: str):
    if plant not in SOLAR_PLANTS:
        raise HTTPException(status_code=404, detail=f"Unknown plant '{plant}'. Valid: {list(SOLAR_PLANTS)}")
    return SOLAR_PLANTS[plant], solar_states[plant]


def _get_wind_or_404(plant: str):
    if plant not in WIND_PLANTS:
        raise HTTPException(status_code=404, detail=f"Unknown wind plant '{plant}'. Valid: {list(WIND_PLANTS)}")
    return WIND_PLANTS[plant], wind_states[plant]


@app.get("/api/solar/{plant}/status")
async def solar_plant_status(plant: str):
    cfg, state = _get_plant_or_404(plant)
    if state["model"] is None and not state["training"] and state["error"] is None:
        asyncio.create_task(train_plant_model(plant))
    return {
        "plant":        plant,
        "label":        cfg["label"],
        "ready":        state["model"] is not None,
        "training":     state["training"],
        "error":        state["error"],
        "trained_at":   state["trained_at"],
        "record_count": state["record_count"],
        "date_range":   state["date_range"],
        "r2":           state["r2"],
    }


@app.post("/api/solar/{plant}/train")
async def solar_plant_train(plant: str, background_tasks: BackgroundTasks):
    cfg, state = _get_plant_or_404(plant)
    if state["training"]:
        raise HTTPException(status_code=409, detail=f"{cfg['label']} model is already training.")
    background_tasks.add_task(train_plant_model, plant)
    return {"message": f"{cfg['label']} model training started."}


@app.get("/api/solar/{plant}/forecast")
async def solar_plant_forecast(
    plant: str,
    target_date: str = Query(..., alias="date", description="YYYY-MM-DD"),
    fmt: str = Query("json", alias="format"),
):
    cfg, state = _get_plant_or_404(plant)
    if state["model"] is None:
        if not state["training"]:
            asyncio.create_task(train_plant_model(plant))
        raise HTTPException(status_code=503, detail=f"{cfg['label']} model is training — please try again in about 30 seconds.")
    try:
        dt = datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    if dt > date.today() + timedelta(days=16):
        raise HTTPException(status_code=400, detail="Open-Meteo only forecasts up to 16 days ahead.")

    if target_date in state["history"]:
        day_hist = state["history"][target_date]
        try:
            weather = await fetch_solar_weather(target_date, target_date, cfg["lat"], cfg["lon"], use_forecast_api=False)
            day_wx = weather.get(target_date, {})
        except Exception:
            day_wx = {}
        hourly = []
        for om_hour in range(24):
            kwh = day_hist.get(om_hour, 0.0)
            wx = day_wx.get(om_hour)
            hourly.append({"hour": om_hour, "ghi": round(wx["ghi"], 1) if wx else None, "kwh": kwh})
        kwhs = [h["kwh"] for h in hourly if h["kwh"] is not None]
        peak = max(kwhs) if kwhs else None
        summary = {
            "peak_kwh": peak,
            "peak_hour": next((h["hour"] for h in hourly if h["kwh"] == peak), None) if peak else None,
            "total_kwh": round(sum(kwhs), 1) if kwhs else None,
        }
        if fmt == "csv":
            return _solar_to_csv(hourly)
        return {"date": target_date, "plant": plant, "type": "historical", "hourly": hourly, "summary": summary}

    use_forecast_api = dt >= date.today() - timedelta(days=7)
    try:
        weather = await fetch_solar_weather(target_date, target_date, cfg["lat"], cfg["lon"],
                                            use_forecast_api=use_forecast_api)
        day_wx = weather.get(target_date, {})
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Solar weather API error: {exc}")

    if not day_wx:
        raise HTTPException(status_code=404, detail=f"No solar weather data for {target_date}.")

    mdl = state["model"]
    hourly = []
    for om_hour in range(24):
        hr_1_24 = om_hour + 1
        wx = day_wx.get(om_hour)
        if wx is None or wx["ghi"] <= 10:
            hourly.append({"hour": om_hour, "ghi": round(wx["ghi"], 1) if wx else None, "kwh": 0.0})
            continue
        pred = float(max(mdl.predict(build_solar_features(hr_1_24, wx["ghi"], wx["temp_f"], target_date))[0], 0))
        hourly.append({"hour": om_hour, "ghi": round(wx["ghi"], 1), "kwh": round(pred, 1)})

    kwhs = [h["kwh"] for h in hourly]
    peak = max(kwhs) if kwhs else None
    summary = {
        "peak_kwh": peak,
        "peak_hour": next((h["hour"] for h in hourly if h["kwh"] == peak), None),
        "total_kwh": round(sum(kwhs), 1),
    }
    if fmt == "csv":
        return _solar_to_csv(hourly)
    return {"date": target_date, "plant": plant, "type": "forecast", "hourly": hourly, "summary": summary}


@app.get("/api/wind/{plant}/status")
async def wind_plant_status(plant: str):
    cfg, state = _get_wind_or_404(plant)
    if state["model"] is None and not state["training"] and state["error"] is None:
        asyncio.create_task(train_wind_model(plant))
    return {
        "plant":        plant,
        "label":        cfg["label"],
        "ready":        state["model"] is not None,
        "training":     state["training"],
        "error":        state["error"],
        "trained_at":   state["trained_at"],
        "record_count": state["record_count"],
        "date_range":   state["date_range"],
        "r2":           state["r2"],
    }


@app.post("/api/wind/{plant}/train")
async def wind_plant_train(plant: str, background_tasks: BackgroundTasks):
    cfg, state = _get_wind_or_404(plant)
    if state["training"]:
        raise HTTPException(status_code=409, detail=f"{cfg['label']} model is already training.")
    background_tasks.add_task(train_wind_model, plant)
    return {"message": f"{cfg['label']} wind model training started."}


@app.get("/api/wind/{plant}/forecast")
async def wind_plant_forecast(
    plant: str,
    target_date: str = Query(..., alias="date", description="YYYY-MM-DD"),
    fmt: str = Query("json", alias="format"),
):
    cfg, state = _get_wind_or_404(plant)
    if state["model"] is None:
        if not state["training"]:
            asyncio.create_task(train_wind_model(plant))
        raise HTTPException(status_code=503, detail=f"{cfg['label']} model is training — please try again in about 30 seconds.")
    try:
        dt = datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    if dt > date.today() + timedelta(days=16):
        raise HTTPException(status_code=400, detail="Open-Meteo only forecasts up to 16 days ahead.")

    if target_date in state["history"]:
        day_hist = state["history"][target_date]
        try:
            weather = await fetch_wind_weather(target_date, target_date, cfg["lat"], cfg["lon"], use_forecast_api=False)
            day_wx = weather.get(target_date, {})
        except Exception:
            day_wx = {}
        hourly = []
        for om_hour in range(24):
            kwh = day_hist.get(om_hour, 0.0)
            wx  = day_wx.get(om_hour)
            hourly.append({"hour": om_hour, "wind_speed": round(wx["wind_speed"], 2) if wx else None, "kwh": kwh})
        kwhs = [h["kwh"] for h in hourly if h["kwh"] is not None]
        peak = max(kwhs) if kwhs else None
        summary = {
            "peak_kwh":   peak,
            "peak_hour":  next((h["hour"] for h in hourly if h["kwh"] == peak), None) if peak else None,
            "total_kwh":  round(sum(kwhs), 1) if kwhs else None,
        }
        if fmt == "csv":
            buf = io.StringIO()
            csv.writer(buf).writerows([["Hour", "Wind Speed (m/s)", "Generation (kWh)"]] +
                [[h["hour"], h["wind_speed"], h["kwh"]] for h in hourly])
            return Response(content=buf.getvalue(), media_type="text/csv")
        return {"date": target_date, "plant": plant, "type": "historical", "hourly": hourly, "summary": summary}

    use_forecast_api = dt >= date.today() - timedelta(days=7)
    try:
        weather = await fetch_wind_weather(target_date, target_date, cfg["lat"], cfg["lon"],
                                           use_forecast_api=use_forecast_api)
        day_wx = weather.get(target_date, {})
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Wind weather API error: {exc}")
    if not day_wx:
        raise HTTPException(status_code=404, detail=f"No wind weather data for {target_date}.")

    mdl = state["model"]
    hourly = []
    for om_hour in range(24):
        wx = day_wx.get(om_hour)
        if wx is None:
            hourly.append({"hour": om_hour, "wind_speed": None, "kwh": 0.0})
            continue
        pred = float(max(mdl.predict(build_wind_features(om_hour + 1, wx["wind_speed"], wx["wind_dir"], target_date))[0], 0))
        hourly.append({"hour": om_hour, "wind_speed": round(wx["wind_speed"], 2), "kwh": round(pred, 1)})

    kwhs = [h["kwh"] for h in hourly]
    peak = max(kwhs) if kwhs else None
    summary = {
        "peak_kwh":  peak,
        "peak_hour": next((h["hour"] for h in hourly if h["kwh"] == peak), None),
        "total_kwh": round(sum(kwhs), 1),
    }
    if fmt == "csv":
        buf = io.StringIO()
        csv.writer(buf).writerows([["Hour", "Wind Speed (m/s)", "Generation (kWh)"]] +
            [[h["hour"], h["wind_speed"], h["kwh"]] for h in hourly])
        return Response(content=buf.getvalue(), media_type="text/csv")
    return {"date": target_date, "plant": plant, "type": "forecast", "hourly": hourly, "summary": summary}


async def _safe_fetch_load_wx(target_date: str, dt: date) -> dict:
    """Returns {om_hour: {temp_f, apparent_f}} for load forecast, or {} on any error."""
    if model_state["model"] is None:
        return {}
    try:
        use_fc = dt >= date.today() - timedelta(days=7)
        wx = await fetch_weather(target_date, target_date, use_forecast_api=use_fc)
        return wx.get(target_date, {})
    except Exception:
        return {}


_realtime_cache: dict = {"expires": 0.0, "data": {}}


def _next_15_past() -> float:
    """Returns monotonic time when realtime cache should next expire (:15 past the next hour, Mountain Time)."""
    tz = ZoneInfo("America/Denver")
    now = datetime.now(tz)
    if now.minute < 15:
        target = now.replace(minute=15, second=0, microsecond=0)
    else:
        target = (now + timedelta(hours=1)).replace(minute=15, second=0, microsecond=0)
    return time.monotonic() + (target - now).total_seconds()


_REALTIME_JSON_URL = "https://raw.githubusercontent.com/crobinson111/lehi-load-forecast/master/data/realtime_load.json"


@app.get("/api/realtime_load")
async def api_realtime_load():
    if time.monotonic() < _realtime_cache["expires"]:
        return _realtime_cache["data"]
    tz = ZoneInfo("America/Denver")
    today = datetime.now(tz).strftime("%Y-%m-%d")
    data: dict = {}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(_REALTIME_JSON_URL)
            r.raise_for_status()
            payload = r.json()
        if payload.get("date") == today:
            data = {int(k): v for k, v in payload.get("hours", {}).items()}
    except Exception as exc:
        logger.warning(f"realtime_load: failed to fetch JSON: {exc}")
        data = _realtime_cache["data"]
    _realtime_cache.update({"expires": _next_15_past(), "data": data})
    return data


@app.get("/api/supply")
async def supply_portfolio(
    target_date: str = Query(..., alias="date", description="YYYY-MM-DD"),
):
    try:
        dt = datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    if dt > date.today() + timedelta(days=16):
        raise HTTPException(status_code=400, detail="Open-Meteo only forecasts up to 16 days ahead.")

    is_historical = target_date in supply_history

    weather_timed_out = False
    try:
        red_mesa_kwh, steele_a_kwh, horse_butte_kwh = await asyncio.wait_for(
            asyncio.gather(
                _plant_hourly_kwh("red-mesa",        target_date, dt),
                _plant_hourly_kwh("steele-a",        target_date, dt),
                _wind_plant_hourly_kwh("horse-butte", target_date, dt),
            ),
            timeout=40.0,
        )
    except asyncio.TimeoutError:
        red_mesa_kwh = steele_a_kwh = horse_butte_kwh = {}
        weather_timed_out = True

    # Load weather fetched separately so it doesn't compete with plant fetches
    day_wx_load: dict = {}
    if not weather_timed_out:
        try:
            day_wx_load = await asyncio.wait_for(
                _safe_fetch_load_wx(target_date, dt),
                timeout=25.0,
            )
        except asyncio.TimeoutError:
            weather_timed_out = True

    # Load — actuals if in history, otherwise model forecast
    load_by_hour: dict = {}
    if target_date in model_state["history"]:
        load_by_hour = model_state["history"][target_date]
    elif model_state["model"] is not None:
        mdl = model_state["model"]
        for om_hour in range(24):
            w = day_wx_load.get(om_hour)
            if w:
                pred = float(max(mdl.predict(build_features(om_hour + 1, w["temp_f"], w["apparent_f"], target_date))[0], 0))
                load_by_hour[om_hour] = round(pred, 1)

    # Overlay realtime actual meter data when available
    _rt_path = os.path.join(PERSIST_DIR, "realtime_load.json")
    if not os.path.exists(_rt_path):
        _rt_path = os.path.join(_BASE_DIR, "data", "realtime_load.json")
    try:
        with open(_rt_path) as _rt_f:
            _rt = json.load(_rt_f)
        if _rt.get("date") == target_date:
            for _hr_s, _kw_v in _rt.get("hours", {}).items():
                _om = int(_hr_s) - 1
                if 0 <= _om <= 23 and _kw_v:
                    load_by_hour[_om] = float(_kw_v)
    except Exception:
        pass

    supply_day = supply_history.get(target_date, {})
    uamps_day  = await _uamps_get_day(dt)

    # Power purchases for this date's month
    supply_purchases: list = []
    _pur_path_s = os.path.join(_BASE_DIR, "data", "power_purchases.csv")
    if os.path.exists(_pur_path_s):
        try:
            _pur_df_s = pd.read_csv(_pur_path_s)
            _pur_df_mo = _pur_df_s[
                (_pur_df_s["year"].astype(int) == dt.year) &
                (_pur_df_s["month"].astype(int) == dt.month)
            ]
            _sp_hols    = _wecc_holidays(dt.year)
            _sp_is_sun  = dt.weekday() == 6
            _sp_is_hol  = dt.isoformat() in _sp_hols
            _sp_all_day = _sp_is_sun or _sp_is_hol
            for _, _pr in _pur_df_mo.iterrows():
                _lbl   = str(_pr["label"]).strip()
                _kw    = float(_pr["mw"]) * 1000.0
                _sched = str(_pr["schedule"]).strip().lower()
                _key   = "pur_" + _lbl.lower().replace(" ", "_")
                _kw_hr: dict = {}
                for _om in range(24):
                    if _sched == "atc":
                        _kw_hr[_om] = _kw
                    elif _sched == "llh":
                        _is_llh = (_om <= 6) or (_om == 23)
                        _kw_hr[_om] = _kw if (_sp_all_day or (not _sp_is_sun and not _sp_is_hol and _is_llh)) else 0.0
                    elif _sched == "hlh":
                        _kw_hr[_om] = _kw if (not _sp_all_day and 7 <= _om <= 22) else 0.0
                    elif _sched == "hlh_ex_sp":
                        _kw_hr[_om] = _kw if (not _sp_all_day and ((7 <= _om <= 12) or (21 <= _om <= 22))) else 0.0
                    elif _sched == "sp":
                        _kw_hr[_om] = _kw if (not _sp_all_day and 13 <= _om <= 20) else 0.0
                    else:
                        _kw_hr[_om] = 0.0
                supply_purchases.append({
                    "label": _lbl, "key": _key, "mw": float(_pr["mw"]),
                    "schedule": _sched, "kw_by_hour": _kw_hr,
                })
        except Exception as _exc:
            logger.warning(f"Supply: purchases load failed: {_exc}")

    hourly = []
    for om_hour in range(24):
        supply = supply_day.get(om_hour, {})
        rm  = round(red_mesa_kwh.get(om_hour, 0.0), 1)
        sa  = round(steele_a_kwh.get(om_hour, 0.0), 1)
        if is_historical:
            nb  = round(supply.get("nebo",    0.0), 1)
            hb  = round(supply.get("h_butte", 0.0), 1)
            px  = round(supply.get("px",      0.0), 1)
            os_ = round(supply.get("os",      0.0), 1)
        else:
            nb  = round(_nebo_scheduled(), 1)
            hb  = round(horse_butte_kwh.get(om_hour, 0.0), 1)
            px  = round(_px_scheduled(om_hour, dt), 1)
            os_ = round(_os_scheduled(om_hour, dt), 1)
        if uamps_day:
            uhr       = uamps_day.get(om_hour + 1, {})
            crsp      = round(float(uhr.get("crsp",      0)), 1)
            provo_riv = round(float(uhr.get("provo_riv", 0)), 1)
            veyo      = round(float(uhr.get("veyo",      0)), 1)
        else:
            crsp = provo_riv = veyo = 0.0
        pur_kw = sum(p["kw_by_hour"][om_hour] for p in supply_purchases)
        total = round(rm + sa + nb + hb + px + os_ + crsp + provo_riv + veyo + pur_kw, 1)
        h_entry = {
            "hour":      om_hour,
            "red_mesa":  rm,
            "steele_a":  sa,
            "nebo":      nb,
            "h_butte":   hb,
            "px":        px,
            "os":        os_,
            "crsp":      crsp,
            "provo_riv": provo_riv,
            "veyo":      veyo,
            "total":     total,
            "load":      load_by_hour.get(om_hour),
        }
        for p in supply_purchases:
            h_entry[p["key"]] = p["kw_by_hour"][om_hour]
        hourly.append(h_entry)

    warnings = [
        f"{SOLAR_PLANTS[k]['label']} model not ready"
        for k in ("red-mesa", "steele-a")
        if solar_states[k]["model"] is None
    ]
    if wind_states["horse-butte"]["model"] is None:
        warnings.append("Horse Butte model not ready")
    if weather_timed_out:
        warnings.append("Weather API timed out — solar/wind/load forecast unavailable. Try refreshing.")

    day_total = round(sum(h["total"] for h in hourly), 1)
    return {
        "date": target_date,
        "type": "historical" if is_historical else "forecast",
        "hourly": hourly,
        "day_total": day_total,
        "solar_warnings": warnings,
        "uamps_available": uamps_day is not None,
        "purchases": [
            {"label": p["label"], "key": p["key"], "mw": p["mw"], "schedule": p["schedule"]}
            for p in supply_purchases
        ],
    }


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api/uamps/test")
async def uamps_test():
    """Diagnose UAMPS data sources — returns pretty-printed JSON."""
    today_str = date.today().isoformat()
    csv_url   = (os.getenv("UAMPS_SCHEDULE_CSV_URL") or "").strip()
    result = {
        "csv_url_set":       bool(csv_url),
        "csv_url":           csv_url or "(not set)",
        "csv_dates_loaded":  len(uamps_schedule),
        "csv_has_today":     today_str in uamps_schedule,
        "csv_date_range":    [min(uamps_schedule), max(uamps_schedule)] if uamps_schedule else None,
        "csv_sample_today":  {str(k): v for k, v in list((uamps_schedule.get(today_str) or {}).items())[:3]},
        "live_creds_set":    bool(os.getenv("UAMPS_USER_ID")) and bool(os.getenv("UAMPS_PASSWORD")),
    }
    return Response(content=json.dumps(result, indent=2), media_type="application/json")


@app.get("/api/dam-lmp")
async def dam_lmp_api(
    target_date: str = Query(..., alias="date", description="YYYY-MM-DD"),
):
    """Fetch DAM LMP from CAISO OASIS for ELAP_PACE-APND."""
    try:
        dt = datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format, use YYYY-MM-DD")

    next_dt = dt + timedelta(days=1)

    def to_utc_str(d: datetime) -> str:
        aware = d.replace(tzinfo=PACIFIC_TZ)
        return aware.astimezone(timezone.utc).strftime("%Y%m%dT%H:%M-0000")

    params = {
        "queryname": "PRC_LMP",
        "market_run_id": "DAM",
        "startdatetime": to_utc_str(dt),
        "enddatetime": to_utc_str(next_dt),
        "version": 1,
        "node": CAISO_NODE,
        "resultformat": 6,
    }
    caiso_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/zip, application/octet-stream, */*",
    }

    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.get(
            "https://oasis.caiso.com/oasisapi/SingleZip",
            params=params, headers=caiso_headers,
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"CAISO returned HTTP {resp.status_code}")
    if resp.content[:1] == b"<":
        snippet = resp.content[:400].decode("utf-8", errors="replace")
        raise HTTPException(status_code=502, detail=f"CAISO error response: {snippet}")

    try:
        rows = []
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            for name in z.namelist():
                if name.endswith(".csv"):
                    with z.open(name) as f:
                        reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                        rows.extend(list(reader))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=502, detail="CAISO response was not a valid ZIP file")

    lmp_rows = [r for r in rows if r.get("LMP_TYPE") == "LMP"]
    if not lmp_rows:
        raise HTTPException(status_code=404, detail=f"No DAM LMP data for {target_date} — may not be published yet")

    result = []
    for row in lmp_rows:
        interval_start = row.get("INTERVALSTARTTIME_GMT") or row.get("INTERVAL_START_GMT") or ""
        if not interval_start:
            continue
        try:
            dt_utc = datetime.strptime(interval_start[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            dt_pt = dt_utc.astimezone(PACIFIC_TZ)
            result.append({"hour": dt_pt.hour, "lmp": round(float(row.get("MW", 0)), 4)})
        except Exception:
            continue

    result.sort(key=lambda x: x["hour"])
    return {"date": target_date, "node": CAISO_NODE, "hourly": result}


@app.get("/gen_schedule_upload.png")
async def serve_gen_image():
    path = os.path.join(PERSIST_DIR, "gen_schedule_upload.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No image uploaded yet")
    return FileResponse(path, media_type="image/png")


@app.post("/api/gen-image")
async def save_gen_image(payload: dict):
    import base64 as _b64
    required_pin = os.environ.get("GEN_SCH_PIN", "").strip()
    if required_pin:
        submitted_pin = (payload.get("pin") or "").strip()
        if submitted_pin != required_pin:
            raise HTTPException(status_code=403, detail="Wrong PIN")
    image_data = payload.get("image", "")
    name = (payload.get("name") or "Unknown").strip()[:80]
    if not image_data:
        raise HTTPException(status_code=400, detail="No image provided")
    _, _, b64 = image_data.partition(",")
    img_bytes = _b64.b64decode(b64)
    with open(os.path.join(PERSIST_DIR, "gen_schedule_upload.png"), "wb") as f:
        f.write(img_bytes)
    tz = ZoneInfo("America/Denver")
    now = datetime.now(tz)
    meta = {"name": name, "timestamp": now.isoformat()}
    with open(os.path.join(PERSIST_DIR, "gen_schedule_meta.json"), "w") as f:
        json.dump(meta, f)
    return {"ok": True}


@app.get("/api/gen-image-meta")
async def get_gen_image_meta():
    path = os.path.join(PERSIST_DIR, "gen_schedule_meta.json")
    if not os.path.exists(path):
        return {"has_image": False}
    with open(path) as f:
        meta = json.load(f)
    meta["has_image"] = True
    return meta


@app.get("/api/availability")
async def get_availability():
    path = os.path.join(PERSIST_DIR, "availability.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


@app.post("/api/availability")
async def save_availability(payload: dict):
    name = (payload.pop("_name", None) or "Unknown").strip()[:80]
    now = datetime.now(ZoneInfo("America/Denver"))
    payload["_meta"] = {"name": name, "timestamp": now.isoformat()}
    with open(os.path.join(PERSIST_DIR, "availability.json"), "w") as f:
        json.dump(payload, f)
    return {"ok": True, "timestamp": now.isoformat(), "name": name}


@app.get("/api/history_week")
async def history_week(
    end_date: str = Query(default=None, description="YYYY-MM-DD, defaults to yesterday"),
):
    """7 days of hourly actual load vs supply, ending at end_date (default: yesterday)."""
    tz       = ZoneInfo("America/Denver")
    yesterday = (datetime.now(tz) - timedelta(days=1)).date()

    if end_date:
        try:
            ed = date.fromisoformat(end_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
        if ed > yesterday:
            ed = yesterday
    else:
        ed = yesterday

    sd = ed - timedelta(days=6)

    # UAMPS historical data — keyed {date_str: {om_hour: kw}}
    uamps_hist: dict = {}
    try:
        u_path = os.path.join(_BASE_DIR, "data", "uamps_schedule.csv")
        if os.path.exists(u_path):
            u_df = pd.read_csv(u_path)
            _sub = [c for c in ["crsp","hunter","provo_riv","mr","pv_wind","veyo","olmsted"]
                    if c in u_df.columns]
            if _sub:
                u_df["_uamps"] = u_df[_sub].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
                for _, row in u_df.iterrows():
                    ds = str(row["date"])[:10]
                    if sd.isoformat() <= ds <= ed.isoformat():
                        om = int(row["hr"]) - 1
                        uamps_hist.setdefault(ds, {})[om] = round(float(row["_uamps"]))
    except Exception:
        pass

    # Solar historical data — keyed {date_str: {om_hour: kw}}
    def _load_solar_hist(filename: str) -> dict:
        out: dict = {}
        try:
            p = os.path.join(_BASE_DIR, "data", filename)
            if not os.path.exists(p):
                return out
            df = pd.read_csv(p)
            df["date"] = df["date"].astype(str)
            df["kwh"]  = pd.to_numeric(df["kwh"], errors="coerce").fillna(0)
            for _, row in df.iterrows():
                ds = str(row["date"])[:10]
                if sd.isoformat() <= ds <= ed.isoformat():
                    om = int(row["hr"]) - 1
                    out.setdefault(ds, {})[om] = round(float(row["kwh"]))
        except Exception:
            pass
        return out

    rm_hist  = _load_solar_hist("red_mesa_history.csv")
    sa_hist  = _load_solar_hist("steele_a_history.csv")

    hist  = model_state["history"]
    dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    points: list = []
    d = sd
    while d <= ed:
        ds         = d.isoformat()
        day_load   = hist.get(ds, {})
        day_supply = supply_history.get(ds, {})
        day_uamps  = uamps_hist.get(ds, {})
        day_rm     = rm_hist.get(ds, {})
        day_sa     = sa_hist.get(ds, {})
        for om in range(24):
            load_kw  = day_load.get(om)
            sup      = day_supply.get(om, {})
            nebo     = int(sup.get("nebo",    0) or 0)
            h_butte  = int(sup.get("h_butte", 0) or 0)
            px       = int(sup.get("px",      0) or 0)
            os_      = int(sup.get("os",      0) or 0)
            uamps_kw = day_uamps.get(om, 0)
            rm_kw    = day_rm.get(om,    0)
            sa_kw    = day_sa.get(om,    0)
            total    = nebo + h_butte + px + os_ + uamps_kw + rm_kw + sa_kw
            points.append({
                "date":    ds,
                "dow":     dow_names[d.weekday()],
                "hour":    om,
                "load":    round(load_kw) if load_kw is not None else None,
                "nebo":    nebo,
                "h_butte": h_butte,
                "px":      px,
                "os":      os_,
                "uamps":   uamps_kw,
                "red_mesa": rm_kw,
                "steele_a": sa_kw,
                "total_supply": total if total > 0 else None,
            })
        d += timedelta(days=1)

    return {"start_date": sd.isoformat(), "end_date": ed.isoformat(), "points": points}


@app.get("/api/longterm")
async def longterm_forecast(
    month: int = Query(..., ge=1, le=12),
    year:  int = Query(..., ge=2020, le=2040),
):
    """Average hourly load + supply breakdown for a future month, using historical averages."""
    import calendar as _cal

    mdl = model_state["model"]
    if mdl is None:
        raise HTTPException(status_code=503, detail="Model not ready — please wait or retrain.")

    month_str = f"-{month:02d}-"

    # ── 1. Historical avg temperatures for this calendar month ───────────
    weather_data: dict = _load_weather_csv("LOAD_WEATHER_CSV_URL")
    if not weather_data:
        wx_path = os.path.join(_BASE_DIR, "data", "load_weather.csv")
        if os.path.exists(wx_path):
            df_wx = pd.read_csv(wx_path)
            for _, row in df_wx.iterrows():
                om_hour = int(row["hr"]) - 1
                weather_data.setdefault(str(row["date"]), {})[om_hour] = {
                    "temp_f": float(row["temp_f"]),
                    "apparent_f": float(row["apparent_f"]),
                }

    # Group by (day-of-month, hour) for per-day historical averages
    temp_by_dom: dict = {}   # {dom: {om_hour: {temp_f:[], apparent_f:[]}}}
    temp_monthly: dict = {}  # {om_hour: {temp_f:[], apparent_f:[]}} — monthly fallback
    for d_str, hours in weather_data.items():
        if month_str not in d_str:
            continue
        try:
            dom = int(d_str[8:10])
        except (ValueError, IndexError):
            continue
        for om_hour, wx in hours.items():
            temp_by_dom.setdefault(dom, {}).setdefault(om_hour, {"temp_f": [], "apparent_f": []})["temp_f"].append(wx["temp_f"])
            temp_by_dom[dom][om_hour]["apparent_f"].append(wx["apparent_f"])
            temp_monthly.setdefault(om_hour, {"temp_f": [], "apparent_f": []})["temp_f"].append(wx["temp_f"])
            temp_monthly[om_hour]["apparent_f"].append(wx["apparent_f"])

    if not temp_monthly:
        raise HTTPException(status_code=404, detail=f"No historical weather data for month {month}. Run this for a month that appears in past years.")

    monthly_avg_temps = {
        om: {"temp_f": sum(v["temp_f"]) / len(v["temp_f"]),
             "apparent_f": sum(v["apparent_f"]) / len(v["apparent_f"])}
        for om, v in temp_monthly.items()
    }
    dom_avg_temps = {
        dom: {om: {"temp_f": sum(v["temp_f"]) / len(v["temp_f"]),
                   "apparent_f": sum(v["apparent_f"]) / len(v["apparent_f"])}
              for om, v in hrs.items()}
        for dom, hrs in temp_by_dom.items()
    }

    # ── 1b. Year-over-year growth factor ─────────────────────────────────────
    # Collect prior-year same-month actuals (used as forecast base when available).
    # Compute YoY growth by comparing same calendar day + same hour across years.
    # Start with the same calendar month; step back one month at a time until we
    # have at least 14 matched days so weather anomalies don't dominate the ratio.
    history = model_state["history"]
    prior_year = year - 1

    prior_month_daily: dict = {}   # {dom: {om_hour: load}} for prior year same month
    prior_month_str_exact = f"{prior_year}-{month:02d}-"
    for d_str, day_data in history.items():
        if not d_str.startswith(prior_month_str_exact):
            continue
        try:
            dom = int(d_str[8:10])
        except (ValueError, IndexError):
            continue
        prior_month_daily[dom] = day_data

    yoy_num = 0.0    # sum of current-year actuals for matched day+hour pairs
    yoy_den = 0.0    # sum of prior-year actuals for the same pairs
    yoy_days = 0
    for mo_offset in range(0, 6):
        chk_mo = month - mo_offset
        chk_yr = year
        while chk_mo <= 0:
            chk_mo += 12
            chk_yr -= 1
        prior_chk_yr = chk_yr - 1
        chk_mo_str = f"-{chk_mo:02d}-"
        for d_str, curr_day in history.items():
            if not d_str.startswith(str(chk_yr)):
                continue
            if chk_mo_str not in d_str:
                continue
            try:
                dom = int(d_str[8:10])
            except (ValueError, IndexError):
                continue
            prior_d = f"{prior_chk_yr}-{chk_mo:02d}-{dom:02d}"
            if prior_d not in history:
                continue
            prior_day = history[prior_d]
            for oh in range(24):
                c = curr_day.get(oh)
                p = prior_day.get(oh)
                if c is not None and p is not None and p > 0:
                    yoy_num += c
                    yoy_den += p
            yoy_days += 1
        if yoy_days >= 14:
            break

    if yoy_den > 0 and yoy_days >= 7:
        yoy_growth: float | None = round(max(0.85, min(1.35, yoy_num / yoy_den)), 4)
    else:
        yoy_growth = None

    # ── 2. Predict load for each day × hour ──────────────────────────────
    _, days_in_month = _cal.monthrange(year, month)
    month_days = [date(year, month, d) for d in range(1, days_in_month + 1)]

    day_hour_loads: list = []
    hour_loads: dict = {om: [] for om in range(24)}

    if prior_month_daily and yoy_growth is not None:
        # Primary path: scale prior-year same-month actuals by the YoY growth
        # factor. Preserves the real seasonal load shape; only stretches magnitude.
        prior_vals = list(prior_month_daily.values())
        for day in month_days:
            base = prior_month_daily.get(day.day)
            if base is None:
                base = {oh: sum(v.get(oh, 0.0) for v in prior_vals) / len(prior_vals)
                        for oh in range(24)}
            d_loads = {oh: v * yoy_growth for oh, v in base.items() if v is not None}
            day_hour_loads.append(d_loads)
            for oh, v in d_loads.items():
                hour_loads[oh].append(v)
    else:
        # Fallback: model-based prediction using averaged historical temperatures
        for day in month_days:
            date_str = day.isoformat()
            dom_temps = dom_avg_temps.get(day.day, monthly_avg_temps)
            d_loads: dict = {}
            for om_hour in range(24):
                wx = dom_temps.get(om_hour) or monthly_avg_temps.get(om_hour)
                if wx is None:
                    continue
                pred = float(max(mdl.predict(
                    build_features(om_hour + 1, wx["temp_f"], wx["apparent_f"], date_str)
                )[0], 0))
                d_loads[om_hour] = pred
                hour_loads[om_hour].append(pred)
            day_hour_loads.append(d_loads)

    # ── 3. Historical avg supply for this calendar month ─────────────────
    def _avg_by_hr(df: pd.DataFrame, val_col: str) -> dict:
        """Returns {om_hour: avg_value} for rows matching the target month."""
        sub = df[df["date"].astype(str).str.contains(month_str)].copy()
        sub["om_hour"] = sub["hr"].astype(int) - 1
        sub[val_col] = pd.to_numeric(sub[val_col], errors="coerce").fillna(0)
        return sub.groupby("om_hour")[val_col].mean().to_dict()

    # Base supply (nebo, h_butte, px, os)
    sup_url = os.environ.get("SUPPLY_HISTORY_CSV_URL")
    if sup_url:
        import io as _io
        sup_df = pd.read_csv(_io.StringIO(httpx.get(sup_url, timeout=30).text))
    else:
        sup_df = pd.read_csv(os.path.join(_BASE_DIR, "data", "supply_history.csv"))
    for col in ["nebo", "h_butte", "px", "os"]:
        sup_df[col] = pd.to_numeric(sup_df[col], errors="coerce").fillna(0)
    # Nebo and Horse Butte are weather/dispatch dependent — use historical avg
    avg_nebo    = _avg_by_hr(sup_df, "nebo")
    avg_h_butte = _avg_by_hr(sup_df, "h_butte")
    # PX and OS: check future_schedule.csv for locked-in contract values first.
    # Fall back to most-recent same-calendar-month historical data if not found.
    future_sched_path = os.path.join(_BASE_DIR, "data", "future_schedule.csv")
    avg_px: dict = {}
    avg_os: dict = {}
    if os.path.exists(future_sched_path):
        fs = pd.read_csv(future_sched_path)
        fs.columns = fs.columns.str.strip().str.lower()
        fs_match = fs[(fs["year"].astype(int) == year) & (fs["month"].astype(int) == month)]
        # Only use if at least one row has a non-empty PX or OS value
        has_values = fs_match[["px","os"]].apply(pd.to_numeric, errors="coerce").notna().any(axis=None)
        if not fs_match.empty and has_values:
            mon_sat_count = sum(1 for d in month_days if d.weekday() != 6)
            mon_sat_scale = mon_sat_count / days_in_month
            px_by_hour = {om: 0.0 for om in range(24)}
            os_by_hour = {om: 0.0 for om in range(24)}
            for _, row in fs_match.iterrows():
                hr_str = str(row["hr"]).strip()
                # Parse range "8-23" or single "14"
                if "-" in hr_str:
                    parts = hr_str.split("-")
                    hr_start, hr_end = int(parts[0]), int(parts[1])
                else:
                    hr_start = hr_end = int(hr_str)
                # Values are in MW — convert to kW; blank = 0
                px_mw = pd.to_numeric(row["px"], errors="coerce")
                os_mw = pd.to_numeric(row["os"], errors="coerce")
                px_kw = float(px_mw) * 1000 if pd.notna(px_mw) else 0.0
                os_kw = float(os_mw) * 1000 if pd.notna(os_mw) else 0.0
                for hr in range(hr_start, hr_end + 1):
                    om = hr - 1
                    if 0 <= om <= 23:
                        px_by_hour[om] += px_kw * mon_sat_scale
                        os_by_hour[om] += os_kw * mon_sat_scale
            avg_px = px_by_hour
            avg_os = os_by_hour
        # Nebo: override historical average with future_schedule value if provided
        if "nebo" in fs.columns:
            nebo_has = fs_match["nebo"].apply(pd.to_numeric, errors="coerce").notna().any()
            if not fs_match.empty and nebo_has:
                nebo_by_hour = {om: avg_nebo.get(om, 0) for om in range(24)}
                for _, row in fs_match.iterrows():
                    hr_str = str(row["hr"]).strip()
                    if "-" in hr_str:
                        parts = hr_str.split("-")
                        hr_start, hr_end = int(parts[0]), int(parts[1])
                    else:
                        hr_start = hr_end = int(hr_str)
                    nebo_mw = pd.to_numeric(row["nebo"], errors="coerce")
                    if pd.notna(nebo_mw):
                        nebo_kw = float(nebo_mw) * 1000
                        for hr in range(hr_start, hr_end + 1):
                            om = hr - 1
                            if 0 <= om <= 23:
                                nebo_by_hour[om] = nebo_kw
                avg_nebo = nebo_by_hour
    if not avg_px:
        same_month_dates = sup_df[sup_df["date"].astype(str).str.contains(month_str)]["date"].astype(str)
        ref_date = same_month_dates.max() if not same_month_dates.empty else sup_df["date"].astype(str).max()
        recent_rows = sup_df[sup_df["date"].astype(str) == ref_date].copy()
        recent_rows["om_hour"] = recent_rows["hr"].astype(int) - 1
        avg_px = recent_rows.set_index("om_hour")["px"].to_dict()
        avg_os = recent_rows.set_index("om_hour")["os"].to_dict()

    # Red Mesa solar — load from local CSV (in repo), fall back to URL env var
    def _load_solar_csv(local_name: str, url_env: str) -> pd.DataFrame:
        local = os.path.join(_BASE_DIR, "data", local_name)
        if os.path.exists(local):
            df = pd.read_csv(local)
        else:
            url = os.environ.get(url_env, "")
            if not url:
                return pd.DataFrame(columns=["date", "hr", "kwh"])
            df = pd.read_csv(_io.StringIO(httpx.get(url, timeout=30).text))
        df["date"] = df["date"].astype(str)
        df["hr"]   = df["hr"].astype(int)
        df["kwh"]  = pd.to_numeric(df["kwh"], errors="coerce").fillna(0)
        return df[["date", "hr", "kwh"]]

    def _operational_only(df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows before the plant's first non-zero production date."""
        df = df.copy()
        df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce").fillna(0)
        active = df[df["kwh"] > 0]
        if active.empty:
            return df
        return df[df["date"].astype(str) >= active["date"].astype(str).min()]

    avg_rm = _avg_by_hr(_operational_only(_load_solar_csv("red_mesa_history.csv",  "RED_MESA_CSV_URL")),  "kwh")
    avg_sa = _avg_by_hr(_operational_only(_load_solar_csv("steele_a_history.csv",  "STEELE_A_CSV_URL")),  "kwh")

    # UAMPS (no historical Aug data — use all available, averaged by hour)
    uamps_url = os.environ.get("UAMPS_CSV_URL")
    if uamps_url:
        u_df = pd.read_csv(_io.StringIO(httpx.get(uamps_url, timeout=30).text))
    else:
        u_df = pd.read_csv(os.path.join(_BASE_DIR, "data", "uamps_schedule.csv"))
    u_df["om_hour"] = u_df["hr"].astype(int) - 1
    _uamps_sub_cols = ["crsp", "hunter", "provo_riv", "mr", "pv_wind", "veyo", "olmsted"]
    for col in _uamps_sub_cols:
        if col not in u_df.columns:
            u_df[col] = 0
        u_df[col] = pd.to_numeric(u_df[col], errors="coerce").fillna(0)
    u_df["uamps"] = u_df[_uamps_sub_cols].sum(axis=1)
    avg_uamps    = u_df.groupby("om_hour")["uamps"].mean().to_dict()
    avg_uamps_sub = {col: u_df.groupby("om_hour")[col].mean().to_dict() for col in _uamps_sub_cols}

    # ── 3b. Fixed power purchases for this month ──────────────────────────
    # Reads data/power_purchases.csv: label,year,month,mw,rate_per_mwh,schedule
    # schedule types: "atc" (all hours every day), "llh" (HE 24-7 Mon-Sat + all Sun/holidays)
    purchases: list = []
    pur_path = os.path.join(_BASE_DIR, "data", "power_purchases.csv")
    if os.path.exists(pur_path):
        try:
            pur_df = pd.read_csv(pur_path)
            pur_df_mo = pur_df[
                (pur_df["year"].astype(int) == year) &
                (pur_df["month"].astype(int) == month)
            ]
            holidays = _wecc_holidays(year)
            for _, pur_row in pur_df_mo.iterrows():
                lbl      = str(pur_row["label"]).strip()
                kw       = float(pur_row["mw"]) * 1000.0
                rate     = float(pur_row["rate_per_mwh"])
                sched    = str(pur_row["schedule"]).strip().lower()
                key      = "pur_" + lbl.lower().replace(" ", "_")
                kw_by_hour: dict = {}
                for om in range(24):
                    total_kw = 0.0
                    for day in month_days:
                        dow        = day.weekday()           # 0=Mon, 6=Sun
                        is_sun     = dow == 6
                        is_hol     = day.isoformat() in holidays
                        all_day    = is_sun or is_hol
                        if sched == "atc":
                            total_kw += kw
                        elif sched == "llh":
                            is_llh_hr = (om <= 6) or (om == 23)
                            if all_day or (not is_sun and not is_hol and is_llh_hr):
                                total_kw += kw
                        elif sched == "hlh":
                            # HE 8-23 = om 7-22, Mon-Sat non-holidays
                            if not all_day and 7 <= om <= 22:
                                total_kw += kw
                        elif sched == "hlh_ex_sp":
                            # HLH excluding super peak (HE 14-21 = om 13-20)
                            if not all_day and ((7 <= om <= 12) or (21 <= om <= 22)):
                                total_kw += kw
                        elif sched == "sp":
                            # Super peak: HE 14-21 = om 13-20, Mon-Sat non-holidays
                            if not all_day and 13 <= om <= 20:
                                total_kw += kw
                    kw_by_hour[om] = round(total_kw / days_in_month, 1)
                purchases.append({
                    "label": lbl, "key": key, "rate": rate,
                    "schedule": sched, "mw": float(pur_row["mw"]),
                    "kw_by_hour": kw_by_hour,
                })
        except Exception as _exc:
            logger.warning(f"Power purchases load failed: {_exc}")

    # ── 4. Build hourly averages for bar chart ────────────────────────────
    month_name = _cal.month_name[month]
    hourly = []
    for om in range(24):
        load     = round(sum(hour_loads[om]) / len(hour_loads[om])) if hour_loads[om] else 0
        nebo     = round(avg_nebo.get(om, 0))
        h_butte  = round(avg_h_butte.get(om, 0))
        px       = round(avg_px.get(om, 0))
        os_      = round(avg_os.get(om, 0))
        red_mesa = round(avg_rm.get(om, 0))
        steele_a = round(avg_sa.get(om, 0))
        uamps    = round(avg_uamps.get(om, 0))
        pur_kw   = sum(p["kw_by_hour"][om] for p in purchases)
        supply       = nebo + h_butte + px + os_ + red_mesa + steele_a + uamps + pur_kw
        raw_shortage = max(0, load - supply)
        internal_gen = min(raw_shortage, 21000)
        shortage     = raw_shortage - internal_gen
        h_entry = {
            "hour": om, "load": load,
            "nebo": nebo, "h_butte": h_butte, "px": px, "os": os_,
            "red_mesa": red_mesa, "steele_a": steele_a, "uamps": uamps,
            "internal_gen": internal_gen,
            "total_supply": supply + internal_gen, "shortage": shortage,
        }
        for p in purchases:
            h_entry[p["key"]] = p["kw_by_hour"][om]
        hourly.append(h_entry)

    # ── 5. Per-day shortage sums → average across month ───────────────────
    # Sum shortage per hour for each actual day (zeroed per hour, no cross-hour netting),
    # then average the daily totals. This is the operationally correct shortage metric.
    supply_by_hr = {
        om: (round(avg_nebo.get(om, 0)) + round(avg_h_butte.get(om, 0)) +
             round(avg_px.get(om, 0))   + round(avg_os.get(om, 0)) +
             round(avg_rm.get(om, 0))   + round(avg_sa.get(om, 0)) +
             round(avg_uamps.get(om, 0)) +
             sum(p["kw_by_hour"][om] for p in purchases))
        for om in range(24)
    }
    daily_shortages_kwh: list = []
    for d in day_hour_loads:
        day_total = 0.0
        for om in range(24):
            raw = max(0.0, d.get(om, 0.0) - supply_by_hr[om])
            day_total += raw - min(raw, 21000.0)
        daily_shortages_kwh.append(day_total)
    avg_daily_shortage_mwh = round(sum(daily_shortages_kwh) / len(daily_shortages_kwh) / 1000, 1) if daily_shortages_kwh else 0.0

    avg_load   = round(sum(h["load"]         for h in hourly) / 24)
    avg_supply = round(sum(h["total_supply"] for h in hourly) / 24)
    peak_h = max(hourly, key=lambda h: h["load"])

    _pie_keys = ["nebo", "h_butte", "px", "os", "red_mesa", "steele_a", "uamps", "internal_gen", "shortage"]
    _pie_keys += [p["key"] for p in purchases]
    pie = {k: round(sum(h[k] for h in hourly)) for k in _pie_keys}

    # ── 6. Blended resource cost ──────────────────────────────────────────
    blended_cost_per_mwh = None
    cost_pie:   dict = {}
    cost_rates: dict = {}
    costs_path = os.path.join(_BASE_DIR, "data", "resource_costs.csv")
    if os.path.exists(costs_path):
        try:
            c_df = pd.read_csv(costs_path)
            c_df["month"] = pd.to_numeric(c_df["month"], errors="coerce")

            def _res_cost(res: str) -> float:
                mo_row = c_df[(c_df["resource"] == res) & (c_df["month"] == month)]
                if not mo_row.empty:
                    return float(mo_row["cost_per_mwh"].iloc[0])
                any_row = c_df[(c_df["resource"] == res) & c_df["month"].isna()]
                if not any_row.empty:
                    return float(any_row["cost_per_mwh"].iloc[0])
                return 0.0

            # Step 1: total monthly MWh per resource
            # Use exactly the same data as the resource graphs (hourly array).
            resource_mwh: dict = {}

            # Non-UAMPS resources — directly from hourly (includes internal_gen)
            for res in ["nebo", "h_butte", "red_mesa", "steele_a", "px", "os",
                        "internal_gen", "shortage"]:
                monthly_kwh = sum(h.get(res, 0) for h in hourly) * days_in_month
                resource_mwh[res] = monthly_kwh / 1000.0

            # UAMPS sub-resources — proportionally split hourly["uamps"] (the
            # forecasted value used in the graph) by each sub-resource's share
            for col in _uamps_sub_cols:
                if col == "veyo":
                    resource_mwh["veyo"] = 0.0  # standby only — assume 0 MW
                    continue
                monthly_kwh = 0.0
                for om in range(24):
                    uamps_graph = hourly[om]["uamps"]
                    uamps_raw   = avg_uamps.get(om, 0)
                    sub_raw     = avg_uamps_sub[col].get(om, 0.0)
                    sub_kw      = uamps_graph * (sub_raw / uamps_raw) if uamps_raw > 0 else 0.0
                    monthly_kwh += sub_kw * days_in_month
                resource_mwh[col] = monthly_kwh / 1000.0

            # Power purchase MWh — keyed by purchase key, rated from CSV
            _pur_rates = {p["key"]: p["rate"] for p in purchases}
            for p in purchases:
                monthly_kwh = sum(h.get(p["key"], 0) for h in hourly) * days_in_month
                resource_mwh[p["key"]] = monthly_kwh / 1000.0

            def _get_rate(res: str) -> float:
                r = _pur_rates.get(res)
                return r if r is not None else _res_cost(res)

            # Step 2: cost per resource = MWh × $/MWh
            # Step 3: blended = total cost / total MWh (weighted average)
            total_cost = sum(mwh * _get_rate(res) for res, mwh in resource_mwh.items())
            total_mwh  = sum(resource_mwh.values())
            if total_mwh > 0:
                blended_cost_per_mwh = round(total_cost / total_mwh, 2)
            cost_pie   = {res: round(mwh * _get_rate(res))
                          for res, mwh in resource_mwh.items()
                          if round(mwh * _get_rate(res)) > 0}
            cost_rates = {res: _get_rate(res) for res in cost_pie}
        except Exception as _exc:
            logger.warning(f"Blended cost calculation failed: {_exc}")

    return {
        "month": month, "year": year,
        "label": f"{month_name} {year}",
        "hourly": hourly,
        "summary": {
            "avg_load": avg_load, "avg_supply": avg_supply,
            "avg_daily_shortage_mwh": avg_daily_shortage_mwh,
            "peak_load": peak_h["load"], "peak_hour": peak_h["hour"],
        },
        "pie": pie,
        "yoy_growth": yoy_growth,
        "yoy_days": yoy_days,
        "blended_cost_per_mwh": blended_cost_per_mwh,
        "cost_pie": cost_pie,
        "cost_rates": cost_rates,
        "purchases": [{"label": p["label"], "key": p["key"], "mw": p["mw"],
                        "rate": p["rate"], "schedule": p["schedule"]} for p in purchases],
    }


@app.get("/api/dayplan")
async def day_plan(
    target_date: str = Query(..., alias="date"),
):
    """Single-day load forecast with full hourly supply breakdown."""
    import calendar as _cal

    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not ready.")
    try:
        dt = datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    today = date.today()
    if dt > today + timedelta(days=16):
        raise HTTPException(status_code=400, detail="Date is beyond the 16-day forecast window.")

    month     = dt.month
    year      = dt.year
    month_str = f"-{month:02d}-"
    mdl       = model_state["model"]

    # ── 1. Hourly load ─────────────────────────────────────────────────────
    if target_date in model_state["history"]:
        day_hist = model_state["history"][target_date]
        try:
            wx_data = await fetch_weather(target_date, target_date, use_forecast_api=False)
            day_wx  = wx_data.get(target_date, {})
        except Exception:
            day_wx = {}
        hour_loads = {om: day_hist.get(om) for om in range(24)}
        hour_temps  = {om: day_wx.get(om, {}).get("temp_f") for om in range(24)}
        data_type   = "historical"
    else:
        use_fc = dt >= today - timedelta(days=7)
        try:
            wx_data = await fetch_weather(target_date, target_date, use_forecast_api=use_fc)
            day_wx  = wx_data.get(target_date, {})
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Weather API error: {exc}")
        if not day_wx:
            raise HTTPException(status_code=404, detail=f"No weather data for {target_date}.")
        hour_loads = {}
        hour_temps  = {}
        for om in range(24):
            wx = day_wx.get(om)
            if wx:
                pred = float(max(mdl.predict(build_features(om + 1, wx["temp_f"], wx["apparent_f"], target_date))[0], 0))
                hour_loads[om] = round(pred, 1)
                hour_temps[om]  = round(wx["temp_f"], 1)
            else:
                hour_loads[om] = None
                hour_temps[om]  = None
        data_type = "forecast"

    # ── 2. Supply (same logic as /api/longterm for this month/year) ────────
    import io as _io

    def _avg_by_hr(df: pd.DataFrame, val_col: str) -> dict:
        sub = df[df["date"].astype(str).str.contains(month_str)].copy()
        sub["om_hour"] = sub["hr"].astype(int) - 1
        sub[val_col] = pd.to_numeric(sub[val_col], errors="coerce").fillna(0)
        return sub.groupby("om_hour")[val_col].mean().to_dict()

    sup_url = os.environ.get("SUPPLY_HISTORY_CSV_URL")
    if sup_url:
        sup_df = pd.read_csv(_io.StringIO(httpx.get(sup_url, timeout=30).text))
    else:
        sup_df = pd.read_csv(os.path.join(_BASE_DIR, "data", "supply_history.csv"))
    for col in ["nebo", "h_butte", "px", "os"]:
        sup_df[col] = pd.to_numeric(sup_df[col], errors="coerce").fillna(0)

    avg_nebo    = _avg_by_hr(sup_df, "nebo")
    avg_h_butte = _avg_by_hr(sup_df, "h_butte")

    future_sched_path = os.path.join(_BASE_DIR, "data", "future_schedule.csv")
    avg_px: dict = {}
    avg_os: dict = {}
    if os.path.exists(future_sched_path):
        fs = pd.read_csv(future_sched_path)
        fs.columns = fs.columns.str.strip().str.lower()
        fs_match = fs[(fs["year"].astype(int) == year) & (fs["month"].astype(int) == month)]
        has_values = fs_match[["px","os"]].apply(pd.to_numeric, errors="coerce").notna().any(axis=None)
        if not fs_match.empty and has_values:
            is_sunday = dt.weekday() == 6
            px_by_hour = {om: 0.0 for om in range(24)}
            os_by_hour = {om: 0.0 for om in range(24)}
            if not is_sunday:
                for _, row in fs_match.iterrows():
                    hr_str = str(row["hr"]).strip()
                    s, e = (int(hr_str.split("-")[0]), int(hr_str.split("-")[1])) if "-" in hr_str else (int(hr_str), int(hr_str))
                    px_kw = float(pd.to_numeric(row["px"], errors="coerce") or 0) * 1000
                    os_kw = float(pd.to_numeric(row["os"], errors="coerce") or 0) * 1000
                    for hr in range(s, e + 1):
                        om = hr - 1
                        if 0 <= om <= 23:
                            px_by_hour[om] += px_kw
                            os_by_hour[om] += os_kw
            avg_px = px_by_hour
            avg_os = os_by_hour
        if "nebo" in fs.columns:
            nebo_has = fs_match["nebo"].apply(pd.to_numeric, errors="coerce").notna().any()
            if not fs_match.empty and nebo_has:
                nebo_by_hour = {om: avg_nebo.get(om, 0) for om in range(24)}
                for _, row in fs_match.iterrows():
                    hr_str = str(row["hr"]).strip()
                    s, e = (int(hr_str.split("-")[0]), int(hr_str.split("-")[1])) if "-" in hr_str else (int(hr_str), int(hr_str))
                    nebo_mw = pd.to_numeric(row["nebo"], errors="coerce")
                    if pd.notna(nebo_mw):
                        nebo_kw = float(nebo_mw) * 1000
                        for hr in range(s, e + 1):
                            om = hr - 1
                            if 0 <= om <= 23:
                                nebo_by_hour[om] = nebo_kw
                avg_nebo = nebo_by_hour
    if not avg_px:
        same = sup_df[sup_df["date"].astype(str).str.contains(month_str)]["date"].astype(str)
        ref  = same.max() if not same.empty else sup_df["date"].astype(str).max()
        rec  = sup_df[sup_df["date"].astype(str) == ref].copy()
        rec["om_hour"] = rec["hr"].astype(int) - 1
        avg_px = rec.set_index("om_hour")["px"].to_dict()
        avg_os = rec.set_index("om_hour")["os"].to_dict()

    def _load_solar_csv(local_name: str, url_env: str) -> pd.DataFrame:
        local = os.path.join(_BASE_DIR, "data", local_name)
        if os.path.exists(local):
            df = pd.read_csv(local)
        else:
            url = os.environ.get(url_env, "")
            if not url:
                return pd.DataFrame(columns=["date", "hr", "kwh"])
            df = pd.read_csv(_io.StringIO(httpx.get(url, timeout=30).text))
        df["date"] = df["date"].astype(str)
        df["hr"]   = df["hr"].astype(int)
        df["kwh"]  = pd.to_numeric(df["kwh"], errors="coerce").fillna(0)
        return df[["date", "hr", "kwh"]]

    def _operational_only(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce").fillna(0)
        active = df[df["kwh"] > 0]
        if active.empty:
            return df
        return df[df["date"].astype(str) >= active["date"].astype(str).min()]

    avg_rm = _avg_by_hr(_operational_only(_load_solar_csv("red_mesa_history.csv",  "RED_MESA_CSV_URL")),  "kwh")
    avg_sa = _avg_by_hr(_operational_only(_load_solar_csv("steele_a_history.csv",  "STEELE_A_CSV_URL")),  "kwh")

    uamps_url = os.environ.get("UAMPS_CSV_URL")
    if uamps_url:
        u_df = pd.read_csv(_io.StringIO(httpx.get(uamps_url, timeout=30).text))
    else:
        u_df = pd.read_csv(os.path.join(_BASE_DIR, "data", "uamps_schedule.csv"))
    u_df["om_hour"] = u_df["hr"].astype(int) - 1
    _uamps_sub_cols_lt = ["crsp", "hunter", "provo_riv", "mr", "pv_wind", "veyo", "olmsted"]
    for col in _uamps_sub_cols_lt:
        if col not in u_df.columns:
            u_df[col] = 0
        u_df[col] = pd.to_numeric(u_df[col], errors="coerce").fillna(0)
    u_df["uamps"] = u_df[_uamps_sub_cols_lt].sum(axis=1)
    avg_uamps = u_df.groupby("om_hour")["uamps"].mean().to_dict()

    # ── 3. Build hourly output ─────────────────────────────────────────────
    hourly = []
    daily_shortage_kw = 0.0
    for om in range(24):
        load     = hour_loads.get(om)
        temp     = hour_temps.get(om)
        nebo     = round(avg_nebo.get(om, 0))
        h_butte  = round(avg_h_butte.get(om, 0))
        px       = round(avg_px.get(om, 0))
        os_      = round(avg_os.get(om, 0))
        red_mesa = round(avg_rm.get(om, 0))
        steele_a = round(avg_sa.get(om, 0))
        uamps    = round(avg_uamps.get(om, 0))
        supply   = nebo + h_butte + px + os_ + red_mesa + steele_a + uamps
        if load is not None:
            raw      = max(0, load - supply)
            intgen   = min(raw, 21000)
            shortage = raw - intgen
            surplus  = max(0, supply + intgen - load)
            daily_shortage_kw += shortage
        else:
            intgen = shortage = surplus = 0
        hourly.append({
            "hour": om, "temp_f": temp, "load": round(load) if load is not None else None,
            "nebo": nebo, "h_butte": h_butte, "px": px, "os": os_,
            "red_mesa": red_mesa, "steele_a": steele_a, "uamps": uamps,
            "internal_gen": intgen,
            "total_supply": supply + intgen, "shortage": shortage, "surplus": surplus,
        })

    loads  = [h["load"] for h in hourly if h["load"] is not None]
    peak_h = max(hourly, key=lambda h: h["load"] or 0)
    label  = f"{_cal.day_name[dt.weekday()]}, {_cal.month_name[month]} {dt.day}, {year}"

    return {
        "date": target_date, "label": label, "type": data_type,
        "hourly": hourly,
        "summary": {
            "avg_load":             round(sum(loads) / len(loads)) if loads else 0,
            "peak_load":            peak_h["load"],
            "peak_hour":            peak_h["hour"],
            "daily_shortage_mwh":   round(daily_shortage_kw / 1000, 1),
        },
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")
