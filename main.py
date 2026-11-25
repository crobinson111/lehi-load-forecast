import csv
import calendar as _cal_mod
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
    url = (os.environ.get(url_env_key) or "").strip()
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
    h = set()
    # Fixed-date holidays; NERC rule: a Sunday holiday is observed the following Monday
    for _mo, _dy in ((1, 1), (7, 4), (12, 25)):
        _fd = _d(year, _mo, _dy)
        h.add((_fd + timedelta(days=1) if _fd.weekday() == 6 else _fd).isoformat())
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


def _nebo_scheduled(om_hour: int, dt: date) -> float:
    """Nebo (kW) for a future hour from data/future_schedule.csv (all days, the
    Hr range given). Falls back to the old fixed 10 MW if the month has no entry."""
    path = os.path.join(_BASE_DIR, "data", "future_schedule.csv")
    try:
        fs = pd.read_csv(path)
        fs.columns = fs.columns.str.strip().str.lower()
        rows = fs[(fs["year"].astype(int) == dt.year) & (fs["month"].astype(int) == dt.month)]
        hr = om_hour + 1
        found = False
        total_kw = 0.0
        for _, row in rows.iterrows():
            mw = pd.to_numeric(row.get("nebo"), errors="coerce")
            if pd.isna(mw):
                continue
            found = True
            hr_str = str(row["hr"]).strip()
            s, e = (int(x) for x in hr_str.split("-")) if "-" in hr_str else (int(hr_str), int(hr_str))
            if s <= hr <= e:
                total_kw = float(mw) * 1000
        if found:
            return total_kw
    except Exception as exc:
        logger.warning(f"_nebo_scheduled: could not read future_schedule.csv: {exc}")
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


def _contract_scheduled(col: str, om_hour: int, dt: date) -> float:
    """Contract volume (kW) for PX or OS from data/future_schedule.csv only.
    Mon-Sat, excluding WECC holidays. No contract entry for the month -> 0.
    Never falls back to historical data."""
    if dt in _WECC_HOLIDAYS or dt.weekday() == 6:
        return 0.0
    path = os.path.join(_BASE_DIR, "data", "future_schedule.csv")
    if not os.path.exists(path):
        return 0.0
    try:
        fs = pd.read_csv(path)
        fs.columns = fs.columns.str.strip().str.lower()
        rows = fs[(fs["year"].astype(int) == dt.year) & (fs["month"].astype(int) == dt.month)]
        total_kw = 0.0
        hr = om_hour + 1
        for _, row in rows.iterrows():
            mw = pd.to_numeric(row[col], errors="coerce")
            if pd.isna(mw):
                continue
            hr_str = str(row["hr"]).strip()
            if "-" in hr_str:
                s, e = (int(x) for x in hr_str.split("-"))
            else:
                s = e = int(hr_str)
            if s <= hr <= e:
                total_kw += float(mw) * 1000
        return total_kw
    except Exception as exc:
        logger.warning(f"_contract_scheduled({col}): could not read future_schedule.csv: {exc}")
        return 0.0


def _px_scheduled(om_hour: int, dt: date) -> float:
    return _contract_scheduled("px", om_hour, dt)


def _os_scheduled(om_hour: int, dt: date) -> float:
    return _contract_scheduled("os", om_hour, dt)


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

        # Augment with realtime_history.json so the model learns from recent
        # actual meter readings that haven't made it into the static spreadsheet yet.
        _rh_path = os.path.join(PERSIST_DIR, "realtime_history.json")
        if not os.path.exists(_rh_path):
            _rh_path = os.path.join(_BASE_DIR, "data", "realtime_history.json")
        try:
            with open(_rh_path) as _rh_f:
                _rh_data = json.load(_rh_f)
            existing_dates = set(df["date"].unique())
            extra_rows = []
            for d_str, hrs in _rh_data.items():
                if d_str in existing_dates:
                    continue
                for hr_str, kw in hrs.items():
                    hr = int(hr_str)
                    if kw and 1 <= hr <= 24:
                        extra_rows.append({"date": d_str, "hr": hr, "load": float(kw)})
            if extra_rows:
                extra_df = pd.DataFrame(extra_rows)
                df = pd.concat([df, extra_df], ignore_index=True)
                logger.info(f"Augmented training with {len(extra_rows)} rows from realtime_history.json "
                            f"({extra_df['date'].nunique()} dates: {extra_df['date'].min()} → {extra_df['date'].max()})")
        except Exception as _exc:
            logger.warning(f"Could not load realtime_history.json for training: {_exc}")

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
            # The weather CSV may not cover the most recent days that came in via
            # realtime_history. Fetch archive weather for any gap dates.
            all_train_dates = set(train_df["date"].unique())
            missing_wx = sorted(all_train_dates - set(weather.keys()))
            if missing_wx:
                logger.info(f"Fetching archive weather for {len(missing_wx)} gap dates ({missing_wx[0]} → {missing_wx[-1]})")
                try:
                    gap_weather = await fetch_weather(missing_wx[0], missing_wx[-1])
                    weather.update(gap_weather)
                except Exception as _exc:
                    logger.warning(f"Could not fetch gap weather: {_exc}")

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
        train_dates = pd.to_datetime(train_df["date"])
        days_ago = (max_date - train_dates).dt.days.values
        time_weights = np.exp(-days_ago / 14.0)

        # Same-calendar-month floor: prior-year rows for the current month
        # decay to near-zero (exp(-365/14) ≈ 1e-12), losing all seasonal shape
        # information. Give them a floor equivalent to a 45-day-old row so the
        # model retains the historical morning/evening profile for this month.
        same_month = train_dates.dt.month.values == max_date.month
        SEASONAL_FLOOR = np.exp(-45.0 / 14.0)  # ~0.04
        seasonal_weights = np.where(same_month, np.maximum(time_weights, SEASONAL_FLOOR), time_weights)

        # Same-day-type multiplier: weekday rows get 5× weight when the most
        # recent training day is a weekday (and vice versa for weekends).
        # This anchors weekday forecasts to recent weekday actuals.
        max_dow = max_date.dayofweek  # 0=Mon … 6=Sun
        max_is_weekday = max_dow < 5
        train_dow = train_dates.dt.dayofweek.values
        same_type = (train_dow < 5) == max_is_weekday
        day_type_mult = np.where(same_type, 5.0, 1.0)

        sample_weights = seasonal_weights * day_type_mult

        mdl = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        mdl.fit(X, y, ridge__sample_weight=sample_weights)
        # Score on recent 60 days only — the decay-weighted model is tuned for
        # recent data, so grading on 2-year-old rows gives a misleading R².
        recent_mask = days_ago <= 60
        score_X = X[recent_mask] if recent_mask.sum() >= 24 else X
        score_y = y[recent_mask] if recent_mask.sum() >= 24 else y
        r2 = float(mdl.score(score_X, score_y))

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
    end_str   = (date.today() + timedelta(days=14)).strftime("%Y-%m-%d")
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

        # Forecast-API temps show what the NWP model predicted vs what actually happened
        forecast_day_temps: dict = {}
        try:
            fc_weather = await fetch_weather(target_date, target_date, use_forecast_api=True)
            forecast_day_temps = fc_weather.get(target_date, {})
        except Exception:
            pass

        mdl = model_state["model"]
        hourly = []
        for om_hour in range(24):
            load = day_hist.get(om_hour)
            wx    = day_temps.get(om_hour)
            fc_wx = forecast_day_temps.get(om_hour)
            forecast_load = None
            if mdl is not None and wx is not None:
                try:
                    forecast_load = round(float(max(mdl.predict(
                        build_features(om_hour + 1, wx["temp_f"], wx["apparent_f"], target_date)
                    )[0], 0)), 1)
                except Exception:
                    pass
            hourly.append({
                "hour": om_hour,
                "temp_f": round(wx["temp_f"], 1) if wx else None,
                "forecast_temp_f": round(fc_wx["temp_f"], 1) if fc_wx else None,
                "load": load,
                "forecast": forecast_load,
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

    # Load weather fetched independently — don't skip even if plants timed out
    day_wx_load: dict = {}
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

    # Overlay actuals from rolling history (recent dates not yet in training data)
    _rh_path = os.path.join(PERSIST_DIR, "realtime_history.json")
    if not os.path.exists(_rh_path):
        _rh_path = os.path.join(_BASE_DIR, "data", "realtime_history.json")
    try:
        with open(_rh_path) as _rh_f:
            _rh = json.load(_rh_f)
        if target_date in _rh:
            for _hr_s, _kw_v in _rh[target_date].items():
                _om = int(_hr_s) - 1
                if 0 <= _om <= 23 and _kw_v:
                    load_by_hour[_om] = float(_kw_v)
    except Exception:
        pass

    # Overlay today's realtime data (most current — may have more hours than history)
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

    # Past days: metered PX/OS already include contract deliveries — don't add purchases again.
    if is_historical:
        supply_purchases = []

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
            nb  = round(_nebo_scheduled(om_hour, dt), 1)
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

    hist = dict(model_state["history"])  # shallow copy so we can augment
    _rh_path = os.path.join(PERSIST_DIR, "realtime_history.json")
    if not os.path.exists(_rh_path):
        _rh_path = os.path.join(_BASE_DIR, "data", "realtime_history.json")
    try:
        with open(_rh_path) as _rh_f:
            _rh_data = json.load(_rh_f)
        for _ds, _hrs in _rh_data.items():
            if _ds not in hist:
                hist[_ds] = {int(_hr) - 1: float(_kw) for _hr, _kw in _hrs.items() if _kw}
    except Exception:
        pass

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



_MKT_CACHE: dict = {}
def _market_price(month: int, adder: float = 10.0) -> dict:
    """Market PX price for a calendar month: higher of Palo / Mid-C futures + adder, for HLH and LLH
    (data/market_prices.csv; each month's price is used for that month in every year)."""
    if not _MKT_CACHE:
        try:
            _mp = pd.read_csv(os.path.join(_BASE_DIR, "data", "market_prices.csv"))
            for _, r in _mp.iterrows():
                _MKT_CACHE[int(r["month"])] = (max(float(r["palo_hlh"]), float(r["midc_hlh"])),
                                               max(float(r["palo_llh"]), float(r["midc_llh"])))
        except Exception as _e:
            logger.warning(f"market_prices.csv not read: {_e}")
    h, l = _MKT_CACHE.get(month, (53.0, 53.0))
    return {"hlh": h + adder, "llh": l + adder}


@app.get("/api/longterm")
async def longterm_forecast(
    month: int = Query(..., ge=1, le=12),
    year:  int = Query(..., ge=2020, le=2040),
    sell_rate: float | None = Query(None, ge=0, description="PX sell-back $/MWh (default: half the shortage price)"),
    cap_rate:  float | None = Query(None, ge=0, description="Gen spinning reserve $/MWh (default: half the shortage price)"),
    px_cap_share: float = Query(0.5, ge=0, le=1, description="Share of PX length that also earns a capacity payment"),
    px_cap_rate: float | None = Query(None, ge=0, description="PX capacity $/MWh (default: half the market PX price)"),
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

    # Base year: most recent year before the target year that has actuals for
    # this calendar month (e.g. July 2028 -> July 2026 if July 2027 hasn't happened).
    base_year = None
    for _y in range(year - 1, year - 6, -1):
        _pfx = f"{_y}-{month:02d}-"
        if sum(1 for d in history if d.startswith(_pfx)) >= 20:
            base_year = _y
            break
    years_ahead = (year - base_year) if base_year is not None else 1

    prior_month_daily: dict = {}   # {dom: {om_hour: load}} for base year same month
    if base_year is not None:
        prior_month_str_exact = f"{base_year}-{month:02d}-"
        for d_str, day_data in history.items():
            if not d_str.startswith(prior_month_str_exact):
                continue
            try:
                dom = int(d_str[8:10])
            except (ValueError, IndexError):
                continue
            prior_month_daily[dom] = day_data

    # YoY growth: compare same calendar day + hour vs one year earlier, starting
    # from the month BEFORE the target month (a forecast must not use the target
    # month's own actuals) or the latest month with actuals, whichever is
    # earlier, and stepping back until at least 14 days are matched.
    latest_hist = max(history) if history else None
    anchor_yr, anchor_mo = (year, month - 1) if month > 1 else (year - 1, 12)
    if latest_hist:
        _ly, _lm = int(latest_hist[:4]), int(latest_hist[5:7])
        if (_ly, _lm) < (anchor_yr, anchor_mo):
            anchor_yr, anchor_mo = _ly, _lm

    yoy_num = 0.0    # sum of current-year actuals for matched day+hour pairs
    yoy_den = 0.0    # sum of prior-year actuals for the same pairs
    yoy_days = 0
    for mo_offset in range(0, 6):
        chk_mo = anchor_mo - mo_offset
        chk_yr = anchor_yr
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

    # Total growth applied to the base-year actuals: one year of growth per year ahead.
    growth_applied = round(yoy_growth ** years_ahead, 4) if yoy_growth is not None else None

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
            d_loads = {oh: v * growth_applied for oh, v in base.items() if v is not None}
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
                        px_by_hour[om] += px_kw   # full contract MW; Sundays handled day-by-day below
                        os_by_hour[om] += os_kw
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
        # PX and OS are contracts: no entry in future_schedule.csv means 0.
        # Never fall back to historical PX/OS.
        avg_px = {om: 0.0 for om in range(24)}
        avg_os = {om: 0.0 for om in range(24)}

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
                def _on(om: int, all_day: bool, _sched=sched) -> bool:
                    if _sched == "atc":
                        return True
                    if _sched == "llh":
                        return all_day or om <= 6 or om == 23
                    if _sched == "hlh":        # HE 8-23, Mon-Sat non-holidays
                        return not all_day and 7 <= om <= 22
                    if _sched == "hlh_ex_sp":  # HLH excluding super peak HE 14-21
                        return not all_day and (7 <= om <= 12 or 21 <= om <= 22)
                    if _sched == "sp":         # super peak HE 14-21, Mon-Sat non-holidays
                        return not all_day and 13 <= om <= 20
                    return False
                # Charts/tables: the contracted MW in its usual (Mon-Sat) hours
                kw_by_hour = {om: (kw if _on(om, False) else 0.0) for om in range(24)}
                # Totals: what is actually delivered each day (Sundays/holidays)
                kw_by_day = []
                for day in month_days:
                    all_day = day.weekday() == 6 or day.isoformat() in holidays
                    kw_by_day.append({om: (kw if _on(om, all_day) else 0.0) for om in range(24)})
                purchases.append({
                    "label": lbl, "key": key, "rate": rate,
                    "schedule": sched, "mw": float(pur_row["mw"]),
                    "kw_by_hour": kw_by_hour, "kw_by_day": kw_by_day,
                })
        except Exception as _exc:
            logger.warning(f"Power purchases load failed: {_exc}")

    # ── 4. Build hourly averages for bar chart ────────────────────────────
    month_name = _cal.month_name[month]
    hourly = []
    for om in range(24):
        load     = round(sum(hour_loads[om]) / len(hour_loads[om])) if hour_loads[om] else 0
        max_load = round(max(hour_loads[om])) if hour_loads[om] else 0
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
            "hour": om, "load": load, "max_load": max_load,
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
    # Contracts count only on the days they deliver (schedule PX/OS: Mon-Sat;
    # purchases: their HLH/LLH/ATC day rules), so monthly totals stay exact.
    base_by_hr = {
        om: (round(avg_nebo.get(om, 0)) + round(avg_h_butte.get(om, 0)) +
             round(avg_rm.get(om, 0))   + round(avg_sa.get(om, 0)) +
             round(avg_uamps.get(om, 0)))
        for om in range(24)
    }
    month_kwh = {"px": 0.0, "os": 0.0, "internal_gen": 0.0, "shortage": 0.0}
    for p in purchases:
        month_kwh[p["key"]] = 0.0
    daily_shortages_kwh: list = []
    short_sum_by_hr = [0.0] * 24
    # Average shortage by hour in three buckets: HLH (HE 8-23 Mon-Sat non-holiday),
    # Mon-Sat LLH (HE 1-7 & 24 Mon-Sat non-holiday), and Sunday/holiday (all hours)
    _hol_lt = _wecc_holidays(year)
    hl_sum, hl_cnt = [0.0] * 24, [0] * 24
    ll_sum, ll_cnt = [0.0] * 24, [0] * 24
    sh_sum, sh_cnt = [0.0] * 24, [0] * 24
    days_detail: list = []   # per-day load and supply (before internal gen) for the What-If tab
    px_long_kwh = 0.0    # supply above load, up to the contract volume that hour (sold back)
    px_long_hlh_kwh = 0.0   # the part of px_long_kwh in HLH hours (HE 8-23 Mon-Sat, non-holiday)
    gen_long_kwh = 0.0   # Lehi gen capacity (21 MW) not needed that hour (spinning reserve)
    for day, d in zip(month_days, day_hour_loads):
        sched_on = day.weekday() != 6
        lowday = day.weekday() == 6 or day.isoformat() in _hol_lt
        day_total = 0.0
        day_supply = [0.0] * 24
        day_contract = [0.0] * 24   # PX + OS + purchased contracts (sellable when long)
        for om in range(24):
            px_kw = round(avg_px.get(om, 0)) if sched_on else 0.0
            os_kw = round(avg_os.get(om, 0)) if sched_on else 0.0
            month_kwh["px"] += px_kw
            month_kwh["os"] += os_kw
            pur = 0.0
            for p in purchases:
                v = p["kw_by_day"][day.day - 1][om]
                month_kwh[p["key"]] += v
                pur += v
            day_supply[om] = base_by_hr[om] + px_kw + os_kw + pur
            day_contract[om] = px_kw + os_kw + pur
            raw = max(0.0, d.get(om, 0.0) - day_supply[om])
            ig = min(raw, 21000.0)
            _pxl = min(max(0.0, day_supply[om] - d.get(om, 0.0)), day_contract[om])
            px_long_kwh += _pxl
            if not lowday and 7 <= om <= 22:
                px_long_hlh_kwh += _pxl
            gen_long_kwh += 21000.0 - ig
            month_kwh["internal_gen"] += ig
            month_kwh["shortage"] += raw - ig
            short_sum_by_hr[om] += raw - ig
            if lowday:
                sh_sum[om] += raw - ig; sh_cnt[om] += 1
            elif 7 <= om <= 22:
                hl_sum[om] += raw - ig; hl_cnt[om] += 1
            else:
                ll_sum[om] += raw - ig; ll_cnt[om] += 1
            day_total += raw - ig
        daily_shortages_kwh.append(day_total)
        days_detail.append({
            "date": day.isoformat(),
            "load": [round(d.get(om, 0.0)) for om in range(24)],
            "supply": [round(v) for v in day_supply],
            "contract": [round(v) for v in day_contract],
        })
    n_days = len(days_detail) or 1
    avg_short_by_hr = [v / n_days for v in short_sum_by_hr]
    avg_daily_shortage_mwh = round(sum(daily_shortages_kwh) / len(daily_shortages_kwh) / 1000, 1) if daily_shortages_kwh else 0.0

    avg_load   = round(sum(h["load"]         for h in hourly) / 24)
    avg_supply = round(sum(h["total_supply"] for h in hourly) / 24)
    peak_h = max(hourly, key=lambda h: h["load"])

    _pie_keys = ["nebo", "h_butte", "px", "os", "red_mesa", "steele_a", "uamps", "internal_gen", "shortage"]
    _pie_keys += [p["key"] for p in purchases]
    pie = {k: round(month_kwh[k] / days_in_month) if k in month_kwh else round(sum(h[k] for h in hourly))
           for k in _pie_keys}

    # ── 6. Blended resource cost ──────────────────────────────────────────
    blended_cost_per_mwh = None
    month_total_cost: float | None = None
    cost_pie:   dict = {}
    cost_rates: dict = {}
    shortage_rate: float | None = None   # assumed $/MWh for shortage (None = no rate set)
    energy_cost: float | None = None
    sell_back_rate = capacity_rate = sell_back_revenue = capacity_revenue = None
    px_capacity_revenue = px_cap_rate_hlh = px_cap_rate_llh = None
    _month_load_mwh = sum(sum(d.values()) for d in day_hour_loads) / 1000.0
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

            _sr = c_df[(c_df["resource"] == "shortage") & ((c_df["month"] == month) | c_df["month"].isna())]
            if not _sr.empty:
                shortage_rate = _res_cost("shortage")

            # Step 1: total monthly MWh per resource
            # Use exactly the same data as the resource graphs (hourly array).
            resource_mwh: dict = {}

            # Non-UAMPS resources — directly from hourly (includes internal_gen)
            for res in ["nebo", "h_butte", "red_mesa", "steele_a"]:
                monthly_kwh = sum(h.get(res, 0) for h in hourly) * days_in_month
                resource_mwh[res] = monthly_kwh / 1000.0
            # Contracts, internal gen, shortage — actual day-by-day energy
            for res in ["px", "os", "internal_gen", "shortage"]:
                resource_mwh[res] = month_kwh[res] / 1000.0

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
                resource_mwh[p["key"]] = month_kwh[p["key"]] / 1000.0

            def _get_rate(res: str) -> float:
                r = _pur_rates.get(res)
                return r if r is not None else _res_cost(res)

            # Step 2: energy cost per resource = MWh × $/MWh
            total_cost = sum(mwh * _get_rate(res) for res, mwh in resource_mwh.items())
            energy_cost = round(total_cost, 2)
            # Step 3: revenue when long. Assumption: PX sell-back and gen spinning reserve
            # are each paid at half the month's shortage price unless the user gives a price.
            _half = 0.5 * _res_cost("shortage")
            sell_back_rate = float(sell_rate) if isinstance(sell_rate, (int, float)) else _half
            capacity_rate  = float(cap_rate)  if isinstance(cap_rate,  (int, float)) else _half
            sell_back_revenue = round(px_long_kwh / 1000.0 * sell_back_rate, 2)
            capacity_revenue  = round(gen_long_kwh / 1000.0 * capacity_rate, 2)
            # PX capacity: a share of the PX length also earns a capacity payment (on top of
            # its sell-back) at half the market PX price (higher of Palo / Mid-C + $10) for
            # HLH / LLH hours, unless the user gives a price.
            _mp = _market_price(month)
            px_cap_rate_hlh = float(px_cap_rate) if isinstance(px_cap_rate, (int, float)) else round(_mp["hlh"] / 2, 2)
            px_cap_rate_llh = float(px_cap_rate) if isinstance(px_cap_rate, (int, float)) else round(_mp["llh"] / 2, 2)
            _pcs = float(px_cap_share) if isinstance(px_cap_share, (int, float)) else 0.5
            px_capacity_revenue = round(_pcs * (px_long_hlh_kwh * px_cap_rate_hlh
                                                + (px_long_kwh - px_long_hlh_kwh) * px_cap_rate_llh) / 1000.0, 2)
            # Step 4: net total cost and blended = net cost per MWh of load
            month_total_cost = round(total_cost - sell_back_revenue - px_capacity_revenue - capacity_revenue, 2)
            if _month_load_mwh > 0:
                blended_cost_per_mwh = round(month_total_cost / _month_load_mwh, 2)
            cost_pie   = {res: round(mwh * _get_rate(res))
                          for res, mwh in resource_mwh.items()
                          if round(mwh * _get_rate(res)) > 0}
            cost_rates = {res: _get_rate(res) for res in cost_pie}
        except Exception as _exc:
            logger.warning(f"Blended cost calculation failed: {_exc}")

    # ── 7. Actual hourly averages for this calendar month ────────────────
    # Combines model_state["history"] with realtime_history.json so partial
    # months still show as many days as are available.
    actual_by_hour: dict = {om: [] for om in range(24)}
    month_prefix = f"{year}-{month:02d}-"
    _all_history = dict(model_state["history"])
    _rh_path_lt = os.path.join(PERSIST_DIR, "realtime_history.json")
    if not os.path.exists(_rh_path_lt):
        _rh_path_lt = os.path.join(_BASE_DIR, "data", "realtime_history.json")
    try:
        with open(_rh_path_lt) as _f:
            _rh_lt = json.load(_f)
        for _ds, _hrs in _rh_lt.items():
            if _ds not in _all_history:
                _all_history[_ds] = {int(_hr) - 1: float(_kw) for _hr, _kw in _hrs.items() if _kw}
    except Exception:
        pass
    # Also scan all years for the historical calendar-month peak
    month_str_any = f"-{month:02d}-"
    hist_peak_val: float | None = None
    hist_peak_date: str | None = None
    hist_peak_hour: int | None = None
    for d_str, day_data in _all_history.items():
        is_target_year = d_str.startswith(month_prefix)
        is_any_year    = month_str_any in d_str
        if is_target_year:
            for om, load_val in day_data.items():
                if load_val is not None:
                    actual_by_hour[int(om)].append(float(load_val))
        if is_any_year:
            for om, load_val in day_data.items():
                lv = float(load_val) if load_val is not None else None
                if lv and (hist_peak_val is None or lv > hist_peak_val):
                    hist_peak_val  = lv
                    hist_peak_date = d_str
                    hist_peak_hour = int(om)
    actual_hourly = [
        round(sum(actual_by_hour[om]) / len(actual_by_hour[om])) if actual_by_hour[om] else None
        for om in range(24)
    ]

    return {
        "month": month, "year": year,
        "label": f"{month_name} {year}",
        "hourly": hourly,
        "actual_hourly": actual_hourly,
        "hist_peak": {
            "load": round(hist_peak_val) if hist_peak_val else None,
            "date": hist_peak_date,
            "hour": hist_peak_hour,
        },
        "summary": {
            "avg_load": avg_load, "avg_supply": avg_supply,
            "avg_daily_shortage_mwh": avg_daily_shortage_mwh,
            "peak_load": peak_h["load"], "peak_hour": peak_h["hour"],
        },
        "pie": pie,
        "month_load_kwh": round(sum(sum(d.values()) for d in day_hour_loads)),
        "month_peak_kw": max((h["max_load"] for h in hourly), default=0),
        "avg_shortage_by_hour": [round(v) for v in avg_short_by_hr],
        "avg_shortage_hlh_by_hour": [round(hl_sum[i] / hl_cnt[i]) if hl_cnt[i] else None for i in range(24)],
        "avg_shortage_llh_by_hour": [round(ll_sum[i] / ll_cnt[i]) if ll_cnt[i] else None for i in range(24)],
        "avg_shortage_sunhol_by_hour": [round(sh_sum[i] / sh_cnt[i]) if sh_cnt[i] else None for i in range(24)],
        "days": days_detail,
        "month_shortage_kwh": round(sum(daily_shortages_kwh)),
        "month_total_cost": month_total_cost,
        "energy_cost": energy_cost,
        "sell_back_rate": sell_back_rate,
        "capacity_rate": capacity_rate,
        "sell_back_revenue": sell_back_revenue,
        "capacity_revenue": capacity_revenue,
        "px_capacity_revenue": px_capacity_revenue,
        "px_cap_share": px_cap_share if isinstance(px_cap_share, (int, float)) else 0.5,
        "px_cap_rate_hlh": px_cap_rate_hlh,
        "px_cap_rate_llh": px_cap_rate_llh,
        "px_long_mwh": round(px_long_kwh / 1000.0, 1),
        "px_long_hlh_mwh": round(px_long_hlh_kwh / 1000.0, 1),
        "gen_long_mwh": round(gen_long_kwh / 1000.0, 1),
        "yoy_growth": yoy_growth,
        "yoy_days": yoy_days,
        "base_year": base_year,
        "years_ahead": years_ahead,
        "growth_applied": growth_applied,
        "blended_cost_per_mwh": blended_cost_per_mwh,
        "cost_pie": cost_pie,
        "cost_rates": cost_rates,
        "shortage_rate": shortage_rate,
        "purchases": [{"label": p["label"], "key": p["key"], "mw": p["mw"],
                        "rate": p["rate"], "schedule": p["schedule"]} for p in purchases],
    }


def _pool_scenarios() -> dict:
    """data/pool_scenarios.csv -> {month: {typical_hl, typical_ll, worst_hl, worst_ll, worst_year}}.
    Typical = average Sch Log pool price for the month over 2019-2025; worst = the worst year (2022)."""
    out: dict = {}
    try:
        _d = pd.read_csv(os.path.join(_BASE_DIR, "data", "pool_scenarios.csv"))
        for _, r in _d.iterrows():
            out[int(r["month"])] = {k: (float(r[k]) if k != "typical_years" else str(r[k]))
                                    for k in ("typical_hl", "typical_ll", "worst_hl", "worst_ll", "typical_years")}
            out[int(r["month"])]["worst_year"] = int(r["worst_year"])
    except Exception as _e:
        logger.warning(f"pool_scenarios.csv not read: {_e}")
    return out


@app.get("/api/pool_scenarios")
async def pool_scenarios():
    """Monthly pool price scenarios (typical 7-year average and worst year) from the Sch Log."""
    return {"months": _pool_scenarios()}


@app.get("/api/optimal")
async def optimal_purchases(
    start: str = Query("2027-01", description="YYYY-MM"),
    end:   str = Query("2029-12", description="YYYY-MM"),
    hlh_price: float | None = Query(None, ge=0, description="$/MWh for added HLH (default: market price + adder)"),
    llh_price: float | None = Query(None, ge=0, description="$/MWh for added LLH (default: market price + adder)"),
    adder: float = Query(10.0, description="$/MWh added to the higher of Palo / Mid-C"),
    sell_rate: float | None = Query(None, ge=0),
    cap_rate:  float | None = Query(None, ge=0),
    max_mw: int = Query(150, ge=0, le=300),
    pool: str | None = Query(None, description="'typical' or 'worst': price shortage at Sch Log pool prices (HL / LL); "
                                               "'best': the plan that loses the least either way (typical vs worst)"),
    px_cap_share: float = Query(0.5, ge=0, le=1, description="Share of PX length that also earns a capacity payment"),
    px_cap_rate: float | None = Query(None, ge=0, description="PX capacity $/MWh (default: half the block's purchase price)"),
):
    """Minimum-cost additional HLH and LLH MW for each month.
    For each month the Long Term forecast's day-by-day load / supply is re-run with
    H MW added in every HLH hour and L MW in every LLH hour. Each hour then costs:
    internal gen x gen rate + shortage x shortage price + added MW x purchase price
    - PX length x sell-back price - PX length x share x PX capacity price
    - unused gen capacity x spinning reserve price.
    HLH and LLH hours don't interact, so H and L are searched separately (1 MW steps)."""
    sy, sm = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    _sr = sell_rate if isinstance(sell_rate, (int, float)) else None
    _cr = cap_rate if isinstance(cap_rate, (int, float)) else None
    _hp_in = float(hlh_price) if isinstance(hlh_price, (int, float)) else None
    _lp_in = float(llh_price) if isinstance(llh_price, (int, float)) else None
    _add = float(adder) if isinstance(adder, (int, float)) else 10.0
    # Market futures (data/market_prices.csv): purchase price = higher of Palo / Mid-C + adder.
    # Each calendar month's price is used for that month in every year.
    _mkt: dict = {}
    try:
        _mp = pd.read_csv(os.path.join(_BASE_DIR, "data", "market_prices.csv"))
        for _, r in _mp.iterrows():
            _mkt[int(r["month"])] = {
                "hlh": max(float(r["palo_hlh"]), float(r["midc_hlh"])),
                "llh": max(float(r["palo_llh"]), float(r["midc_llh"])),
                "hlh_hub": "Palo" if float(r["palo_hlh"]) >= float(r["midc_hlh"]) else "Mid-C",
                "llh_hub": "Palo" if float(r["palo_llh"]) >= float(r["midc_llh"]) else "Mid-C",
            }
    except Exception as _e:
        logger.warning(f"market_prices.csv not read: {_e}")
    _mx = int(max_mw) if isinstance(max_mw, int) else 150
    ig_rate = 0.0
    try:
        _c = pd.read_csv(os.path.join(_BASE_DIR, "data", "resource_costs.csv"))
        _r = _c[(_c["resource"] == "internal_gen")]
        if not _r.empty:
            ig_rate = float(_r["cost_per_mwh"].iloc[0])
    except Exception:
        pass
    GEN = 21000.0
    _best_mode = pool == "best"
    _pool = pool if pool in ("typical", "worst") else None
    _ps = _pool_scenarios() if (_pool or _best_mode) else {}
    _pcs = float(px_cap_share) if isinstance(px_cap_share, (int, float)) else 0.5
    _pcr = float(px_cap_rate) if isinstance(px_cap_rate, (int, float)) else None
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        lt = await longterm_forecast(month=m, year=y, sell_rate=_sr, cap_rate=_cr,
                                     px_cap_share=_pcs, px_cap_rate=_pcr)
        _mk = _mkt.get(m)
        _hp = _hp_in if _hp_in is not None else (_mk["hlh"] + _add if _mk else 63.0)
        _lp = _lp_in if _lp_in is not None else (_mk["llh"] + _add if _mk else 63.0)
        _hsrc = "your price" if _hp_in is not None else (f"{_mk['hlh_hub']} + ${_add:g}" if _mk else "default")
        _lsrc = "your price" if _lp_in is not None else (f"{_mk['llh_hub']} + ${_add:g}" if _mk else "default")
        sh_def = lt.get("shortage_rate") or 0.0           # default shortage price (for the fixed part)
        hol  = _wecc_holidays(y)
        hlh_hrs, llh_hrs = [], []          # (load, supply, contract) per hour
        for d in lt.get("days", []):
            dt = date.fromisoformat(d["date"])
            low = dt.weekday() == 6 or d["date"] in hol
            con = d.get("contract") or [0] * 24
            for om in range(24):
                rec = (d["load"][om], d["supply"][om], con[om])
                (hlh_hrs if (not low and 7 <= om <= 22) else llh_hrs).append(rec)

        def _rates(scen, cls, price):
            """(shortage, sell-back, spinning reserve, PX capacity) $/MWh for one block of hours."""
            k = "hl" if cls == "hlh" else "ll"
            sh = _ps[m][f"{scen}_{k}"] if (scen and m in _ps) else sh_def
            return (sh,
                    _sr if _sr is not None else sh / 2,
                    _cr if _cr is not None else sh / 2,
                    _pcr if _pcr is not None else price / 2)

        def _block(hrs, add_kw, price, r):
            """Energy cost and revenues for one block of hours with add_kw added in every hour."""
            sh_r, s_r, c_r, p_r = r
            ig_k = sh_k = pxl_k = genl_k = 0.0
            for ld, su, co in hrs:
                raw = max(0.0, ld - su - add_kw)
                ig = min(raw, GEN)
                ig_k += ig; sh_k += raw - ig
                pxl_k += min(max(0.0, su + add_kw - ld), co + add_kw)
                genl_k += GEN - ig
            pur = add_kw * len(hrs) / 1000.0 * price
            energy = ig_k / 1000.0 * ig_rate + sh_k / 1000.0 * sh_r + pur
            sell = pxl_k / 1000.0 * s_r
            pxcap = pxl_k / 1000.0 * _pcs * p_r
            cap = genl_k / 1000.0 * c_r
            return {"energy": energy, "purchase": pur, "sell": sell, "pxcap": pxcap, "cap": cap,
                    "net": energy - sell - pxcap - cap, "shortage_mwh": sh_k / 1000.0}

        def _need_mw(hrs):
            """Most the block ever needs: the largest hourly shortfall before Lehi gens
            (load - supply), rounded up to whole MW. Never buy more than this, so power
            isn't bought just to sell it back."""
            mx = max((ld - su for ld, su, _ in hrs), default=0.0)
            return max(0, int(-(-mx // 1000)))

        def _curve(hrs, price, r):
            return [_block(hrs, mw * 1000.0, price, r) for mw in range(0, min(_mx, _need_mw(hrs)) + 1)]

        def _argmin(c):
            best = 0
            for i in range(1, len(c)):
                if c[i]["net"] < c[best]["net"] - 0.5:
                    best = i
            return best

        # Parts of the plan cost that don't depend on the purchase (Nebo, UAMPS, existing PX, ...):
        # the forecast's energy cost minus its gen + shortage cost at the default shortage price
        _z = (sh_def, 0, 0, 0)
        fixed = (lt.get("energy_cost") or 0.0) - (_block(hlh_hrs, 0.0, 0.0, _z)["energy"]
                                                   + _block(llh_hrs, 0.0, 0.0, _z)["energy"])
        row = {
            "year": y, "month": m, "label": f"{_cal_mod.month_abbr[m]} {y}",
            "hlh_need_mw": _need_mw(hlh_hrs), "llh_need_mw": _need_mw(llh_hrs),
            "hlh_hours": len(hlh_hrs), "llh_hours": len(llh_hrs),
            "load_mwh": round((lt.get("month_load_kwh") or 0) / 1000.0, 1),
            "shortage_rate": sh_def,
            "hlh_price": round(_hp, 2), "llh_price": round(_lp, 2),
            "hlh_price_source": _hsrc, "llh_price_source": _lsrc,
        }

        if _best_mode:
            # Best plan: for each block, the MW whose worst-case regret (cost above the plan
            # that would have been cheapest in hindsight) is smallest across typical / worst prices.
            tot = {"typ_net": fixed, "worst_net": fixed, "typ_base": fixed, "worst_base": fixed, "regret": 0.0,
                   "purchase_cost": 0.0}
            for cls, hrs, price in (("hlh", hlh_hrs, _hp), ("llh", llh_hrs, _lp)):
                ct = _curve(hrs, price, _rates("typical", cls, price))
                cw = _curve(hrs, price, _rates("worst", cls, price))
                bt, bw = _argmin(ct), _argmin(cw)
                reg = [max(ct[i]["net"] - ct[bt]["net"], cw[i]["net"] - cw[bw]["net"]) for i in range(len(ct))]
                br = 0
                for i in range(1, len(reg)):
                    if reg[i] < reg[br] - 0.5:
                        br = i
                row[f"{cls}_typical_mw"], row[f"{cls}_worst_mw"], row[f"{cls}_mw"] = bt, bw, br
                row[f"{cls}_regret"] = round(reg[br], 2)
                row[f"{cls}_mwh"] = round(br * len(hrs), 1)
                tot["typ_net"] += ct[br]["net"]; tot["worst_net"] += cw[br]["net"]
                tot["typ_base"] += ct[0]["net"]; tot["worst_base"] += cw[0]["net"]
                tot["regret"] += reg[br]; tot["purchase_cost"] += ct[br]["purchase"]
            row.update({
                "purchase_cost": round(tot["purchase_cost"], 2),
                "net_cost_typical": round(tot["typ_net"], 2), "net_cost_worst": round(tot["worst_net"], 2),
                "base_net_typical": round(tot["typ_base"], 2), "base_net_worst": round(tot["worst_base"], 2),
                "savings_typical": round(tot["typ_base"] - tot["typ_net"], 2),
                "savings_worst": round(tot["worst_base"] - tot["worst_net"], 2),
                "max_regret": round(tot["regret"], 2),
            })
        else:
            rh, rl = _rates(_pool, "hlh", _hp), _rates(_pool, "llh", _lp)
            ch, cl = _curve(hlh_hrs, _hp, rh), _curve(llh_hrs, _lp, rl)
            h_mw, l_mw = _argmin(ch), _argmin(cl)
            hb, lb = ch[h_mw], cl[l_mw]
            base_net = round(fixed + ch[0]["net"] + cl[0]["net"], 2)   # no added purchase, same prices
            energy = fixed + hb["energy"] + lb["energy"]
            sell, pxcap, cap = hb["sell"] + lb["sell"], hb["pxcap"] + lb["pxcap"], hb["cap"] + lb["cap"]
            net = energy - sell - pxcap - cap
            row.update({
                "hlh_mw": h_mw, "llh_mw": l_mw,
                "hlh_mwh": round(h_mw * len(hlh_hrs), 1), "llh_mwh": round(l_mw * len(llh_hrs), 1),
                "purchase_cost": round(hb["purchase"] + lb["purchase"], 2),
                "energy_cost": round(energy, 2),
                "sell_back_revenue": round(sell, 2),
                "px_capacity_revenue": round(pxcap, 2),
                "capacity_revenue": round(cap, 2),
                "net_cost": round(net, 2),
                "base_net_cost": base_net,
                "savings": round(base_net - net, 2),
                "shortage_mwh": round(hb["shortage_mwh"] + lb["shortage_mwh"], 1),
                "shortage_hlh": round(rh[0], 2), "shortage_llh": round(rl[0], 2),
                "sell_back_rate": round(rh[1], 2), "capacity_rate": round(rh[2], 2),
                "sell_back_llh": round(rl[1], 2), "capacity_llh": round(rl[2], 2),
                "px_cap_rate_hlh": round(rh[3], 2), "px_cap_rate_llh": round(rl[3], 2),
            })
        months.append(row)
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return {"adder": _add, "internal_gen_rate": ig_rate, "pool": "best" if _best_mode else _pool,
            "px_cap_share": _pcs, "months": months}


@app.get("/api/yearly")
async def yearly_summary(
    year: int = Query(..., ge=2020, le=2040),
    sell_rate: float | None = Query(None, ge=0),
    cap_rate:  float | None = Query(None, ge=0),
    px_cap_share: float = Query(0.5, ge=0, le=1),
    px_cap_rate: float | None = Query(None, ge=0),
):
    """Predicted vs actual by month for one year.
    Predicted: the Long Term forecast for each month.
    Actuals: data/monthly_actuals.csv, from the Power Bills workbooks' LFY Data
    sheets (load = Total Kwh Purchased, total cost = Energy Expense, blended rate =
    Blended Cost with Generation (actual)). Months without a bill use metered load."""
    import calendar as _cal

    # Actual metered load (load history + realtime history), complete months only
    _all_history = dict(model_state["history"])
    _rh = os.path.join(PERSIST_DIR, "realtime_history.json")
    if not os.path.exists(_rh):
        _rh = os.path.join(_BASE_DIR, "data", "realtime_history.json")
    try:
        with open(_rh) as _f:
            for _ds, _hrs in json.load(_f).items():
                if _ds not in _all_history:
                    _all_history[_ds] = {int(_h) - 1: float(_kw) for _h, _kw in _hrs.items() if _kw}
    except Exception:
        pass

    # Actuals from bills
    bills: dict = {}
    _bp = os.path.join(_BASE_DIR, "data", "monthly_actuals.csv")
    if os.path.exists(_bp):
        try:
            _b = pd.read_csv(_bp)
            for _, r in _b.iterrows():
                if int(r["year"]) != year:
                    continue
                def _num(col):
                    v = pd.to_numeric(r.get(col), errors="coerce")
                    return float(v) if pd.notna(v) else None
                bills[int(r["month"])] = {
                    "load_kwh":     _num("load_kwh"),
                    "shortage_kwh": _num("shortage_kwh"),
                    "total_cost":   _num("total_cost"),
                    "blended_rate": _num("blended_rate"),
                }
        except Exception as exc:
            logger.warning(f"monthly_actuals.csv read failed: {exc}")

    months = []
    for m in range(1, 13):
        row = {"month": m, "name": _cal.month_name[m]}
        try:
            lt = await longterm_forecast(month=m, year=year,
                                         sell_rate=sell_rate if isinstance(sell_rate, (int, float)) else None,
                                         cap_rate=cap_rate if isinstance(cap_rate, (int, float)) else None,
                                         px_cap_share=px_cap_share if isinstance(px_cap_share, (int, float)) else 0.5,
                                         px_cap_rate=px_cap_rate if isinstance(px_cap_rate, (int, float)) else None)
            row["pred_load_kwh"]     = lt["month_load_kwh"]
            row["pred_shortage_kwh"] = lt["month_shortage_kwh"]
            row["pred_total_cost"]   = lt["month_total_cost"]
            row["pred_energy_cost"]  = lt["energy_cost"]
            row["pred_sell_back_revenue"] = lt["sell_back_revenue"]
            row["pred_capacity_revenue"]  = lt["capacity_revenue"]
            row["pred_px_capacity_revenue"] = lt.get("px_capacity_revenue")
            row["sell_back_rate"] = lt["sell_back_rate"]
            row["capacity_rate"]  = lt["capacity_rate"]
            row["pred_peak_kw"]      = lt["month_peak_kw"]
            _ash = lt["avg_shortage_by_hour"]
            _mn, _mx = min(range(24), key=lambda i: _ash[i]), max(range(24), key=lambda i: _ash[i])
            row["min_avg_shortage_kw"], row["min_avg_shortage_he"] = _ash[_mn], _mn + 1
            row["max_avg_shortage_kw"], row["max_avg_shortage_he"] = _ash[_mx], _mx + 1
            for _cls in ("hlh", "llh", "sunhol"):
                _v = lt[f"avg_shortage_{_cls}_by_hour"]
                _ix = [i for i in range(24) if _v[i] is not None]
                if _ix:
                    _a, _b = min(_ix, key=lambda i: _v[i]), max(_ix, key=lambda i: _v[i])
                    row[f"{_cls}_min_kw"], row[f"{_cls}_min_he"] = _v[_a], _a + 1
                    row[f"{_cls}_max_kw"], row[f"{_cls}_max_he"] = _v[_b], _b + 1
        except HTTPException as exc:
            if exc.status_code == 503:
                raise
            row["pred_load_kwh"] = row["pred_shortage_kwh"] = row["pred_total_cost"] = None
            row["pred_peak_kw"] = row["min_avg_shortage_kw"] = row["max_avg_shortage_kw"] = None
            row["min_avg_shortage_he"] = row["max_avg_shortage_he"] = None
        days = _cal.monthrange(year, m)[1]
        prefix = f"{year}-{m:02d}-"
        mdays = [d for d in _all_history if d.startswith(prefix)]
        meter_load = (round(sum(sum(v for v in _all_history[d].values() if v) for d in mdays))
                      if len(mdays) >= days else None)
        b = bills.get(m, {})
        # Bill figures take priority; meters fill load for months without a bill yet
        row["load_kwh"]      = round(b["load_kwh"]) if b.get("load_kwh") else meter_load
        row["load_source"]   = "bill" if b.get("load_kwh") else ("meters" if meter_load else None)
        row["shortage_kwh"]  = b.get("shortage_kwh")
        row["total_cost"]    = b.get("total_cost")
        row["blended_rate"]  = b.get("blended_rate")
        months.append(row)
    return {"year": year, "months": months}


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
        # PX and OS are contracts: no entry in future_schedule.csv means 0.
        # Never fall back to historical PX/OS.
        avg_px = {om: 0.0 for om in range(24)}
        avg_os = {om: 0.0 for om in range(24)}

    # PX contracts in power_purchases.csv (labels starting "PX") for this day,
    # added on top of any future_schedule.csv PX.
    _pur_path_d = os.path.join(_BASE_DIR, "data", "power_purchases.csv")
    if os.path.exists(_pur_path_d):
        try:
            _pd_df = pd.read_csv(_pur_path_d)
            _pd_df = _pd_df[(_pd_df["year"].astype(int) == year) & (_pd_df["month"].astype(int) == month)
                            & _pd_df["label"].astype(str).str.strip().str.upper().str.startswith("PX")]
            _all_day = dt.weekday() == 6 or dt.isoformat() in _wecc_holidays(year)
            avg_px = dict(avg_px)
            for _, _pr in _pd_df.iterrows():
                _kw = float(_pr["mw"]) * 1000.0
                _sc = str(_pr["schedule"]).strip().lower()
                for om in range(24):
                    if _sc == "atc":
                        on = True
                    elif _sc == "llh":
                        on = _all_day or om <= 6 or om == 23
                    elif _sc == "hlh":
                        on = not _all_day and 7 <= om <= 22
                    elif _sc == "hlh_ex_sp":
                        on = not _all_day and (7 <= om <= 12 or 21 <= om <= 22)
                    elif _sc == "sp":
                        on = not _all_day and 13 <= om <= 20
                    else:
                        on = False
                    if on:
                        avg_px[om] = avg_px.get(om, 0.0) + _kw
        except Exception as _exc:
            logger.warning(f"Day plan: PX purchases load failed: {_exc}")

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


app.mount("/", StaticFiles(directory="static"), name="static")
