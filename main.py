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
    "crsp":      5,
    "provo_riv": 8,
    "veyo":      15,
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
    url = os.environ.get("UAMPS_SCHEDULE_CSV_URL")
    if not url:
        return {}
    try:
        resp = httpx.get(url, timeout=30.0)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df["date"] = df["date"].astype(str)
        df["hr"]   = df["hr"].astype(int)
        for col in ("crsp", "provo_riv", "veyo"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        result: dict = {}
        for _, row in df.iterrows():
            result.setdefault(str(row["date"]), {})[int(row["hr"])] = {
                "crsp":      int(row["crsp"]),
                "provo_riv": int(row["provo_riv"]),
                "veyo":      int(row["veyo"]),
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
_WEATHER_CACHE_TTL = 1800  # 30 minutes
_openmeteo_semaphore: asyncio.Semaphore | None = None  # created after event loop starts


def _get_openmeteo_sem() -> asyncio.Semaphore:
    global _openmeteo_semaphore
    if _openmeteo_semaphore is None:
        _openmeteo_semaphore = asyncio.Semaphore(1)
    return _openmeteo_semaphore

async def fetch_weather(start_date: str, end_date: str, use_forecast_api: bool = False) -> dict:
    """Returns {date_str: {openmeteo_hour_0_to_23: temp_f}}"""
    cache_key = (start_date, end_date, use_forecast_api)
    cached = _weather_cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < _WEATHER_CACHE_TTL:
        logger.info(f"Weather cache hit for {start_date}–{end_date}")
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
    async with _get_openmeteo_sem():
        async with httpx.AsyncClient(timeout=120.0) as client:
            for attempt in range(6):
                resp = await client.get(base_url, params=params)
                if resp.status_code == 429:
                    wait = 20 * (attempt + 1)
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
    _weather_cache[cache_key] = (time.monotonic(), result)
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
    async with _get_openmeteo_sem():
        async with httpx.AsyncClient(timeout=120.0) as client:
            for attempt in range(6):
                resp = await client.get(base_url, params=params)
                if resp.status_code == 429:
                    wait = 20 * (attempt + 1)
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
    async with _get_openmeteo_sem():
        async with httpx.AsyncClient(timeout=120.0) as client:
            for attempt in range(6):
                resp = await client.get(base_url, params=params)
                if resp.status_code == 429:
                    wait = 20 * (attempt + 1)
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

        mdl = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        mdl.fit(X, y)
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(train_model())
    asyncio.create_task(_init_supply())
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
    return {"date": target_date, "type": "forecast", "hourly": hourly, "summary": summary}


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

    red_mesa_kwh    = await _plant_hourly_kwh("red-mesa", target_date, dt)
    steele_a_kwh    = await _plant_hourly_kwh("steele-a", target_date, dt)
    horse_butte_kwh = await _wind_plant_hourly_kwh("horse-butte", target_date, dt)

    # Load — actuals if in history, otherwise model forecast
    load_by_hour: dict = {}
    if target_date in model_state["history"]:
        load_by_hour = model_state["history"][target_date]
    elif model_state["model"] is not None:
        try:
            use_load_fc = dt >= date.today() - timedelta(days=7)
            wx_load = await fetch_weather(target_date, target_date, use_forecast_api=use_load_fc)
            day_wx_load = wx_load.get(target_date, {})
            mdl = model_state["model"]
            for om_hour in range(24):
                w = day_wx_load.get(om_hour)
                if w:
                    pred = float(max(mdl.predict(build_features(om_hour + 1, w["temp_f"], w["apparent_f"], target_date))[0], 0))
                    load_by_hour[om_hour] = round(pred, 1)
        except Exception:
            pass

    supply_day = supply_history.get(target_date, {})
    uamps_day  = await _uamps_get_day(dt)

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
        total = round(rm + sa + nb + hb + px + os_ + crsp + provo_riv + veyo, 1)
        hourly.append({
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
        })

    warnings = [
        f"{SOLAR_PLANTS[k]['label']} model not ready"
        for k in ("red-mesa", "steele-a")
        if solar_states[k]["model"] is None
    ]
    if wind_states["horse-butte"]["model"] is None:
        warnings.append("Horse Butte model not ready")

    day_total = round(sum(h["total"] for h in hourly), 1)
    return {
        "date": target_date,
        "type": "historical" if is_historical else "forecast",
        "hourly": hourly,
        "day_total": day_total,
        "solar_warnings": warnings,
        "uamps_available": uamps_day is not None,
    }


@app.get("/")
async def root(target_date: str = Query(None, alias="date")):
    if target_date is None:
        return FileResponse("static/index.html")
    if target_date == "today":
        target_date = date.today().strftime("%Y-%m-%d")
    elif target_date == "tomorrow":
        target_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        data = await forecast(target_date=target_date, fmt="json")
    except HTTPException as exc:
        return HTMLResponse(f"<p>Error: {exc.detail}</p>", status_code=exc.status_code)
    hourly = data["hourly"]
    rows = "\n".join(
        f"  <tr><td>{h['hour']}</td><td>{h['temp_f'] if h['temp_f'] is not None else ''}</td><td>{h['load'] if h['load'] is not None else ''}</td></tr>"
        for h in hourly
    )
    html = (
        "<!DOCTYPE html><html><body><table>\n"
        "<tr><th>Hour</th><th>Temp (°F)</th><th>Load (kW)</th></tr>\n"
        f"{rows}\n"
        "</table></body></html>"
    )
    return HTMLResponse(content=html)


@app.get("/api/uamps/test")
async def uamps_test():
    """Diagnose UAMPS connectivity — returns pretty-printed JSON."""
    user_id  = os.getenv("UAMPS_USER_ID", "")
    password = os.getenv("UAMPS_PASSWORD", "")
    if not user_id or not password:
        result = {"ok": False, "error": "UAMPS_USER_ID or UAMPS_PASSWORD not set in environment"}
    else:
        try:
            data = await asyncio.to_thread(_uamps_fetch_day_sync, user_id, password, date.today())
            result = {"ok": True, "hours": len(data), "sample": {str(k): v for k, v in list(data.items())[:3]}}
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
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


app.mount("/", StaticFiles(directory="static", html=True), name="static")
