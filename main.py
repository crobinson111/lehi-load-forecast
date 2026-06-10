import csv
import io
import os
import asyncio
import logging
import shutil
import tempfile
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import httpx
import pandas as pd
from datetime import datetime, timedelta, date
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
import holidays as holidays_lib
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Utah state holidays (includes federal + Pioneer Day Jul 24) through 2040
_HOLIDAYS = holidays_lib.US(state="UT", years=range(2018, 2041))

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

LEHI_LAT = 40.3916
LEHI_LON = -111.8508
EXCEL_PATH = os.environ.get(
    "EXCEL_PATH",
    r"C:\Users\crobinson\OneDrive - Lehi City\Scheduling - Documents\SchLogData.xlsx",
)

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


async def fetch_weather(start_date: str, end_date: str, use_forecast_api: bool = False) -> dict:
    """Returns {date_str: {openmeteo_hour_0_to_23: temp_f}}"""
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
    async with httpx.AsyncClient(timeout=120.0) as client:
        for attempt in range(4):
            resp = await client.get(base_url, params=params)
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
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

        logger.info("Fetching historical weather from Open-Meteo...")
        weather = await fetch_weather(train_df["date"].min(), train_df["date"].max())

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(train_model())
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


app.mount("/", StaticFiles(directory="static", html=True), name="static")
