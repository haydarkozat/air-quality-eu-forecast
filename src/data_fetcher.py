"""
Air Quality Data Fetcher
========================
Fetches PM2.5, PM10, NO2, O3 data from OpenAQ v3 API for European cities.
Falls back to realistic synthetic data if no API key is provided,
so the project runs end-to-end out of the box.
"""

import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

# Major European cities with approximate coordinates
EU_CITIES = {
    "Berlin":     {"lat": 52.5200, "lon": 13.4050, "country": "DE"},
    "Paris":      {"lat": 48.8566, "lon": 2.3522,  "country": "FR"},
    "Madrid":     {"lat": 40.4168, "lon": -3.7038, "country": "ES"},
    "Rome":       {"lat": 41.9028, "lon": 12.4964, "country": "IT"},
    "Amsterdam":  {"lat": 52.3676, "lon": 4.9041,  "country": "NL"},
    "Warsaw":     {"lat": 52.2297, "lon": 21.0122, "country": "PL"},
    "Vienna":     {"lat": 48.2082, "lon": 16.3738, "country": "AT"},
    "Istanbul":   {"lat": 41.0082, "lon": 28.9784, "country": "TR"},
}

OPENAQ_BASE = "https://api.openaq.org/v3"


def fetch_openaq(city: str, days: int = 90, api_key: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Fetch real measurements from OpenAQ v3 for a given city.
    Returns None if no API key or if the request fails.
    """
    api_key = api_key or os.getenv("OPENAQ_API_KEY")
    if not api_key:
        return None

    if city not in EU_CITIES:
        raise ValueError(f"Unknown city: {city}. Available: {list(EU_CITIES.keys())}")

    coords = EU_CITIES[city]
    headers = {"X-API-Key": api_key}

    # Find nearby stations
    try:
        loc_resp = requests.get(
            f"{OPENAQ_BASE}/locations",
            params={
                "coordinates": f"{coords['lat']},{coords['lon']}",
                "radius": 25000,
                "limit": 10,
            },
            headers=headers,
            timeout=20,
        )
        loc_resp.raise_for_status()
        locations = loc_resp.json().get("results", [])
    except Exception as e:
        print(f"[OpenAQ] location lookup failed for {city}: {e}")
        return None

    if not locations:
        return None

    # Collect sensor IDs for parameters we care about
    target_params = {"pm25", "pm10", "no2", "o3"}
    sensor_map = {}  # sensor_id -> parameter name
    for loc in locations:
        for sensor in loc.get("sensors", []):
            pname = sensor.get("parameter", {}).get("name", "").lower()
            if pname in target_params:
                sensor_map[sensor["id"]] = pname

    if not sensor_map:
        return None

    # Pull measurements per sensor
    date_to = datetime.utcnow()
    date_from = date_to - timedelta(days=days)

    all_rows = []
    for sensor_id, param in list(sensor_map.items())[:8]:  # cap to avoid rate limit
        try:
            r = requests.get(
                f"{OPENAQ_BASE}/sensors/{sensor_id}/measurements",
                params={
                    "datetime_from": date_from.isoformat() + "Z",
                    "datetime_to": date_to.isoformat() + "Z",
                    "limit": 1000,
                },
                headers=headers,
                timeout=30,
            )
            r.raise_for_status()
            for m in r.json().get("results", []):
                all_rows.append({
                    "datetime": m["period"]["datetimeFrom"]["utc"],
                    "parameter": param,
                    "value": m["value"],
                })
            time.sleep(0.3)  # be polite
        except Exception as e:
            print(f"[OpenAQ] sensor {sensor_id} failed: {e}")
            continue

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    # Average values within the same hour & pivot
    df["datetime"] = df["datetime"].dt.floor("h")
    df = df.groupby(["datetime", "parameter"])["value"].mean().reset_index()
    df = df.pivot(index="datetime", columns="parameter", values="value").reset_index()
    df["city"] = city
    return df.sort_values("datetime").reset_index(drop=True)


def generate_synthetic(city: str, days: int = 90, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic hourly air quality data for a city.
    Encodes patterns observed in real European cities:
      - Diurnal cycle (rush hour peaks)
      - Weekly cycle (weekend dip)
      - Seasonal trend (worse in winter)
      - Weather coupling (low wind & temperature inversion -> higher PM)
      - City-specific baseline (Warsaw > Berlin > Amsterdam, etc.)
    """
    if city not in EU_CITIES:
        raise ValueError(f"Unknown city: {city}")

    rng = np.random.default_rng(seed + hash(city) % 10000)

    # City baseline PM2.5 (rough WHO-cited annual averages, μg/m³)
    baselines = {
        "Berlin": 12, "Paris": 14, "Madrid": 11, "Rome": 16,
        "Amsterdam": 10, "Warsaw": 22, "Vienna": 13, "Istanbul": 26,
    }
    base = baselines.get(city, 15)

    end = pd.Timestamp.utcnow().floor("h")
    start = end - pd.Timedelta(days=days)
    idx = pd.date_range(start, end, freq="h")
    n = len(idx)

    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    doy = idx.dayofyear.to_numpy()

    # Diurnal: morning + evening rush hour peaks
    diurnal = 3 * (np.exp(-((hour - 8) ** 2) / 6) + np.exp(-((hour - 19) ** 2) / 8))
    # Weekend dip
    weekend = np.where(dow >= 5, -2.0, 0.0)
    # Seasonal: winter peak (heating)
    seasonal = 6 * np.cos(2 * np.pi * (doy - 15) / 365)

    # Simulated weather
    temp = 12 + 12 * np.cos(2 * np.pi * (doy - 200) / 365) + rng.normal(0, 3, n)
    wind = np.clip(rng.gamma(2.0, 1.5, n), 0.3, None)  # m/s
    humidity = np.clip(60 + 15 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 8, n), 20, 100)

    # PM2.5: baseline + cycles + weather effect + noise
    pm25 = base + diurnal + weekend + seasonal \
           - 1.5 * wind \
           + 0.05 * (humidity - 60) \
           + rng.normal(0, 2.5, n)
    pm25 = np.clip(pm25, 1, None)

    # Correlated pollutants
    pm10 = pm25 * 1.6 + rng.normal(0, 3, n)
    pm10 = np.clip(pm10, 2, None)

    no2 = 18 + 8 * (np.exp(-((hour - 8) ** 2) / 4) + np.exp(-((hour - 18) ** 2) / 5)) \
          + np.where(dow >= 5, -4, 0) - 0.8 * wind + rng.normal(0, 4, n)
    no2 = np.clip(no2, 2, None)

    o3 = 45 + 25 * np.exp(-((hour - 15) ** 2) / 20) - 0.4 * no2 + 0.3 * temp + rng.normal(0, 5, n)
    o3 = np.clip(o3, 5, None)

    return pd.DataFrame({
        "datetime": idx,
        "city": city,
        "pm25": np.round(pm25, 2),
        "pm10": np.round(pm10, 2),
        "no2": np.round(no2, 2),
        "o3": np.round(o3, 2),
        "temperature": np.round(temp, 1),
        "wind_speed": np.round(wind, 2),
        "humidity": np.round(humidity, 1),
    })


def get_data(city: str, days: int = 90, prefer_real: bool = True) -> pd.DataFrame:
    """
    Main entry point. Tries OpenAQ first if prefer_real=True and an API key
    is available; otherwise returns synthetic data so the project always works.
    """
    if prefer_real:
        real = fetch_openaq(city, days=days)
        if real is not None and len(real) > 100:
            print(f"[data] Using real OpenAQ data for {city}: {len(real)} rows")
            return real
        print(f"[data] Falling back to synthetic data for {city}")
    return generate_synthetic(city, days=days)


if __name__ == "__main__":
    # Quick smoke test
    df = get_data("Berlin", days=30, prefer_real=False)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['datetime'].min()} -> {df['datetime'].max()}")
