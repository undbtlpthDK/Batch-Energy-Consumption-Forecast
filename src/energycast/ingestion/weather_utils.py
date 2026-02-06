from typing import List

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = [
    "temperature_2m",
    "rain",
    "snowfall",
    "cloud_cover",
    "weather_code",
    "is_day",
    "wind_speed_10m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
]


def fetch_weather(
    latitude: float,
    longitude: float,
    mode: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:

    if mode == "historical":
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date required for historical mode")

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": HOURLY_VARS,
            "timezone": "Europe/Riga",
        }

    elif mode == "forecast":
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": HOURLY_VARS,
            "timezone": "Europe/Riga",
        }

    else:
        raise ValueError("mode must be 'historical' or 'forecast'")

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()

    time_index = pd.date_range(
        start=pd.to_datetime(
            hourly.Time() + response.UtcOffsetSeconds(), unit="s", utc=True
        ),
        end=pd.to_datetime(
            hourly.TimeEnd() + response.UtcOffsetSeconds(), unit="s", utc=True
        ),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    data = {"timestamp": time_index}

    for i, var in enumerate(HOURLY_VARS):
        data[var] = hourly.Variables(i).ValuesAsNumpy()

    return pd.DataFrame(data)


def fetch_and_save_weather_for_regions(
    mode: str,
    regions: pd.DataFrame,
    output_dir: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    forecast_horizon_hours: int | None = None,
) -> List[pd.DataFrame]:

    region_data = []

    for _, row in regions.iterrows():
        region_id = int(row["region_id"])

        df = fetch_weather(
            latitude=row["center_latitude"],
            longitude=row["center_longitude"],
            mode=mode,
            start_date=start_date,
            end_date=end_date,
        )

        df["region_id"] = region_id
        df["data_type"] = mode

        if mode == "forecast":
            if forecast_horizon_hours is not None:
                df = df.iloc[:forecast_horizon_hours]

            forecast_start_ts = df["timestamp"].iloc[0]
            forecast_start_str = forecast_start_ts.strftime("%Y%m%dT%H")

            filename = (
                f"weather_forecast_region_{region_id}"
                f"_from_{forecast_start_str}"
                f"_h{forecast_horizon_hours}.csv"
            )

        else:
            filename = f"weather_historical_region_{region_id}.csv"

        region_data.append(df)

        if output_dir is not None:
            output_path = f"{output_dir}/{filename}"
            df.to_csv(output_path, index=False)
            print(f"Saved {mode} weather for region {region_id} → {filename}")

    return region_data
