from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

DB_URL = (
    "postgresql+psycopg2://"
    "energy_user:energy_password"
    "@localhost:5432/energy_db"
)


def get_engine() -> Engine:
    return create_engine(
        DB_URL,
        pool_pre_ping=True,
    )


def truncate_table(table_name: str, engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {table_name} CASCADE;"))


def create_raw_tables(engine: Engine) -> None:
    """
    Create raw data tables.
    Safe to run multiple times.
    """

    ddl_statements = [

        # REGIONS
        """
        CREATE TABLE IF NOT EXISTS region_centers (
            region_id INTEGER PRIMARY KEY,
            center_latitude DOUBLE PRECISION,
            center_longitude DOUBLE PRECISION
        );
        """,

        # CUSTOMER METADATA

        """
        CREATE TABLE IF NOT EXISTS raw_customer_metadata (
            customer_id VARCHAR,
            customer_name VARCHAR,
            object_id VARCHAR PRIMARY KEY,
            object_address VARCHAR,
            metering_point_id BIGINT,
            tariff_code VARCHAR,
            address_latitude DOUBLE PRECISION,
            address_longitude DOUBLE PRECISION,
            region_id INTEGER,
            CONSTRAINT fk_customer_region
                FOREIGN KEY (region_id)
                REFERENCES region_centers (region_id)
        );
        """,


        # SMART METER READINGS

        """
        CREATE TABLE IF NOT EXISTS raw_smart_meter_readings (
            customer_id VARCHAR NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            object_id VARCHAR NOT NULL,
            metering_point_id BIGINT,
            tariff_code VARCHAR,
            energy_import_kwh DOUBLE PRECISION NOT NULL,
            energy_export_kwh DOUBLE PRECISION NOT NULL,
            ingested_at TIMESTAMP NOT NULL,

            CONSTRAINT pk_sm PRIMARY KEY (timestamp, object_id),
            CONSTRAINT fk_sm_customer
                FOREIGN KEY (object_id)
                REFERENCES raw_customer_metadata (object_id)
        );
        """,

        # WEATHER

        """
        CREATE TABLE IF NOT EXISTS raw_weather (
            timestamp TIMESTAMP NOT NULL,
            region_id INTEGER NOT NULL,
            data_type VARCHAR NOT NULL,

            temperature_2m REAL,
            rain REAL,
            snowfall REAL,
            cloud_cover REAL,
            weather_code INTEGER,
            is_day INTEGER,
            wind_speed_10m REAL,
            relative_humidity_2m REAL,
            apparent_temperature REAL,
            precipitation REAL,

            ingested_at TIMESTAMP NOT NULL,

            CONSTRAINT pk_weather PRIMARY KEY (timestamp, region_id, data_type),
            CONSTRAINT fk_weather_region
                FOREIGN KEY (region_id)
                REFERENCES region_centers (region_id)
        );
        """,
    ]

    with engine.begin() as conn:
        for ddl in ddl_statements:
            conn.execute(text(ddl))


# Data loading
def load_dataframe(
    df: pd.DataFrame,
    table_name: str,
    engine: Engine,
    if_exists: str = "append",
) -> None:
    """
    Load dataframe into an existing table.
    Assumes schema already exists.
    """

    df.to_sql(
        table_name,
        engine,
        if_exists=if_exists,
        index=False,
        method="multi",
        chunksize=10_000,
    )


def load_sm_metadata(engine: Engine) -> pd.DataFrame:
    """
    Load smart meter (customer/object) metadata from raw_customer_metadata.

    Returns
    -------
    pd.DataFrame
        One row per object_id with customer and location metadata.
    """
    query = """
        SELECT
            customer_id,
            customer_name,
            object_id,
            object_address,
            metering_point_id,
            tariff_code,
            address_latitude,
            address_longitude,
            region_id
        FROM raw_customer_metadata
    """

    return pd.read_sql(query, engine)


def _validate_at_date(at_date: str) -> None:
    """
    Validate datetime string format: YYYY-MM-DD HH:MM:SS
    """
    try:
        datetime.strptime(at_date, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        raise ValueError(
            "at_date must follow format: YYYY-MM-DD HH:MM:SS "
            "(e.g. 2017-04-01 00:00:00)"
        )


def load_sm_data(engine: Engine, at_date: str | None = None) -> pd.DataFrame:
    """
    Load smart meter readings data from raw_smart_meter_readings.

    If at_date is provided, only rows with ingested_at > at_date are returned.
    """

    base_query = """
        SELECT
            customer_id,
            timestamp,
            object_id,
            metering_point_id,
            tariff_code,
            energy_import_kwh,
            energy_export_kwh,
            ingested_at
        FROM raw_smart_meter_readings
    """

    params = None

    if at_date is not None:
        _validate_at_date(at_date)
        base_query += " WHERE ingested_at > %(at_date)s"
        params = {"at_date": at_date}

    return pd.read_sql(base_query, engine, params=params)


def load_weather_data(
    engine: Engine, 
    at_date: str | None = None
    ) -> pd.DataFrame:
    """
    Load weather data from raw_weather.

    If at_date is provided, only rows with ingested_at > at_date are returned.
    """

    base_query = """
        SELECT
            timestamp,
            region_id,
            data_type,
            temperature_2m,
            rain,
            snowfall,
            cloud_cover,
            weather_code,
            is_day,
            wind_speed_10m,
            relative_humidity_2m,
            apparent_temperature,
            precipitation,
            ingested_at
        FROM raw_weather
    """

    params = None

    if at_date is not None:
        _validate_at_date(at_date)
        base_query += " WHERE ingested_at > %(at_date)s"
        params = {"at_date": at_date}

    return pd.read_sql(base_query, engine, params=params)
