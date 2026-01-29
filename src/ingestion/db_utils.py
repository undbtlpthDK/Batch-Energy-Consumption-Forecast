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
            customer_id VARCHAR PRIMARY KEY,
            customer_name VARCHAR,
            object_id VARCHAR,
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
            object_id VARCHAR,
            metering_point_id BIGINT,
            tariff_code VARCHAR,
            energy_import_kwh DOUBLE PRECISION NOT NULL,
            energy_export_kwh DOUBLE PRECISION NOT NULL,
            ingested_at TIMESTAMP NOT NULL,

            CONSTRAINT pk_sm PRIMARY KEY (timestamp, customer_id),
            CONSTRAINT fk_sm_customer
                FOREIGN KEY (customer_id)
                REFERENCES raw_customer_metadata (customer_id)
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
