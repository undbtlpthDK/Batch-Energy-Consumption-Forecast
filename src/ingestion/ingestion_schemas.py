import pandera.pandas as pa


def customer_metadata_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "customer_id": pa.Column(str),
            "customer_name": pa.Column(str),
            "object_id": pa.Column(str),
            "object_address": pa.Column(str),
            "metering_point_id": pa.Column(dtype="int64"),
            "tariff_code": pa.Column(str),
            "address_latitude": pa.Column(
                float, checks=pa.Check.in_range(-90, 90)
            ),
            "address_longitude": pa.Column(
                float, checks=pa.Check.in_range(0, 180)
            ),
            "region_id": pa.Column(int),
        }
    )


def smart_meter_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "timestamp": pa.Column("datetime"),
            "customer_id": pa.Column(str),
            "object_id": pa.Column(str, nullable=True),
            "metering_point_id": pa.Column(int, nullable=True),
            "energy_import_kwh": pa.Column(
                float, checks=pa.Check(lambda x: x >= 0)
            ),
            "energy_export_kwh": pa.Column(
                float, checks=pa.Check(lambda x: x >= 0)
            ),
            "tariff_code": pa.Column(str, nullable=True),
            "ingested_at": pa.Column("datetime"),
        }
    )


def weather_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "timestamp": pa.Column("datetime"),
            "temperature_2m": pa.Column("float32", nullable=True),
            "rain": pa.Column("float32", nullable=True),
            "snowfall": pa.Column("float32", nullable=True),
            "cloud_cover": pa.Column("float32", nullable=True),
            "weather_code": pa.Column("float32", nullable=True),
            "is_day": pa.Column("float32", nullable=True),
            "wind_speed_10m": pa.Column("float32", nullable=True),
            "relative_humidity_2m": pa.Column("float32", nullable=True),
            "apparent_temperature": pa.Column("float32", nullable=True),
            "precipitation": pa.Column("float32", nullable=True),
            "region_id": pa.Column(int),
            "data_type": pa.Column(str),
            "ingested_at": pa.Column("datetime"),
        }
    )


def region_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "center_latitude": pa.Column(
                float, checks=pa.Check.in_range(-90, 90)
            ),
            "center_longitude": pa.Column(
                float, checks=pa.Check.in_range(0, 180)
            ),
            "region_id": pa.Column(int),
        }
    )


SCHEMA_REGISTRY = {
    "customer_metadata": customer_metadata_schema,
    "smart_meter": smart_meter_schema,
    "weather": weather_schema,
    "region": region_schema,
}
