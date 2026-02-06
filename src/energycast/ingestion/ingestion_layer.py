from db_utils import create_raw_tables, get_engine
from schemas import SCHEMA_REGISTRY
from utils import (add_ingestion_time, check_duplicates, check_time_continuity,
                   handle_duplicates, pull_meta_data, pull_regions_data,
                   pull_smart_meter_data, pull_weather_data,
                   timestamp_to_datetime)

df_id = pull_meta_data()
df_regions = pull_regions_data()
df_sm = pull_smart_meter_data()
df_weather = pull_weather_data('backtest', df_regions)

df_sm = timestamp_to_datetime(df_sm)
df_weather = timestamp_to_datetime(df_weather)  # type: ignore

sm_duplicated_row_amount = check_duplicates(df_sm, pk="customer_id")
if sm_duplicated_row_amount > 0:
    df_sm = handle_duplicates(df_sm, "smart_meter")
    sm_duplicated_row_amount = check_duplicates(df_sm, pk="customer_id")
    try:
        assert sm_duplicated_row_amount == 0
        print("duplicates removal works")
    except ValueError:
        print("Duplicated smart meters reading")

check_time_continuity(df_weather, id_col="region_id", expected_step_hours=1)
check_time_continuity(df_sm, id_col="object_id", expected_step_hours=1)

df_sm = add_ingestion_time(df_sm, "smart_meter_backtest", forecast_horizon=0)
df_weather = add_ingestion_time(df_weather, mode="weather", forecast_horizon=7)

meta_schema = SCHEMA_REGISTRY['customer_metadata']()
sm_schema = SCHEMA_REGISTRY['smart_meter']()
weather_schema = SCHEMA_REGISTRY['weather']()
region_schema = SCHEMA_REGISTRY['region']()

meta_schema(df_id)
sm_schema(df_sm)
weather_schema(df_weather)
region_schema(df_regions)

dfs = [df_id, df_regions, df_sm, df_weather]
for df in dfs:
    print(df.info())

engine = get_engine()
create_raw_tables(engine)

# Code bellow is dropping the content in db, so needed to be updated
"""
truncate_table("region_centers", engine)
load_dataframe(df_regions, "region_centers", engine)

truncate_table("raw_customer_metadata", engine)
load_dataframe(df_id, "raw_customer_metadata", engine)

truncate_table("raw_smart_meter_readings", engine)
load_dataframe(df_sm, "raw_smart_meter_readings", engine)

truncate_table("raw_weather", engine)
load_dataframe(df_weather, "raw_weather", engine)
"""
