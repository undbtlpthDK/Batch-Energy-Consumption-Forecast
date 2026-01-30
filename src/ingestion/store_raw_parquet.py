from db_utils import (get_engine, load_sm_data, load_sm_metadata,
                      load_weather_data)
from utils import (write_sm_metadata_raw, write_sm_readings_raw,
                   write_weather_raw)

engine = get_engine()

df_id = load_sm_metadata(engine)
write_sm_metadata_raw(df_id)

df_weather = load_weather_data(engine)
write_weather_raw(df_weather)

df_sm = load_sm_data(engine)
write_sm_readings_raw(df_sm)
