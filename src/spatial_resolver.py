from geopy.geocoders import Nominatim
import pandas as pd
import os

geolocator = Nominatim(user_agent="crime_intel_app")

CACHE_FILE = "data/geocode_cache.csv"

def load_cache():
    if os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE)
    return pd.DataFrame(columns=["location_name", "latitude", "longitude"])

def save_cache(cache):
    cache.to_csv(CACHE_FILE, index=False)

def resolve_locations(df):
    cache = load_cache()

    for i, row in df.iterrows():
        if pd.notna(row["latitude"]):
            continue

        loc_name = row.get("location_name")
        if not loc_name:
            continue

        cached = cache[cache["location_name"] == loc_name]
        if not cached.empty:
            df.at[i, "latitude"] = cached.iloc[0]["latitude"]
            df.at[i, "longitude"] = cached.iloc[0]["longitude"]
            continue

        try:
            loc = geolocator.geocode(loc_name)
            if loc:
                df.at[i, "latitude"] = loc.latitude
                df.at[i, "longitude"] = loc.longitude
                cache = pd.concat([
                    cache,
                    pd.DataFrame([{
                        "location_name": loc_name,
                        "latitude": loc.latitude,
                        "longitude": loc.longitude
                    }])
                ])
        except:
            pass

    save_cache(cache)
    return df
