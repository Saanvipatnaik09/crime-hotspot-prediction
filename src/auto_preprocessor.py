import pandas as pd

def auto_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()

    # --- detect datetime ---
    datetime_col = next(
        (c for c in df.columns if "date" in c or "time" in c),
        None
    )
    if datetime_col:
        df["datetime"] = pd.to_datetime(df[datetime_col], errors="coerce")

    # --- detect location name ---
    location_col = next(
        (c for c in df.columns if any(k in c for k in ["city", "district", "area", "location"])),
        None
    )
    if location_col:
        df["location_name"] = df[location_col].astype(str)

    # --- detect lat / lon ---
    lat_col = next((c for c in df.columns if "lat" in c), None)
    lon_col = next((c for c in df.columns if "lon" in c), None)

    if lat_col and lon_col:
        df["latitude"] = df[lat_col]
        df["longitude"] = df[lon_col]
    else:
        df["latitude"] = None
        df["longitude"] = None

    # --- detect crime type (optional) ---
    crime_col = next(
        (c for c in df.columns if "crime" in c or "offence" in c),
        None
    )
    if crime_col:
        df["crime_type"] = df[crime_col].astype(str)
    else:
        df["crime_type"] = "UNKNOWN"

    return df
