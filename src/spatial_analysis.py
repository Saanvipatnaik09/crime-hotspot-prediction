import pandas as pd
import h3

def h3_to_latlon(h3_cell):
    return h3.h3_to_geo(h3_cell)

def load_h3_daily(path):
    """
    Expected columns:
    - h3_cell
    - crime_count
    """
    df = pd.read_csv(path)
    return df

def get_top_hotspots_h3(h3_df, top_n=20):
    return (
        h3_df
        .sort_values("crime_count", ascending=False)
        .head(top_n)
    )