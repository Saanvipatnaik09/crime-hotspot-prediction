import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
import joblib
import h3

from src.auto_preprocessor import auto_preprocess
from src.spatial_resolver import resolve_locations
from src.utils import risk_statement

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Adaptive Crime Hotspot Intelligence System",
    layout="wide"
)

st.title("🧠 Adaptive Crime Hotspot Analysis & Prediction")
st.caption(
    "Dataset-agnostic • Country-agnostic • Spatially adaptive crime intelligence"
)

# ==================================================
# FILE UPLOAD
# ==================================================
uploaded = st.file_uploader(
    "Upload any crime dataset (CSV)",
    type=["csv"]
)

if not uploaded:
    st.stop()

# ==================================================
# AUTO PREPROCESSING
# ==================================================
df = auto_preprocess(uploaded)

# ==================================================
# SPATIAL RESOLUTION (GEOCODING IF NEEDED)
# ==================================================
df = resolve_locations(df)

# ==================================================
# DETECT MODE
# ==================================================
if df["latitude"].notna().sum() > 0:
    spatial_mode = "MAP"
elif "location_name" in df.columns:
    spatial_mode = "AREA"
else:
    spatial_mode = "TEMPORAL"

st.success(f"🔍 Detected Mode: {spatial_mode}")

# ==================================================
# LOAD PRE-TRAINED MODEL (FOR RISK SCORE)
# ==================================================
MODEL_PATH = "models/rf_hotspot_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ==================================================
# SIDEBAR INPUTS
# ==================================================
st.sidebar.header("🔧 Prediction Inputs")

hour = st.sidebar.slider("Hour of Day", 0, 23, 20)
day = st.sidebar.slider("Day of Month", 1, 31, 15)
month = st.sidebar.slider("Month", 1, 12, 6)

# ==================================================
# DATASET-AWARE MAP CENTER (IMPORTANT FIX)
# ==================================================
valid_coords = df[["latitude", "longitude"]].dropna()

if not valid_coords.empty:
    lat = valid_coords.iloc[0]["latitude"]
    lon = valid_coords.iloc[0]["longitude"]
else:
    # Global fallback (India center)
    lat, lon = 20.5937, 78.9629

# ==================================================
# ML RISK PREDICTION
# ==================================================
if spatial_mode != "TEMPORAL":
    X = pd.DataFrame(
        [[hour, day, month, lat, lon]],
        columns=["hour", "day", "month", "latitude", "longitude"]
    )

    X_scaled = scaler.transform(X)
    probability = model.predict_proba(X_scaled)[0][1]
else:
    probability = 0.5

# ==================================================
# VISUALIZATION
# ==================================================
st.subheader("📍 Crime Pattern Visualization")

# ---------------- MAP MODE ----------------
if spatial_mode == "MAP":
    st.info("🗺️ Hotspots generated dynamically from uploaded dataset")

    m = folium.Map(
        location=[lat, lon],
        zoom_start=11,
        tiles="cartodbpositron"
    )

    # --- Build H3 hotspots dynamically ---
    resolution = 8
    df_valid = df.dropna(subset=["latitude", "longitude"]).copy()

    df_valid["h3_cell"] = df_valid.apply(
        lambda r: h3.geo_to_h3(r["latitude"], r["longitude"], resolution),
        axis=1
    )

    h3_df = (
        df_valid.groupby("h3_cell")
        .size()
        .reset_index(name="crime_count")
    )

    # --- Heatmap ---
    heat_data = []
    for _, row in h3_df.iterrows():
        lat_, lon_ = h3.h3_to_geo(row["h3_cell"])
        heat_data.append([lat_, lon_, row["crime_count"]])

    HeatMap(
        heat_data,
        radius=14,
        blur=18,
        min_opacity=0.4
    ).add_to(m)

    # --- Top-N Hotspots ---
    top_hotspots = h3_df.sort_values(
        "crime_count", ascending=False
    ).head(20)

    for _, row in top_hotspots.iterrows():
        lat_, lon_ = h3.h3_to_geo(row["h3_cell"])
        folium.CircleMarker(
            location=[lat_, lon_],
            radius=8,
            color="red",
            fill=True,
            fill_opacity=0.85,
            popup=f"Crimes: {int(row['crime_count'])}"
        ).add_to(m)

    st.components.v1.html(m._repr_html_(), height=550)

# ---------------- AREA MODE ----------------
elif spatial_mode == "AREA":
    st.info("🗺️ District / Area-level risk visualization")

    grouped = (
        df.groupby("location_name")
          .agg({"latitude": "mean", "longitude": "mean"})
          .reset_index()
    )

    m = folium.Map(
        location=[lat, lon],
        zoom_start=5,
        tiles="cartodbpositron"
    )

    for _, row in grouped.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=10,
            color="orange",
            fill=True,
            fill_opacity=0.7,
            popup=row["location_name"]
        ).add_to(m)

    st.components.v1.html(m._repr_html_(), height=550)

# ---------------- TEMPORAL MODE ----------------
else:
    st.warning("⚠️ No spatial data available")
    st.write("Temporal crime pattern analysis only.")

# ==================================================
# RISK SUMMARY
# ==================================================
st.subheader("📊 Risk Assessment")

st.info(
    risk_statement(
        probability,
        location="Automatically Detected Region"
    )
)

# ==================================================
# SYSTEM EXPLANATION (VIVA READY)
# ==================================================
st.subheader("🧠 System Explanation")

st.markdown(f"""
**Detected Mode:** `{spatial_mode}`  

• **MAP** → Exact coordinates → H3-based hotspot detection  
• **AREA** → Place names → geocoded area-level analysis  
• **TEMPORAL** → No spatial info → time-based inference  

Hotspots are computed dynamically from the uploaded dataset,
ensuring city-specific and country-specific accuracy.
""")

# ==================================================
# FOOTER
# ==================================================
st.caption(
    "⚠️ Probabilistic decision-support system. "
    "Not deterministic crime prediction."
)
