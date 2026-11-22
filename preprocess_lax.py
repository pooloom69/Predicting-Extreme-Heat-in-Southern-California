"""
CPSC 483 — Data Preprocessing Script
Station: USW00023174 (Los Angeles International Airport)

Preprocessing Rules:
  1. Filter date range: 1950-01-01 ~ 2025-12-31
  2. Keep only PRCP, TMAX, TMIN (and their attributes)
  3. Parse quality flags → drop rows where quality flag is NOT empty
  4. Convert units: TMAX/TMIN from tenths of °C → °C, PRCP from tenths of mm → mm
  5. Use most recent lat/lon/elevation for consistency
  6. Drop rows missing BOTH TMAX and TMIN
  7. Output: station_id, station_name, date, latitude, longitude, elevation,
            tmax_c, tmin_c, prcp_mm
"""

import pandas as pd
import numpy as np

#CONFIG
INPUT_FILE  = "USW00023174.csv"           # raw NOAA file
OUTPUT_FILE = "USW00023174_cleaned.csv"         # cleaned output
STATION_ID  = "USW00023174"
DATE_START  = "1950-01-01"
DATE_END    = "2025-12-31"

# Load raw data
print("=" * 60)
print("STEP 0: Loading raw data")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)
print(f"  Raw rows: {len(df)}")
print(f"  Date range: {df['DATE'].min()} ~ {df['DATE'].max()}")

# Step 1: Filter date range
print("\nSTEP 1: Filter date range (1950–2025)")

df['DATE'] = pd.to_datetime(df['DATE'])
before = len(df)
df = df[(df['DATE'] >= DATE_START) & (df['DATE'] <= DATE_END)]
print(f"  Dropped {before - len(df)} rows outside range")
print(f"  Remaining: {len(df)} rows")

# Step 2: Keep only needed columns
print("\nSTEP 2: Select relevant columns")

cols_keep = [
    'STATION', 'DATE', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME',
    'PRCP', 'PRCP_ATTRIBUTES',
    'TMAX', 'TMAX_ATTRIBUTES',
    'TMIN', 'TMIN_ATTRIBUTES',
]
df = df[cols_keep].copy()
print(f"  Kept {len(cols_keep)} columns")

# Step 3: Parse quality flags & drop flagged rows
# NOAA attribute format: "measurement_flag,quality_flag,source_code,obs_time"
# quality_flag = second field. Empty = passed QC. Any letter = flagged.
print("\nSTEP 3: Parse quality flags")

def has_quality_flag(attr_value):
    """Return True if the quality flag (2nd field) is NOT empty."""
    if pd.isna(attr_value):
        return False  # no attribute string → no flag
    parts = str(attr_value).split(',')
    if len(parts) >= 2 and parts[1].strip() != '':
        return True
    return False

# Check each variable
for var in ['PRCP', 'TMAX', 'TMIN']:
    attr_col = f'{var}_ATTRIBUTES'
    flagged = df[attr_col].apply(has_quality_flag)
    n_flagged = flagged.sum()
    if n_flagged > 0:
        # Show which flags
        flagged_attrs = df.loc[flagged, attr_col].unique()
        print(f"  {var}: {n_flagged} rows flagged → setting {var} to NaN")
        print(f"    Flag values: {flagged_attrs}")
        df.loc[flagged, var] = np.nan
    else:
        print(f"  {var}: 0 rows flagged ✓")

# Step 4: Convert units
print("\nSTEP 4: Convert units")

# TMAX, TMIN: tenths of °C → °C
df['tmax_c'] = df['TMAX'] / 10.0
df['tmin_c'] = df['TMIN'] / 10.0
print(f"  TMAX → tmax_c (tenths of °C → °C)")
print(f"  TMIN → tmin_c (tenths of °C → °C)")

# PRCP: tenths of mm → mm
df['prcp_mm'] = df['PRCP'] / 10.0
print(f"  PRCP → prcp_mm (tenths of mm → mm)")

#Step 5: Lat/Lon/Elevation → use most recent values
print("\nSTEP 5: Standardize lat/lon/elevation (most recent values)")

latest_row = df.loc[df['DATE'].idxmax()]
lat = latest_row['LATITUDE']
lon = latest_row['LONGITUDE']
elev = latest_row['ELEVATION']
print(f"  Using: lat={lat}, lon={lon}, elev={elev}")

df['latitude']  = lat
df['longitude'] = lon
df['elevation'] = elev

# Step 6: Drop rows missing BOTH TMAX and TMIN
print("\nSTEP 6: Drop rows where BOTH tmax and tmin are missing")

before = len(df)
df = df.dropna(subset=['tmax_c', 'tmin_c'], how='all')
print(f"  Dropped {before - len(df)} rows")
print(f"  Remaining: {len(df)} rows")

# Step 7: Build final output
print("\nSTEP 7: Build final output")

output = pd.DataFrame({
    'station_id':   STATION_ID,
    'station_name': 'LOS ANGELES INTL AP',
    'date':         df['DATE'].dt.strftime('%Y-%m-%d'),
    'latitude':     df['latitude'],
    'longitude':    df['longitude'],
    'elevation':    df['elevation'],
    'tmax_c':       df['tmax_c'].round(1),
    'tmin_c':       df['tmin_c'].round(1),
    'prcp_mm':      df['prcp_mm'].round(1),
})

output.to_csv(OUTPUT_FILE, index=False)
print(f"  Saved to: {OUTPUT_FILE}")
print(f"  Final rows: {len(output)}")
print(f"  Final columns: {list(output.columns)}")

# ── SUMMARY ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Date range: {output['date'].min()} ~ {output['date'].max()}")
print(f"  Total rows: {len(output)}")
print(f"  tmax_c — missing: {output['tmax_c'].isna().sum()}, "
      f"min: {output['tmax_c'].min()}, max: {output['tmax_c'].max()}")
print(f"  tmin_c — missing: {output['tmin_c'].isna().sum()}, "
      f"min: {output['tmin_c'].min()}, max: {output['tmin_c'].max()}")
print(f"  prcp_mm — missing: {output['prcp_mm'].isna().sum()}, "
      f"min: {output['prcp_mm'].min()}, max: {output['prcp_mm'].max()}")
print()
print("  Sample (first 5 rows):")
print(output.head().to_string(index=False))
