"""
CPSC 483 — Palm Springs Station Merge
OLD: USC00046635 (PALM SPRINGS)       — temperature ends 2014-08-12
NEW: USW00093138 (PALM SPRINGS ASOS)  — 1998-02-19 ~ present
The two stations are ~0.8 km apart (elevation 129.5 m vs 124.7 m).

Merge Rules:
  1. Clean both stations with the same rules as preprocess_lax.py
     (date filter 1950–2025, quality flag -> NaN, unit conversion)
  2. Bias check on the overlap period (1998–2014):
       TMAX: NEW is ~0.67°C cooler than OLD  -> adjust OLD TMAX
       TMIN: difference ~0.05°C              -> no adjustment
       PRCP: no systematic difference        -> no adjustment
  3. Adjustment = monthly mean difference (NEW - OLD) of TMAX,
     added to OLD TMAX, so the whole series matches the current ASOS station
  4. For every date and every variable:
       use NEW if available, otherwise use OLD (adjusted)
  5. Use NEW station lat/lon/elevation (current location)
  6. Drop rows where BOTH tmax and tmin are missing
  7. Output: station_id, station_name, date, latitude, longitude, elevation,
            tmax_c, tmin_c, prcp_mm
     (+ source_tmax / source_tmin / source_prcp for tracking which station
      each value came from; drop these before combining if not needed)
"""

import pandas as pd
import numpy as np

# CONFIG
OLD_FILE    = "USC00046635_Palm_previous.csv"
NEW_FILE    = "USW00093138_PalmSprings.csv"
OUTPUT_FILE = "PalmSprings_merged_cleaned.csv"
STATION_ID   = "USC00046635+USW00093138"
STATION_NAME = "PALM SPRINGS (MERGED)"
DATE_START  = "1950-01-01"
DATE_END    = "2025-12-31"
ADJUST_TMAX = True     # set False to merge without bias correction


# Helper: same cleaning rules as preprocess_lax.py
def has_quality_flag(attr_value):
    """Return True if the quality flag (2nd field) is NOT empty."""
    if pd.isna(attr_value):
        return False
    parts = str(attr_value).split(',')
    return len(parts) >= 2 and parts[1].strip() != ''


def clean_station(path, label):
    print(f"\n--- Cleaning {label}: {path}")
    df = pd.read_csv(path, low_memory=False)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df[(df['DATE'] >= DATE_START) & (df['DATE'] <= DATE_END)].copy()
    print(f"  Rows in 1950–2025: {len(df)}")

    for var in ['PRCP', 'TMAX', 'TMIN']:
        flagged = df[f'{var}_ATTRIBUTES'].apply(has_quality_flag)
        if flagged.sum() > 0:
            print(f"  {var}: {flagged.sum()} rows quality-flagged -> NaN")
        df.loc[flagged, var] = np.nan

    out = pd.DataFrame({
        'date':    df['DATE'],
        'tmax_c':  df['TMAX'] / 10.0,
        'tmin_c':  df['TMIN'] / 10.0,
        'prcp_mm': df['PRCP'] / 10.0,
    })
    latest = df.loc[df['DATE'].idxmax()]
    meta = (latest['LATITUDE'], latest['LONGITUDE'], latest['ELEVATION'])
    return out, meta


# STEP 1: Clean both stations
print("=" * 60)
print("STEP 1: Clean both stations")
print("=" * 60)
old, _        = clean_station(OLD_FILE, "OLD (USC00046635)")
new, new_meta = clean_station(NEW_FILE, "NEW (USW00093138)")

# STEP 2: Bias check on overlap period
print("\n" + "=" * 60)
print("STEP 2: Bias check on overlap period")
print("=" * 60)

ov = old.merge(new, on='date', suffixes=('_old', '_new'))
for v in ['tmax_c', 'tmin_c', 'prcp_mm']:
    both = ov.dropna(subset=[f'{v}_old', f'{v}_new'])
    d = both[f'{v}_new'] - both[f'{v}_old']
    print(f"  {v}: n={len(both)}, mean diff (new-old) = {d.mean():+.2f}, "
          f"corr = {both[f'{v}_old'].corr(both[f'{v}_new']):.3f}")

# STEP 3: Monthly TMAX adjustment for OLD station
print("\n" + "=" * 60)
print("STEP 3: TMAX adjustment (monthly mean difference)")
print("=" * 60)

if ADJUST_TMAX:
    both = ov.dropna(subset=['tmax_c_old', 'tmax_c_new'])
    monthly_bias = (both['tmax_c_new'] - both['tmax_c_old']) \
        .groupby(both['date'].dt.month).mean()
    print("  Monthly offset added to OLD TMAX (°C):")
    for m, b in monthly_bias.items():
        print(f"    month {m:2d}: {b:+.2f}")
    old['tmax_c'] = old['tmax_c'] + old['date'].dt.month.map(monthly_bias)
else:
    print("  Skipped (ADJUST_TMAX = False)")

# STEP 4: Merge — NEW first, fill gaps with OLD
print("\n" + "=" * 60)
print("STEP 4: Merge (NEW preferred, OLD fills gaps)")
print("=" * 60)

all_dates = pd.DataFrame({'date': pd.date_range(DATE_START, DATE_END, freq='D')})
m = all_dates.merge(old, on='date', how='left') \
             .merge(new, on='date', how='left', suffixes=('_old', '_new'))

merged = pd.DataFrame({'date': m['date']})
for v, tag in [('tmax_c', 'tmax'), ('tmin_c', 'tmin'), ('prcp_mm', 'prcp')]:
    use_new = m[f'{v}_new'].notna()
    use_old = ~use_new & m[f'{v}_old'].notna()
    merged[v] = np.where(use_new, m[f'{v}_new'], m[f'{v}_old'])
    merged[f'source_{tag}'] = np.select([use_new, use_old],
                                        ['USW00093138', 'USC00046635'], default='')
    print(f"  {v}: {use_new.sum()} days from NEW, {use_old.sum()} days from OLD, "
          f"{(~use_new & ~use_old).sum()} days missing")

# STEP 5–6: Metadata + drop rows missing both temps
print("\n" + "=" * 60)
print("STEP 5-6: Metadata & drop rows missing both TMAX and TMIN")
print("=" * 60)

before = len(merged)
merged = merged.dropna(subset=['tmax_c', 'tmin_c'], how='all')
print(f"  Dropped {before - len(merged)} rows (no temperature from either station)")
print(f"  Using NEW station location: lat={new_meta[0]}, lon={new_meta[1]}, elev={new_meta[2]}")

# STEP 7: Final output
output = pd.DataFrame({
    'station_id':   STATION_ID,
    'station_name': STATION_NAME,
    'date':         merged['date'].dt.strftime('%Y-%m-%d'),
    'latitude':     new_meta[0],
    'longitude':    new_meta[1],
    'elevation':    new_meta[2],
    'tmax_c':       merged['tmax_c'].round(1),
    'tmin_c':       merged['tmin_c'].round(1),
    'prcp_mm':      merged['prcp_mm'].round(1),
    'source_tmax':  merged['source_tmax'],
    'source_tmin':  merged['source_tmin'],
    'source_prcp':  merged['source_prcp'],
})
output.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Saved to: {OUTPUT_FILE}")
print(f"  Date range: {output['date'].min()} ~ {output['date'].max()}")
print(f"  Total rows: {len(output)}")
for v in ['tmax_c', 'tmin_c', 'prcp_mm']:
    print(f"  {v} — missing: {output[v].isna().sum()}, "
          f"min: {output[v].min()}, max: {output[v].max()}")