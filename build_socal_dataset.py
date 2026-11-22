"""
CPSC 483 — Build final SoCal dataset (5 stations, 1950–2025)

Input
  SOCal_Data_Preprocessed.csv   : current combined file (LAX, LA Downtown,
                                  Riverside Fire Sta 3, Palm Springs old)
  USW00023119_Riverside.csv      : March AFB raw NOAA file
  USW00093138_PalmSprings.csv    : Palm Springs ASOS raw NOAA file
  USC00047888_santaana.csv       : Santa Ana Fire Station raw NOAA file

What it does
  - LAX, LA Downtown       : kept as-is
  - Riverside              : Fire Sta 3 (USC00047470, ends 2016) merged with
                             March AFB (USW00023119, 1950–1970 + 2011–2025)
  - Palm Springs           : old station (USC00046635, ends 2014) merged with
                             ASOS (USW00093138, 1998–2025)
  - Santa Ana (Orange Co.) : USC00047888 added from raw file, no merge
                             (temperature continuous 1950–2025)

Merge rule (same for both stations)
  1. Clean the new raw file with the team rules
     (1950–2025, quality flag -> NaN, tenths -> °C / mm)
  2. Compare the two stations on overlapping days. Where the mean difference
     is larger than BIAS_THRESHOLD (°C), shift the OLD station by the monthly
     mean difference so the whole series matches the NEW (current) station.
     PRCP is never adjusted.
  3. For each day and each variable: NEW value if available, otherwise OLD.
  4. If TMAX < TMIN after merging (can happen from the adjustment), set both
     to NaN. Lat/lon/elevation = NEW station. Drop days with no TMAX and no TMIN.

Output: SOCal_Data_Final.csv with the same columns as the input file
  STATION, NAME, DATE, LATITUDE, LONGITUDE, ELEVATION, PRCP_MM, TMAX_C, TMIN_C
"""

import pandas as pd
import numpy as np

# CONFIG
COMBINED_FILE = "SOCal_Data_Preprocessed.csv"
OUTPUT_FILE   = "SOCal_Data_Final.csv"
DATE_START, DATE_END = "1950-01-01", "2025-12-31"
BIAS_THRESHOLD = 0.3     # °C; adjust a variable only if |mean diff| > this

MERGES = [
    dict(name="RIVERSIDE (MERGED: FIRE STA 3 + MARCH AFB), CA US",
         station_id="USC00047470+USW00023119",
         old_id="USC00047470",
         new_raw="USW00023119_Riverside.csv"),
    dict(name="PALM SPRINGS (MERGED: COOP + ASOS), CA US",
         station_id="USC00046635+USW00093138",
         old_id="USC00046635",
         new_raw="USW00093138_PalmSprings.csv"),
]
# Stations added as-is from a raw NOAA file (no merge needed)
ADDS = [
    dict(name="SANTA ANA FIRE STATION, CA US",
         station_id="USC00047888",
         raw="USC00047888_santaana.csv"),
]
COLS = ['STATION', 'NAME', 'DATE', 'LATITUDE', 'LONGITUDE', 'ELEVATION',
        'PRCP_MM', 'TMAX_C', 'TMIN_C']
VARS = ['TMAX_C', 'TMIN_C', 'PRCP_MM']


def has_quality_flag(attr_value):
    """True if the quality flag (2nd field of the attribute string) is not empty."""
    if pd.isna(attr_value):
        return False
    parts = str(attr_value).split(',')
    return len(parts) >= 2 and parts[1].strip() != ''


def clean_raw(path):
    """Apply the team preprocessing rules to a raw NOAA GHCN-Daily CSV."""
    df = pd.read_csv(path, low_memory=False)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df[(df['DATE'] >= DATE_START) & (df['DATE'] <= DATE_END)].copy()
    for var in ['PRCP', 'TMAX', 'TMIN']:
        flagged = df[f'{var}_ATTRIBUTES'].apply(has_quality_flag)
        df.loc[flagged, var] = np.nan
        if flagged.sum():
            print(f"    {var}: {flagged.sum()} quality-flagged values -> NaN")
    latest = df.loc[df['DATE'].idxmax()]
    meta = dict(LATITUDE=latest['LATITUDE'], LONGITUDE=latest['LONGITUDE'],
                ELEVATION=latest['ELEVATION'])
    out = pd.DataFrame({'DATE': df['DATE'],
                        'TMAX_C': df['TMAX'] / 10.0,
                        'TMIN_C': df['TMIN'] / 10.0,
                        'PRCP_MM': df['PRCP'] / 10.0})
    return out, meta


def merge_station(old, new, new_meta, cfg):
    print(f"\n{'=' * 64}\n{cfg['name']}\n{'=' * 64}")
    old = old.copy()

    # Bias check + monthly adjustment of OLD
    ov = old.merge(new, on='DATE', suffixes=('_old', '_new'))
    print(f"  Overlap days: {len(ov)}")
    for v in VARS:
        both = ov.dropna(subset=[f'{v}_old', f'{v}_new'])
        d = both[f'{v}_new'] - both[f'{v}_old']
        adjust = v != 'PRCP_MM' and abs(d.mean()) > BIAS_THRESHOLD
        print(f"  {v}: n={len(both)}, mean diff (new-old) {d.mean():+.2f}, "
              f"corr {both[f'{v}_old'].corr(both[f'{v}_new']):.3f}"
              f"  -> {'ADJUST old' if adjust else 'no adjustment'}")
        if adjust:
            monthly = d.groupby(both['DATE'].dt.month).mean()
            print("     monthly offset:", {k: round(x, 2) for k, x in monthly.items()})
            old[v] = old[v] + old['DATE'].dt.month.map(monthly)

    # NEW first, OLD fills gaps
    days = pd.DataFrame({'DATE': pd.date_range(DATE_START, DATE_END, freq='D')})
    m = days.merge(old, on='DATE', how='left') \
            .merge(new, on='DATE', how='left', suffixes=('_old', '_new'))
    out = pd.DataFrame({'DATE': m['DATE']})
    for v in VARS:
        use_new = m[f'{v}_new'].notna()
        out[v] = np.where(use_new, m[f'{v}_new'], m[f'{v}_old'])
        print(f"  {v}: {use_new.sum()} days from NEW, "
              f"{(~use_new & m[f'{v}_old'].notna()).sum()} from OLD, "
              f"{out[v].isna().sum()} missing")

    # Sanity check: adjustment can rarely push TMAX below TMIN -> set both NaN
    bad = out['TMAX_C'] < out['TMIN_C']
    out.loc[bad, ['TMAX_C', 'TMIN_C']] = np.nan
    print(f"  TMAX < TMIN after merge: {bad.sum()} days -> set to NaN")

    before = len(out)
    out = out.dropna(subset=['TMAX_C', 'TMIN_C'], how='all')
    print(f"  Dropped {before - len(out)} days with no temperature")

    out['STATION'] = cfg['station_id']
    out['NAME'] = cfg['name']
    for k, val in new_meta.items():
        out[k] = val
    return out


# MAIN
print("Loading combined file ...")
combined = pd.read_csv(COMBINED_FILE, low_memory=False)
combined['DATE'] = pd.to_datetime(combined['DATE'])
print(combined.groupby('STATION').size().to_string())

parts = []
replaced = [cfg['old_id'] for cfg in MERGES]
parts.append(combined[~combined['STATION'].isin(replaced)])   # LAX, LA Downtown

for cfg in MERGES:
    print(f"\nCleaning raw file {cfg['new_raw']}")
    new, new_meta = clean_raw(cfg['new_raw'])
    old = combined.loc[combined['STATION'] == cfg['old_id'], ['DATE'] + VARS]
    parts.append(merge_station(old, new, new_meta, cfg))

for cfg in ADDS:
    print(f"\n{'=' * 64}\n{cfg['name']}\n{'=' * 64}")
    print(f"Cleaning raw file {cfg['raw']}")
    df, meta = clean_raw(cfg['raw'])
    before = len(df)
    df = df.dropna(subset=['TMAX_C', 'TMIN_C'], how='all')
    print(f"  Dropped {before - len(df)} days with no temperature")
    df['STATION'] = cfg['station_id']
    df['NAME'] = cfg['name']
    for k, val in meta.items():
        df[k] = val
    parts.append(df)

final = pd.concat(parts, ignore_index=True)[COLS]
final = final.sort_values(['STATION', 'DATE']).reset_index(drop=True)
for v in VARS:
    final[v] = final[v].round(1)
final['DATE'] = final['DATE'].dt.strftime('%Y-%m-%d')
final.to_csv(OUTPUT_FILE, index=False)

print(f"\n{'=' * 64}\nFINAL: {OUTPUT_FILE}  ({len(final)} rows)\n{'=' * 64}")
summary = final.groupby(['STATION', 'NAME']).agg(
    start=('DATE', 'min'), end=('DATE', 'max'), rows=('DATE', 'size'),
    tmax_na=('TMAX_C', lambda s: s.isna().sum()),
    tmin_na=('TMIN_C', lambda s: s.isna().sum()),
    prcp_na=('PRCP_MM', lambda s: s.isna().sum()))
print(summary.to_string())