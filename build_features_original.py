"""
CPSC 483 — Feature engineering for Task A and Task B

Input
  SOCal_Data_Final.csv : cleaned daily data (5 stations, 1950–2025)
  oni.csv              : NOAA CPC ONI v6, one row per year
                         Year,DJF,JFM,FMA,MAM,AMJ,MJJ,JJA,JAS,ASO,SON,OND,NDJ

Output
  TaskA_features.csv : one row per station per day (May–October)
  TaskB_features.csv : one row per station per year
  thresholds.csv     : extreme heat thresholds and station codes (used by both tasks)
  feature_list.csv   : description of every column in the two feature files

Extreme heat
  - Extreme heat day   : daily high > the station's 95th percentile high
  - Extreme heat night : daily low  > the station's 95th percentile low
  - The 95th percentiles use May–October of 1961–1990 only.

Station
  - station_code: 0 = LAX, 1 = LA_Downtown, 2 = Santa_Ana, 3 = Riverside, 4 = Palm_Springs
    (ordered from the coast to the desert).
  - The numbers are only labels. Random Forest can use station_code as is.
    For Logistic Regression and Linear Regression, turn it into five 0/1 columns
    in the training code, e.g. pd.get_dummies(X, columns=['station_code']).

Feature selection
  - Any two features with correlation of 0.4 or more (or -0.4 or less)
    were not kept together. Features more related to the answers were kept first.
  - Checked on 1950–2015 data only. station_code was not part of this check.

Task A (one row per station per day, predict yes/no)
  - Every weather input uses data up to the day before. Same-day values are never used.
  - Low temperature the day before.
  - High-low difference the day before.
  - Time of year: day number counted from January 1 (dayOfYear).
  - Rain: rain the day before, total rain over the past 30 days,
    and days since it last rained (1 mm or more).
  - El Niño / La Niña: ONI value from the 3 months that ended the month before
    (e.g. July uses April–June).
  - Station: station_code.
  - Answers: hot_day (extreme heat day? 1/0), hot_night (extreme heat night? 1/0).

Task B (one row per station per year, predict a number)
  - Answers: number of extreme heat days and nights from May to October.
  - A year is used only if at least 90% of its May–October days have
    both high and low temperatures.
  - Inputs come only from earlier years, never from the same year:
    last year's extreme heat nights, last year's average low,
    average extreme heat days over the past 5 years,
    last year's longest run of extreme heat days.
    Also includes the year and station_code.
"""

import numpy as np
import pandas as pd

# CONFIG
INPUT_FILE    = "SOCal_Data_Final.csv"
ONI_FILE      = "oni.csv"
TASK_A_FILE   = "TaskA_features.csv"
TASK_B_FILE   = "TaskB_features.csv"
THRESH_FILE   = "thresholds.csv"
FEATLIST_FILE = "feature_list.csv"

REF_START, REF_END = 1961, 1990        # reference period for thresholds
PERCENTILE    = 0.95
WARM_MONTHS   = [5, 6, 7, 8, 9, 10]
WARM_DAYS     = 184                    # days from May 1 to Oct 31
RAIN_DAY_MM   = 1.0                    # a "rain day" for days_since_rain
MIN_FRAC      = 0.8                    # 30-day rain total: share of days with data
B_MIN_COVERAGE = 0.9                   # Task B: share of May–Oct days with data
B_AVG_YEARS   = 5                      # Task B: years in the multi-year average
B_AVG_MIN     = 3                      # Task B: valid years needed for that average
CORR_LIMIT    = 0.4                    # feature correlation limit (for the check)
CHECK_END_YEAR = 2015                  # correlation check uses data up to this year

# station id (substring) -> (short name, station_code), coast to desert
STATIONS = {
    '23174': ('LAX', 0),
    '93134': ('LA_Downtown', 1),
    '47888': ('Santa_Ana', 2),
    '47470': ('Riverside', 3),
    '46635': ('Palm_Springs', 4),
}
# ONI season -> its center month (DJF of year Y is centered on Jan of Y)
SEASON_CENTER = {'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6,
                 'JJA': 7, 'JAS': 8, 'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12}


def station_info(station_id):
    """Return (short name, station_code) for a NOAA station id."""
    for key, info in STATIONS.items():
        if key in station_id:
            return info
    raise ValueError(f"Unknown station: {station_id}")


def check_correlation(data, features, label):
    """Print the largest correlation between any two selected features."""
    c = data[features].corr().abs().to_numpy(copy=True)
    np.fill_diagonal(c, 0)
    i, j = np.unravel_index(np.nanargmax(c), c.shape)
    status = "OK" if c[i, j] < CORR_LIMIT else "ABOVE LIMIT"
    print(f"  {label}: largest feature correlation = {c[i, j]:.2f} "
          f"({features[i]} vs {features[j]}) -> {status}")


# PART 1 — Shared: load data, complete calendar, thresholds, answers
print("=" * 60)
print("PART 1: Load data and build answers")
print("=" * 60)

raw = pd.read_csv(INPUT_FILE, low_memory=False)
raw['DATE'] = pd.to_datetime(raw['DATE'])
print(f"  {len(raw)} rows, {raw['STATION'].nunique()} stations")

# Put every station on a complete daily calendar (missing dates -> empty rows)
frames = []
for st, grp in raw.groupby('STATION'):
    full = pd.DataFrame({'DATE': pd.date_range(grp['DATE'].min(), grp['DATE'].max(), freq='D')})
    grp = full.merge(grp, on='DATE', how='left')
    grp['STATION'] = st
    frames.append(grp)
df = pd.concat(frames, ignore_index=True).sort_values(['STATION', 'DATE']).reset_index(drop=True)
df['station'] = df['STATION'].map(lambda s: station_info(s)[0])
df['station_code'] = df['STATION'].map(lambda s: station_info(s)[1])
df['year'] = df['DATE'].dt.year
df['month'] = df['DATE'].dt.month
print(f"  {len(df)} rows after filling the calendar "
      f"({len(df) - len(raw)} missing dates added as empty rows)")

# Thresholds: 95th percentile of May–October high / low, 1961–1990
ref = df[df['year'].between(REF_START, REF_END) & df['month'].isin(WARM_MONTHS)]
th = ref.groupby('STATION')[['TMAX_C', 'TMIN_C']].quantile(PERCENTILE)
th.columns = ['tmax_threshold', 'tmin_threshold']
df = df.join(th, on='STATION')

th_out = th.reset_index()
th_out.insert(1, 'station', th_out['STATION'].map(lambda s: station_info(s)[0]))
th_out.insert(2, 'station_code', th_out['STATION'].map(lambda s: station_info(s)[1]))
th_out = th_out.sort_values('station_code').round(1)
th_out.to_csv(THRESH_FILE, index=False)
print("  Station codes and thresholds (°C):")
print(th_out.to_string(index=False))

# Daily answers (NaN when the temperature is missing)
df['hot_day'] = np.where(df['TMAX_C'].notna(),
                         (df['TMAX_C'] > df['tmax_threshold']).astype(float), np.nan)
df['hot_night'] = np.where(df['TMIN_C'].notna(),
                           (df['TMIN_C'] > df['tmin_threshold']).astype(float), np.nan)


# PART 2 — Task A: daily features
print("\n" + "=" * 60)
print("PART 2: Task A (daily)")
print("=" * 60)

g = df.groupby('STATION')

# Temperature the day before
df['tmin_lag1'] = g['TMIN_C'].shift(1)
df['dtr_lag1'] = (df['TMAX_C'] - df['TMIN_C']).groupby(df['STATION']).shift(1)

# Rain
df['prcp_lag1'] = g['PRCP_MM'].shift(1)
min_days = int(np.ceil(30 * MIN_FRAC))
# 30-day total = 30-day mean x 30, so a few missing days don't shrink the total
df['prcp_sum30'] = g['PRCP_MM'].transform(
    lambda s: s.shift(1).rolling(30, min_periods=min_days).mean()) * 30
rain_date = df['DATE'].where(df['PRCP_MM'] >= RAIN_DAY_MM)
last_rain = rain_date.groupby(df['STATION']).ffill()
last_rain_before = last_rain.groupby(df['STATION']).shift(1)    # up to yesterday
df['days_since_rain'] = (df['DATE'] - last_rain_before).dt.days

# Time of year: day number counted from January 1
# (May 1 = 121, Oct 31 = 304; one day later in leap years)
df['dayOfYear'] = df['DATE'].dt.dayofyear

# ONI: a day in month m uses the season centered on m-2 (ended in m-1)
oni_wide = pd.read_csv(ONI_FILE)
oni = oni_wide.melt(id_vars='Year', var_name='season', value_name='oni').dropna()
oni['key'] = oni['Year'] * 12 + oni['season'].map(SEASON_CENTER) - 1
oni_map = oni.set_index('key')['oni']
day_key = df['year'] * 12 + df['month'] - 1
df['oni_lag'] = (day_key - 2).map(oni_map)
center_to_season = {v: k for k, v in SEASON_CENTER.items()}
print("  ONI month -> season: " + ", ".join(
    f"{m}->{center_to_season[(m - 3) % 12 + 1]}" for m in WARM_MONTHS))

A_WEATHER = ['tmin_lag1', 'dtr_lag1', 'dayOfYear', 'days_since_rain',
             'oni_lag', 'prcp_sum30', 'prcp_lag1']
A_FEATURES = A_WEATHER + ['station_code']
A_ID = ['STATION', 'station', 'DATE', 'year']
A_LABELS = ['hot_day', 'hot_night']

a = df[df['month'].isin(WARM_MONTHS)]
a = a[a['hot_day'].notna() | a['hot_night'].notna()]        # need at least one answer
task_a = a[A_ID + A_FEATURES + A_LABELS].copy()
task_a['DATE'] = task_a['DATE'].dt.strftime('%Y-%m-%d')
num = task_a.select_dtypes('float').columns
task_a[num] = task_a[num].round(3)
task_a.to_csv(TASK_A_FILE, index=False)

print(f"  Saved {TASK_A_FILE}: {len(task_a)} rows, {len(A_FEATURES)} features "
      f"({len(A_WEATHER)} weather/time + station_code)")
print(f"  oni_lag missing: {task_a['oni_lag'].isna().sum()}")
check_correlation(task_a[task_a['year'] <= CHECK_END_YEAR], A_WEATHER, "Task A")
print("  Positive rate (%) by station:")
print((task_a.groupby('station')[A_LABELS].mean() * 100).round(1).to_string())


# PART 3 — Task B: yearly features
print("\n" + "=" * 60)
print("PART 3: Task B (yearly)")
print("=" * 60)

ws = df[df['month'].isin(WARM_MONTHS)].copy()


def longest_run(flags):
    """Longest run of consecutive 1s (missing days break a run)."""
    best = cur = 0
    for v in flags.fillna(0).to_numpy():
        cur = cur + 1 if v == 1 else 0
        best = max(best, cur)
    return best


yr = ws.groupby(['STATION', 'year']).agg(
    n_tmax=('TMAX_C', 'count'),
    n_tmin=('TMIN_C', 'count'),
    hot_day_count=('hot_day', 'sum'),
    hot_night_count=('hot_night', 'sum'),
    mean_tmin=('TMIN_C', 'mean'),
    max_spell_days=('hot_day', longest_run),
).reset_index()

yr['coverage'] = np.minimum(yr['n_tmax'], yr['n_tmin']) / WARM_DAYS
yr['valid'] = yr['coverage'] >= B_MIN_COVERAGE

# Same-year values are kept only for valid years, and used ONLY as
# inputs for later years (never as inputs for the same year)
YEAR_STATS = ['hot_day_count', 'hot_night_count', 'mean_tmin', 'max_spell_days']
yr.loc[~yr['valid'], YEAR_STATS] = np.nan

# Complete year range per station so "last year" is always year - 1
full_years = []
for st, grp in yr.groupby('STATION'):
    years = pd.DataFrame({'year': range(grp['year'].min(), grp['year'].max() + 1)})
    grp = years.merge(grp, on='year', how='left')
    grp['STATION'] = st
    full_years.append(grp)
yr = pd.concat(full_years, ignore_index=True).sort_values(['STATION', 'year'])
yr['valid'] = yr['valid'].fillna(False).astype(bool)
gy = yr.groupby('STATION')

yr['prev_hot_nights'] = gy['hot_night_count'].shift(1)
yr['prev_mean_tmin'] = gy['mean_tmin'].shift(1)
yr['prev_max_spell_days'] = gy['max_spell_days'].shift(1)
yr['avg5_hot_days'] = gy['hot_day_count'].transform(
    lambda s: s.shift(1).rolling(B_AVG_YEARS, min_periods=B_AVG_MIN).mean())

yr['station'] = yr['STATION'].map(lambda s: station_info(s)[0])
yr['station_code'] = yr['STATION'].map(lambda s: station_info(s)[1])

B_MAIN = ['prev_hot_nights', 'year', 'prev_mean_tmin',
          'avg5_hot_days', 'prev_max_spell_days']
B_FEATURES = B_MAIN + ['station_code']
B_ID = ['STATION', 'station', 'coverage']
B_LABELS = ['hot_day_count', 'hot_night_count']

dropped = yr[~yr['valid']][['station', 'year']]
task_b = yr[yr['valid']][B_ID + B_FEATURES + B_LABELS].copy()
num = task_b.select_dtypes('float').columns
task_b[num] = task_b[num].round(3)
task_b.to_csv(TASK_B_FILE, index=False)

print(f"  Saved {TASK_B_FILE}: {len(task_b)} rows, {len(B_FEATURES)} features "
      f"({len(B_MAIN)} main + station_code)")
print(f"  Station-years dropped (coverage < {B_MIN_COVERAGE:.0%}): {len(dropped)}")
for st, grp in dropped.groupby('station'):
    print(f"    {st}: {grp['year'].tolist()}")
check_correlation(task_b[task_b['year'] <= CHECK_END_YEAR], B_MAIN, "Task B")
print("  Mean answer per station:")
print(task_b.groupby('station')[B_LABELS].mean().round(1).to_string())


# PART 4 — Feature list
code_text = ", ".join(f"{code} = {name}" for name, code in STATIONS.values())
DESCRIPTIONS = {
    # ---- shared / id
    'STATION': ('id', 'NOAA station ID (merged stations joined with +)', '', 'raw'),
    'station': ('id', 'Short station name', '', 'raw'),
    'DATE': ('id', 'Date', 'YYYY-MM-DD', 'raw'),
    'year': ('id', 'Calendar year', 'year', 'DATE'),
    'coverage': ('id', 'Share of May–October days with both high and low temperature', 'fraction', 'TMAX, TMIN'),
    'station_code': ('feature', f'Station label ({code_text}). Category, not a quantity: '
                                'use 0/1 columns for Logistic / Linear Regression', 'code', 'STATION'),
    # ---- Task A
    'tmin_lag1': ('feature', 'Low temperature the day before', '°C', 'TMIN'),
    'dtr_lag1': ('feature', 'High minus low temperature the day before', '°C', 'TMAX, TMIN'),
    'dayOfYear': ('feature', 'Day number counted from January 1 '
                             '(May 1 = 121, Oct 31 = 304; one day later in leap years)', 'day', 'DATE'),
    'days_since_rain': ('feature', 'Days since the last day with 1 mm or more of rain', 'days', 'PRCP'),
    'oni_lag': ('feature', 'ONI of the 3 months that ended the month before', '°C anomaly', 'ONI v6'),
    'prcp_sum30': ('feature', 'Total rain over the past 30 days', 'mm', 'PRCP'),
    'prcp_lag1': ('feature', 'Rain the day before', 'mm', 'PRCP'),
    'hot_day': ('answer', '1 if the high is above the station 95th percentile threshold', '0/1', 'TMAX'),
    'hot_night': ('answer', '1 if the low is above the station 95th percentile threshold', '0/1', 'TMIN'),
    # ---- Task B
    'prev_hot_nights': ('feature', 'Extreme heat nights last year', 'nights', 'TMIN'),
    'prev_mean_tmin': ('feature', 'Average May–October low temperature last year', '°C', 'TMIN'),
    'avg5_hot_days': ('feature', 'Average extreme heat days over the past 5 years', 'days', 'TMAX'),
    'prev_max_spell_days': ('feature', 'Longest run of extreme heat days last year', 'days', 'TMAX'),
    'hot_day_count': ('answer', 'Extreme heat days from May to October this year', 'days', 'TMAX'),
    'hot_night_count': ('answer', 'Extreme heat nights from May to October this year', 'nights', 'TMIN'),
}

rows = []
for file, cols in [(TASK_A_FILE, list(task_a.columns)), (TASK_B_FILE, list(task_b.columns))]:
    for c in cols:
        role, desc, unit, src = DESCRIPTIONS[c]
        if c == 'year' and file == TASK_B_FILE:
            role = 'feature'
        rows.append(dict(file=file, column=c, role=role,
                         description=desc, unit=unit, source=src))
pd.DataFrame(rows).to_csv(FEATLIST_FILE, index=False)
print(f"\nSaved {FEATLIST_FILE}: {len(rows)} columns described")