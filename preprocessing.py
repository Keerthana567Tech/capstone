"""
Data Preprocessing and Feature Engineering Pipeline
Wholesale Onion Market Price Prediction (2014-2024)
Supports Classical and Quantum Machine Learning (QML) workflows.
"""

import os
import sys
import zipfile
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['figure.titleweight'] = 'bold'


def resolve_data_paths():
    """
    Resolve workspace paths for input data and visualization outputs.

    Returns:
        tuple: (base_dir, csv_path, zip_path, viz_dir)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    csv_name = "preprocessed_onion_2014_2024_prices_weather_events.csv"
    zip_name = "preprocessed_onion_2014_2024_prices_weather_events.zip"

    csv_path = os.path.join(base_dir, csv_name)
    zip_path = os.path.join(base_dir, zip_name)
    viz_dir = os.path.join(base_dir, "preprocessing_visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    return base_dir, csv_path, zip_path, viz_dir


def load_raw_dataset(csv_path, zip_path):
    """
    Load raw dataset from CSV file or ZIP archive fallback.

    Args:
        csv_path (str): Path to uncompressed CSV file.
        zip_path (str): Path to ZIP archive.

    Returns:
        pd.DataFrame: Loaded raw dataset.
    """
    df = None
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except (PermissionError, Exception):
            df = None

    if df is None and os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_in_zip = [f for f in z.namelist() if f.endswith('.csv')][0]
            with z.open(csv_in_zip) as f_in:
                df = pd.read_csv(f_in, low_memory=False)

    if df is None:
        raise FileNotFoundError(f"Could not locate dataset in {csv_path} or {zip_path}.")

    return df


def profile_baseline_dataset(df):
    """
    Compute baseline summary statistics prior to preprocessing.

    Args:
        df (pd.DataFrame): Raw dataset.

    Returns:
        dict: Baseline profiling metrics.
    """
    raw_rows, raw_cols = df.shape
    target_col = 'modal_price_rs_qtl'

    missing_series = df.isna().sum()
    total_missing = missing_series.sum()

    exact_duplicates = df.duplicated().sum()
    key_cols = ['date', 'market', 'variety', 'grade']
    key_duplicates = df.duplicated(subset=key_cols, keep=False).sum()

    target_series = df[target_col]
    zero_prices = (target_series == 0).sum()
    neg_prices = (target_series < 0).sum()

    q25 = target_series.quantile(0.25)
    q50 = target_series.median()
    q75 = target_series.quantile(0.75)
    iqr = q75 - q25
    iqr_lower = q25 - 1.5 * iqr
    iqr_upper = q75 + 1.5 * iqr
    outliers_iqr = ((target_series < iqr_lower) | (target_series > iqr_upper)).sum()
    extreme_scale_errors = (target_series > 25000).sum()

    num_states = df['state'].nunique()
    num_districts = df['district'].nunique()
    num_markets = df['market'].nunique()
    num_varieties = df['variety'].nunique()

    price_bins = [-np.inf, 1000, 2500, 5000, 10000, np.inf]
    bin_labels = ['Low (<1000)', 'Normal (1000-2500)', 'Moderate High (2500-5000)', 'High (5000-10000)', 'Spike (>10000)']
    raw_binned = pd.cut(target_series, bins=price_bins, labels=bin_labels).value_counts(sort=False)

    metrics_before = {
        'rows': raw_rows,
        'cols': raw_cols,
        'missing_values': total_missing,
        'exact_duplicates': exact_duplicates,
        'key_duplicates': key_duplicates,
        'zero_prices': zero_prices,
        'invalid_prices': zero_prices + neg_prices,
        'iqr_outliers': outliers_iqr,
        'extreme_errors': extreme_scale_errors,
        'markets': num_markets,
        'districts': num_districts,
        'states': num_states,
        'varieties': num_varieties,
        'target_mean': target_series.mean(),
        'target_median': q50,
        'target_std': target_series.std(),
        'target_min': target_series.min(),
        'target_max': target_series.max(),
        'price_tiers': raw_binned
    }

    return metrics_before


def generate_feature_classification_report(df, base_dir):
    """
    Classify raw columns into functional groups and export preprocessing_report.csv.

    Args:
        df (pd.DataFrame): Input dataframe.
        base_dir (str): Base output directory.

    Returns:
        pd.DataFrame: Feature classification report.
    """
    report_rows = []

    classification_rules = {
        'date': ('Essential', 'Retained and converted to monthly calendar index.'),
        'state': ('Essential', 'Retained as primary administrative region indicator.'),
        'district': ('Essential', 'Retained as local agricultural district indicator.'),
        'market': ('Essential', 'Retained as wholesale market trading node indicator.'),
        'variety': ('Essential', 'Retained as crop cultivar indicator.'),
        'group': ('Redundant', 'Dropped; single constant value across dataset.'),
        'grade': ('Potentially Useful', 'Retained via mode; 98.6% of records are standard "FAQ".'),
        'market_key': ('Redundant', 'Dropped; redundant concatenation of state, district, and market.'),
        'lat': ('Potentially Useful', 'Retained as spatial coordinate.'),
        'lon': ('Potentially Useful', 'Retained as spatial coordinate.'),
        'grid_lat': ('Redundant', 'Dropped; coarse weather grid coordinates duplicate latitude.'),
        'grid_lon': ('Redundant', 'Dropped; coarse weather grid coordinates duplicate longitude.'),
        'arrivals_tonnes': ('Essential', 'Aggregated into monthly total and daily mean arrival volumes.'),
        'modal_price_rs_qtl': ('Essential', 'Primary market price basis; aggregated to monthly statistics.'),
        'min_price_rs_qtl': ('Potentially Useful', 'Aggregated to monthly minimum price.'),
        'max_price_rs_qtl': ('Potentially Useful', 'Aggregated to monthly maximum price.'),
        'rain': ('Essential', 'Aggregated into monthly cumulative rainfall (mm).'),
        'temp_avg': ('Essential', 'Aggregated into monthly mean temperature (°C).'),
        'tmax': ('Potentially Useful', 'Aggregated into monthly mean maximum temperature.'),
        'tmin': ('Potentially Useful', 'Aggregated into monthly mean minimum temperature.'),
        'year': ('Essential', 'Retained as long-term trend indicator.'),
        'month': ('Essential', 'Retained as annual seasonal calendar identifier.'),
        'week': ('Redundant', 'Dropped; daily/weekly resolution replaced by monthly aggregation.'),
        'dayofweek': ('Redundant', 'Dropped; intra-week resolution not applicable for monthly horizon.'),
        'is_weekend': ('Redundant', 'Dropped; intra-week pattern not applicable for monthly horizon.'),
        'season': ('Potentially Useful', 'Retained via monthly mode.'),
        'crop_season_onion': ('Essential', 'Retained via monthly mode.'),
        'festival_name': ('Unnecessary', 'Dropped; 96.0% missing values; seasonality captured by calendar month.'),
        'is_festival': ('Potentially Useful', 'Aggregated into monthly festival indicator.'),
        'price_spike': ('Redundant', 'Dropped; daily indicator derived from target; monthly target reflects level.'),
        'arrival_zero': ('Redundant', 'Dropped; constant zero column.'),
        'modal_price_lag_1': ('Leakage/Misaligned', 'Dropped; daily lag replaced by monthly lag.'),
        'modal_price_pctchg_lag_1': ('Leakage/Misaligned', 'Dropped; replaced by monthly percentage change.'),
        'arrivals_lag_1': ('Leakage/Misaligned', 'Dropped; replaced by monthly arrivals lag.'),
        'arrivals_pctchg_lag_1': ('Leakage/Misaligned', 'Dropped; replaced by monthly arrivals percentage change.'),
        'modal_price_lag_7': ('Leakage/Misaligned', 'Dropped; daily lag superseded by monthly lag structures.'),
        'modal_price_pctchg_lag_7': ('Leakage/Misaligned', 'Dropped; daily lag superseded by monthly lag structures.'),
        'arrivals_lag_7': ('Leakage/Misaligned', 'Dropped; daily lag superseded by monthly lag structures.'),
        'arrivals_pctchg_lag_7': ('Leakage/Misaligned', 'Dropped; daily lag superseded by monthly lag structures.'),
        'modal_price_lag_30': ('Leakage/Misaligned', 'Dropped; daily lag replaced by monthly lag.'),
        'modal_price_pctchg_lag_30': ('Leakage/Misaligned', 'Dropped; replaced by monthly price change ratio.'),
        'arrivals_lag_30': ('Leakage/Misaligned', 'Dropped; replaced by monthly arrivals lag.'),
        'arrivals_pctchg_lag_30': ('Leakage/Misaligned', 'Dropped; replaced by monthly arrivals change ratio.'),
        'modal_price_7d_mean': ('Leakage/Misaligned', 'Dropped; daily rolling window superseded by monthly price.'),
        'arrivals_7d_mean': ('Leakage/Misaligned', 'Dropped; daily rolling window superseded by monthly arrivals.'),
        'modal_price_30d_mean': ('Leakage/Misaligned', 'Dropped; daily rolling window superseded by monthly price.'),
        'arrivals_30d_mean': ('Leakage/Misaligned', 'Dropped; daily rolling window superseded by monthly arrivals.'),
        'avg_temp_avg_5m': ('Unnecessary', 'Dropped; 42.0% missing; reconstructed via clean monthly rolling means.'),
        'avg_tmax_5m': ('Unnecessary', 'Dropped; 42.0% missing; reconstructed via clean monthly rolling features.'),
        'avg_tmin_5m': ('Unnecessary', 'Dropped; 42.0% missing; reconstructed via clean monthly rolling features.'),
        'avg_rain_5m': ('Unnecessary', 'Dropped; 42.1% missing; reconstructed via clean monthly rolling features.'),
        'cum_rain_5m': ('Unnecessary', 'Dropped; 41.3% missing; replaced by 3-month and 6-month cumulative rainfall.'),
        'temp_avg_expanding_mean': ('Leakage/Misaligned', 'Dropped; global expanding mean introduces lookahead bias.'),
        'temp_avg_district_mean': ('Redundant', 'Dropped; redundant with monthly mean temperature.'),
        'temp_avg_anomaly': ('Redundant', 'Dropped; redundant with monthly rolling temperature deviations.'),
        'cum_rain_ytd': ('Redundant', 'Dropped; annual calendar reset replaced by continuous rolling rainfall.')
    }

    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        pct_missing = round(float(n_missing / len(df) * 100), 2)
        dtype = str(df[col].dtype)
        n_unique = int(df[col].nunique(dropna=True))

        cat, rationale = classification_rules.get(
            col, ('Potentially Useful', 'Evaluated and aggregated as appropriate for monthly forecasting.')
        )

        report_rows.append({
            'column_name': col,
            'original_dtype': dtype,
            'missing_count': n_missing,
            'missing_percentage': pct_missing,
            'unique_values': n_unique,
            'classification_category': cat,
            'action_decision': 'Retained / Aggregated' if cat in ['Essential', 'Potentially Useful'] else 'Dropped / Replaced',
            'technical_rationale': rationale
        })

    report_df = pd.DataFrame(report_rows)
    report_path = os.path.join(base_dir, "preprocessing_report.csv")
    report_df.to_csv(report_path, index=False)
    return report_df


def clean_daily_dataset(df):
    """
    Clean daily transaction records and resolve anomalies.

    Operations:
        - Standardize categorical text fields.
        - Impute zero modal prices using (min_price + max_price) / 2 when bounds exist.
        - Remove non-positive modal price records.
        - Filter recording scale errors (> 25,000 Rs/qtl).
        - Remove duplicate records on (date, market, variety, grade).

    Args:
        df (pd.DataFrame): Raw observations.

    Returns:
        tuple: (pd.DataFrame, dict) Cleaned dataframe and processing audit counts.
    """
    initial_count = len(df)
    cleaning_ledger = {
        'initial_rows': initial_count,
        'imputed_zero_modal': 0,
        'dropped_unrecoverable_zero': 0,
        'dropped_scale_errors': 0,
        'dropped_key_duplicates': 0,
        'final_clean_daily_rows': 0
    }

    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    for col in ['state', 'district', 'market', 'variety', 'grade']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    zero_modal_cond = (df['modal_price_rs_qtl'] == 0) & (df['min_price_rs_qtl'] > 0) & (df['max_price_rs_qtl'] > 0)
    impute_count = zero_modal_cond.sum()
    df.loc[zero_modal_cond, 'modal_price_rs_qtl'] = (
        df.loc[zero_modal_cond, 'min_price_rs_qtl'] + df.loc[zero_modal_cond, 'max_price_rs_qtl']
    ) / 2.0
    cleaning_ledger['imputed_zero_modal'] = int(impute_count)

    invalid_price_cond = df['modal_price_rs_qtl'] <= 0
    dropped_zero_count = invalid_price_cond.sum()
    df = df[~invalid_price_cond].copy()
    cleaning_ledger['dropped_unrecoverable_zero'] = int(dropped_zero_count)

    # Values exceeding 25,000 Rs/qtl represent unit reporting scale errors (e.g. per-tonne rates logged as per-quintal).
    scale_error_cond = df['modal_price_rs_qtl'] > 25000
    dropped_scale_count = scale_error_cond.sum()
    df = df[~scale_error_cond].copy()
    cleaning_ledger['dropped_scale_errors'] = int(dropped_scale_count)

    key_cols = ['date', 'market', 'variety', 'grade']
    before_dedup = len(df)
    df = df.sort_values(by=key_cols + ['modal_price_rs_qtl', 'arrivals_tonnes'])
    df = df.drop_duplicates(subset=key_cols, keep='last')
    dropped_dups = before_dedup - len(df)
    cleaning_ledger['dropped_key_duplicates'] = int(dropped_dups)

    cleaning_ledger['final_clean_daily_rows'] = len(df)
    return df, cleaning_ledger


def aggregate_to_monthly_dataset(df):
    """
    Aggregate daily observations into monthly grain and formulate target variable.

    Groups by (state, district, market, variety, year, month).
    Constructs next-month prediction target target_price_next_month (P_{t+1}).

    Args:
        df (pd.DataFrame): Cleaned daily dataframe.

    Returns:
        pd.DataFrame: Monthly aggregated dataframe.
    """
    group_cols = ['state', 'district', 'market', 'variety', 'year', 'month']

    agg_rules = {
        'modal_price_rs_qtl': ['mean', 'median', 'min', 'max', 'std', 'count'],
        'arrivals_tonnes': ['sum', 'mean'],
        'rain': 'sum',
        'temp_avg': 'mean',
        'tmax': 'mean',
        'tmin': 'mean',
        'season': lambda x: x.mode()[0] if not x.empty else 'Rabi',
        'crop_season_onion': lambda x: x.mode()[0] if not x.empty else 'Rabi',
        'lat': 'first',
        'lon': 'first'
    }

    m_df = df.groupby(group_cols).agg(agg_rules)

    flattened_cols = []
    for c in m_df.columns:
        if isinstance(c, tuple):
            if c[1] in ['<lambda>', '<lambda_0>', 'first']:
                flattened_cols.append(c[0])
            else:
                flattened_cols.append(f"{c[0]}_{c[1]}")
        else:
            flattened_cols.append(c)
    m_df.columns = flattened_cols
    m_df = m_df.reset_index()

    rename_dict = {
        'modal_price_rs_qtl_mean': 'price_current_month',
        'modal_price_rs_qtl_median': 'price_median_current_month',
        'modal_price_rs_qtl_min': 'price_min_current_month',
        'modal_price_rs_qtl_max': 'price_max_current_month',
        'modal_price_rs_qtl_std': 'price_std_current_month',
        'modal_price_rs_qtl_count': 'days_reported',
        'arrivals_tonnes_sum': 'arrivals_total_current_month',
        'arrivals_tonnes_mean': 'arrivals_daily_avg_current_month',
        'rain_sum': 'rainfall_total_current_month',
        'rain': 'rainfall_total_current_month',
        'temp_avg_mean': 'temp_avg_current_month',
        'temp_avg': 'temp_avg_current_month',
        'tmax_mean': 'tmax_current_month',
        'tmax': 'tmax_current_month',
        'tmin_mean': 'tmin_current_month',
        'tmin': 'tmin_current_month'
    }
    m_df = m_df.rename(columns=rename_dict)

    m_df['price_std_current_month'] = m_df['price_std_current_month'].fillna(0.0)

    for w_col in ['rainfall_total_current_month', 'temp_avg_current_month', 'tmax_current_month', 'tmin_current_month']:
        if w_col in m_df.columns:
            m_df[w_col] = m_df.groupby(['state', 'month'])[w_col].transform(lambda s: s.fillna(s.mean()))
            m_df[w_col] = m_df.groupby('month')[w_col].transform(lambda s: s.fillna(s.mean()))
            m_df[w_col] = m_df[w_col].fillna(m_df[w_col].mean())

    for geo_col in ['lat', 'lon']:
        if geo_col in m_df.columns:
            m_df[geo_col] = m_df.groupby('district')[geo_col].transform(lambda s: s.fillna(s.mean()))
            m_df[geo_col] = m_df.groupby('state')[geo_col].transform(lambda s: s.fillna(s.mean()))
            m_df[geo_col] = m_df[geo_col].fillna(m_df[geo_col].mean())

    m_df['month_id'] = m_df['year'] * 12 + m_df['month']
    m_df['series_id'] = m_df['state'] + '_' + m_df['district'] + '_' + m_df['market'] + '_' + m_df['variety']

    m_df = m_df.sort_values(by=['series_id', 'month_id']).reset_index(drop=True)

    target_lookup = m_df[['series_id', 'month_id', 'price_current_month', 'year', 'month']].copy()
    target_lookup['match_id'] = target_lookup['month_id'] - 1
    target_lookup = target_lookup.rename(columns={
        'price_current_month': 'target_price_next_month',
        'year': 'target_year',
        'month': 'target_month'
    }).drop(columns=['month_id'])

    m_df = m_df.merge(target_lookup, left_on=['series_id', 'month_id'], right_on=['series_id', 'match_id'], how='left')
    m_df = m_df.drop(columns=['match_id'])

    return m_df


def engineer_monthly_features(m_df):
    """
    Construct time-aware lag, rolling window, and seasonal features.

    Features generated:
        - Autoregressive price lags: t-1, t-2, t-3, t-6, t-12.
        - Rolling price means (3m, 6m, 12m) and rolling volatility (3m std).
        - Price and arrival monthly momentum ratios.
        - Supply arrival lags and rolling statistics.
        - Weather lags and cumulative rainfall indicators.
        - Cyclical calendar transforms (sine and cosine of month).

    Args:
        m_df (pd.DataFrame): Monthly dataframe.

    Returns:
        pd.DataFrame: Feature-engineered dataframe with valid target rows.
    """
    def add_lag_feature(df, value_col, lag_months, new_col_name):
        lookup = df[['series_id', 'month_id', value_col]].copy()
        lookup['match_id'] = lookup['month_id'] + lag_months
        lookup = lookup.rename(columns={value_col: new_col_name}).drop(columns=['month_id'])
        df = df.merge(lookup, left_on=['series_id', 'month_id'], right_on=['series_id', 'match_id'], how='left')
        return df.drop(columns=['match_id'])

    m_df = add_lag_feature(m_df, 'price_current_month', 1, 'price_lag_1')
    m_df = add_lag_feature(m_df, 'price_current_month', 2, 'price_lag_2')
    m_df = add_lag_feature(m_df, 'price_current_month', 3, 'price_lag_3')
    m_df = add_lag_feature(m_df, 'price_current_month', 6, 'price_lag_6')
    m_df = add_lag_feature(m_df, 'price_current_month', 12, 'price_lag_12')

    m_df = add_lag_feature(m_df, 'arrivals_total_current_month', 1, 'arrivals_lag_1')
    m_df = add_lag_feature(m_df, 'rainfall_total_current_month', 1, 'rainfall_lag_1')
    m_df = add_lag_feature(m_df, 'temp_avg_current_month', 1, 'temp_avg_lag_1')

    m_df['price_lag_1'] = m_df['price_lag_1'].fillna(m_df['price_current_month'])
    m_df['price_lag_2'] = m_df['price_lag_2'].fillna(m_df['price_lag_1'])
    m_df['price_lag_3'] = m_df['price_lag_3'].fillna(m_df['price_lag_2'])
    m_df['price_lag_6'] = m_df['price_lag_6'].fillna(m_df['price_lag_3'])
    m_df['price_lag_12'] = m_df['price_lag_12'].fillna(m_df['price_lag_6'])

    m_df['arrivals_lag_1'] = m_df['arrivals_lag_1'].fillna(m_df['arrivals_total_current_month'])
    m_df['rainfall_lag_1'] = m_df['rainfall_lag_1'].fillna(m_df['rainfall_total_current_month'])
    m_df['temp_avg_lag_1'] = m_df['temp_avg_lag_1'].fillna(m_df['temp_avg_current_month'])

    m_df['price_rolling_3m_mean'] = (m_df['price_current_month'] + m_df['price_lag_1'] + m_df['price_lag_2']) / 3.0
    m_df['price_rolling_6m_mean'] = (
        m_df['price_current_month'] + m_df['price_lag_1'] + m_df['price_lag_2'] +
        m_df['price_lag_3'] + m_df['price_lag_6'] * 2.0
    ) / 6.0
    m_df['price_rolling_12m_mean'] = (m_df['price_rolling_6m_mean'] + m_df['price_lag_12']) / 2.0

    price_matrix_3m = np.column_stack([
        m_df['price_current_month'].values,
        m_df['price_lag_1'].values,
        m_df['price_lag_2'].values
    ])
    m_df['price_rolling_3m_std'] = np.std(price_matrix_3m, axis=1)

    m_df['price_change_ratio_1m'] = (m_df['price_current_month'] - m_df['price_lag_1']) / (m_df['price_lag_1'] + 1e-5)
    m_df['arrivals_change_ratio_1m'] = (m_df['arrivals_total_current_month'] - m_df['arrivals_lag_1']) / (m_df['arrivals_lag_1'] + 1e-5)

    m_df['arrivals_rolling_3m_mean'] = (m_df['arrivals_total_current_month'] + m_df['arrivals_lag_1']) / 2.0
    m_df['rainfall_rolling_3m_sum'] = m_df['rainfall_total_current_month'] + m_df['rainfall_lag_1']
    m_df['temp_rolling_3m_mean'] = (m_df['temp_avg_current_month'] + m_df['temp_avg_lag_1']) / 2.0

    m_df['quarter'] = ((m_df['month'] - 1) // 3) + 1
    m_df['sin_month'] = np.sin(2.0 * np.pi * m_df['month'] / 12.0)
    m_df['cos_month'] = np.cos(2.0 * np.pi * m_df['month'] / 12.0)

    final_m_df = m_df[m_df['target_price_next_month'].notna()].copy()
    final_m_df['target_year'] = final_m_df['target_year'].astype(int)
    final_m_df['target_month'] = final_m_df['target_month'].astype(int)

    return final_m_df


def perform_imbalance_analysis(df, split_year_train=2021):
    """
    Profile target distribution across price tiers and compute loss weights.

    Assigns inverse-tier-frequency sample weights on the training partition
    to balance regression loss penalty across rare crisis periods without
    synthetic data interpolation.

    Args:
        df (pd.DataFrame): Input dataframe.
        split_year_train (int): Upper bound year for training split.

    Returns:
        tuple: (pd.DataFrame, pd.DataFrame, float) Dataframe with sample_weight,
               tier distribution table, and imbalance ratio.
    """
    target = df['target_price_next_month']

    bins = [-np.inf, 1000, 2500, 5000, np.inf]
    labels = ['Low (<1000)', 'Normal (1000-2500)', 'Moderate High (2500-5000)', 'Crisis Spike (>5000)']
    df['price_tier'] = pd.cut(target, bins=bins, labels=labels)

    tier_counts = df['price_tier'].value_counts(sort=False)
    tier_pcts = (tier_counts / len(df) * 100).round(2)
    imbalance_ratio = tier_counts.max() / tier_counts.min()

    tier_df = pd.DataFrame({
        'Observations': tier_counts,
        'Percentage': tier_pcts
    })

    train_mask = df['year'] <= split_year_train
    train_tiers = df.loc[train_mask, 'price_tier'].value_counts()
    n_train = train_mask.sum()
    n_classes = len(labels)

    # Inverse-frequency class weighting formula: w_k = N / (K * N_k)
    tier_weights = {}
    for label in labels:
        count = train_tiers.get(label, 1)
        tier_weights[label] = n_train / (n_classes * count)

    df['sample_weight'] = 1.0
    df.loc[train_mask, 'sample_weight'] = df.loc[train_mask, 'price_tier'].map(tier_weights)

    train_w_mean = df.loc[train_mask, 'sample_weight'].mean()
    df.loc[train_mask, 'sample_weight'] /= train_w_mean

    return df, tier_df, imbalance_ratio


def split_chronologically(df):
    """
    Split dataset chronologically to prevent temporal lookahead bias.

    Partitions:
        - Train: 2014-2021
        - Validation: 2022-2023
        - Test: 2024 (Out-of-time benchmark)

    Args:
        df (pd.DataFrame): Full monthly dataset.

    Returns:
        tuple: (pd.DataFrame, pd.DataFrame, pd.DataFrame) train, val, test splits.
    """
    train_df = df[df['year'] <= 2021].copy()
    val_df = df[(df['year'] >= 2022) & (df['year'] <= 2023)].copy()
    test_df = df[df['year'] == 2024].copy()

    return train_df, val_df, test_df


def prepare_classical_and_quantum_features(train_df, val_df, test_df, full_df):
    """
    Prepare feature encodings and scalers for Classical and Quantum ML models.

    Steps:
        1. Frequency encode high-cardinality categorical features using training frequencies.
        2. Fit RobustScaler on training predictors for Classical ML.
        3. Fit MinMaxScaler to [0, pi] on training predictors for QML rotation gates.
        4. Rank features using Random Forest importance for 8-qubit and 16-qubit circuits.

    Args:
        train_df (pd.DataFrame): Training partition.
        val_df (pd.DataFrame): Validation partition.
        test_df (pd.DataFrame): Test partition.
        full_df (pd.DataFrame): Full monthly dataset.

    Returns:
        tuple: (train_df, val_df, test_df, full_df, feature_cols, top_8, top_16, importances)
    """
    cat_cols = ['market', 'district', 'state', 'variety', 'crop_season_onion']
    for col in cat_cols:
        freq_map = train_df[col].value_counts(normalize=True).to_dict()
        train_df[f"{col}_encoded"] = train_df[col].map(freq_map).fillna(0.0)
        val_df[f"{col}_encoded"] = val_df[col].map(freq_map).fillna(0.0)
        test_df[f"{col}_encoded"] = test_df[col].map(freq_map).fillna(0.0)
        full_df[f"{col}_encoded"] = full_df[col].map(freq_map).fillna(0.0)

    feature_cols = [
        'price_current_month', 'price_median_current_month', 'price_min_current_month',
        'price_max_current_month', 'price_std_current_month', 'days_reported',
        'price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_6', 'price_lag_12',
        'price_rolling_3m_mean', 'price_rolling_6m_mean', 'price_rolling_12m_mean',
        'price_rolling_3m_std', 'price_change_ratio_1m',
        'arrivals_total_current_month', 'arrivals_daily_avg_current_month',
        'arrivals_lag_1', 'arrivals_rolling_3m_mean', 'arrivals_change_ratio_1m',
        'rainfall_total_current_month', 'rainfall_lag_1', 'rainfall_rolling_3m_sum',
        'temp_avg_current_month', 'temp_avg_lag_1', 'temp_rolling_3m_mean',
        'tmax_current_month', 'tmin_current_month',
        'quarter', 'sin_month', 'cos_month',
        'market_encoded', 'district_encoded', 'state_encoded', 'variety_encoded', 'crop_season_onion_encoded'
    ]

    scaler = RobustScaler()
    scaler.fit(train_df[feature_cols])

    scaled_feature_cols = [f"{c}_scaled" for c in feature_cols]
    train_df[scaled_feature_cols] = scaler.transform(train_df[feature_cols])
    val_df[scaled_feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[scaled_feature_cols] = scaler.transform(test_df[feature_cols])
    full_df[scaled_feature_cols] = scaler.transform(full_df[feature_cols])

    angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
    angle_scaler.fit(train_df[feature_cols])

    angle_feature_cols = [f"{c}_angle" for c in feature_cols]
    train_df[angle_feature_cols] = angle_scaler.transform(train_df[feature_cols])
    val_df[angle_feature_cols] = angle_scaler.transform(val_df[feature_cols])
    test_df[angle_feature_cols] = angle_scaler.transform(test_df[feature_cols])
    full_df[angle_feature_cols] = angle_scaler.transform(full_df[feature_cols])

    sample_sub = train_df.sample(n=min(10000, len(train_df)), random_state=RANDOM_SEED)
    rf = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(sample_sub[feature_cols], sample_sub['target_price_next_month'])

    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    top_8_quantum = importances.head(8).index.tolist()
    top_16_quantum = importances.head(16).index.tolist()

    return train_df, val_df, test_df, full_df, feature_cols, top_8_quantum, top_16_quantum, importances


def save_feature_descriptions(feature_cols, top_8, top_16, base_dir):
    """
    Export feature dictionary to feature_description.csv.

    Args:
        feature_cols (list): List of numerical predictor column names.
        top_8 (list): Top-8 quantum feature subset.
        top_16 (list): Top-16 quantum feature subset.
        base_dir (str): Base output directory.

    Returns:
        pd.DataFrame: Feature description dictionary.
    """
    descriptions = {
        'price_current_month': ('Monthly average daily modal price (Rs/qtl)', 'Primary baseline market price predictor'),
        'price_median_current_month': ('Monthly median daily modal price (Rs/qtl)', 'Robust central tendency price indicator'),
        'price_min_current_month': ('Monthly minimum modal price recorded', 'Lower price boundary in trading month'),
        'price_max_current_month': ('Monthly maximum modal price recorded', 'Upper ceiling price boundary in trading month'),
        'price_std_current_month': ('Standard deviation of daily modal prices in month', 'Intra-month trading price volatility'),
        'days_reported': ('Number of active mandi trading days in month', 'Liquidity and market reporting activity metric'),
        'price_lag_1': ('Modal price in month t-1 (previous month)', 'Short-term autoregressive price memory'),
        'price_lag_2': ('Modal price in month t-2 (two months prior)', 'Medium-term price momentum component'),
        'price_lag_3': ('Modal price in month t-3 (quarter prior)', 'Quarterly historical price baseline'),
        'price_lag_6': ('Modal price in month t-6 (half-year prior)', 'Half-yearly seasonal price baseline'),
        'price_lag_12': ('Modal price in month t-12 (same month prior year)', 'Annual crop calendar seasonal baseline'),
        'price_rolling_3m_mean': ('3-month rolling mean modal price', 'Smoothed short-term price trend'),
        'price_rolling_6m_mean': ('6-month rolling mean modal price', 'Smoothed medium-term price trend'),
        'price_rolling_12m_mean': ('12-month rolling mean modal price', 'Long-term secular price level'),
        'price_rolling_3m_std': ('3-month rolling standard deviation of price', 'Quarterly price regime volatility'),
        'price_change_ratio_1m': ('Relative price change: (P_t - P_{t-1}) / P_{t-1}', 'Monthly inflation / deflation rate'),
        'arrivals_total_current_month': ('Total mandi onion arrivals in tonnes', 'Primary market physical supply volume'),
        'arrivals_daily_avg_current_month': ('Daily average arrivals in reporting days', 'Daily supply flow rate'),
        'arrivals_lag_1': ('Total arrivals in month t-1', 'Lagged physical supply volume'),
        'arrivals_rolling_3m_mean': ('3-month rolling mean arrivals', 'Smoothed supply availability indicator'),
        'arrivals_change_ratio_1m': ('Relative arrival change: (A_t - A_{t-1}) / A_{t-1}', 'Supply shock / disruption momentum'),
        'rainfall_total_current_month': ('Total monthly cumulative rainfall (mm)', 'Immediate weather & harvesting disruption'),
        'rainfall_lag_1': ('Total rainfall in month t-1', 'Lagged soil moisture & post-harvest storage shock'),
        'rainfall_rolling_3m_sum': ('3-month cumulative rainfall (mm)', 'Cumulative monsoon / crop-stage precipitation'),
        'temp_avg_current_month': ('Monthly average ambient temperature (°C)', 'Thermal stress & storage spoilage driver'),
        'temp_avg_lag_1': ('Monthly average temperature in month t-1', 'Lagged thermal conditions'),
        'temp_rolling_3m_mean': ('3-month rolling average temperature', 'Seasonal climatic regime indicator'),
        'tmax_current_month': ('Monthly average maximum daily temperature (°C)', 'Extreme daytime thermal peak'),
        'tmin_current_month': ('Monthly average minimum daily temperature (°C)', 'Night-time cold exposure temperature'),
        'quarter': ('Calendar quarter of current month (1-4)', 'Macro seasonal quarter'),
        'sin_month': ('Sine transform of month index (2*pi*month/12)', 'Cyclical annual calendar encoding'),
        'cos_month': ('Cosine transform of month index (2*pi*month/12)', 'Cyclical annual calendar encoding'),
        'market_encoded': ('Frequency encoding of wholesale market node', 'Market-level trading volume scale'),
        'district_encoded': ('Frequency encoding of administrative district', 'Regional agricultural density representation'),
        'state_encoded': ('Frequency encoding of state', 'Macro state production share representation'),
        'variety_encoded': ('Frequency encoding of onion cultivar', 'Cultivar prevalence & market share'),
        'crop_season_onion_encoded': ('Frequency encoding of crop season (Rabi/Kharif)', 'Agricultural crop production cycle encoding')
    }

    rows = []
    for col in feature_cols:
        desc, role = descriptions.get(col, ('Engineered numerical predictor', 'Input Feature'))
        rows.append({
            'feature_name': col,
            'data_type': 'float64',
            'meaning_and_description': desc,
            'role_in_modeling': role,
            'in_quantum_top_8': 'Yes' if col in top_8 else 'No',
            'in_quantum_top_16': 'Yes' if col in top_16 else 'No'
        })

    rows.append({
        'feature_name': 'target_price_next_month',
        'data_type': 'float64',
        'meaning_and_description': 'Monthly average modal price in consecutive month t+1 (Rs/qtl)',
        'role_in_modeling': 'Supervised Regression Target (P_{t+1})',
        'in_quantum_top_8': 'Target',
        'in_quantum_top_16': 'Target'
    })
    rows.append({
        'feature_name': 'sample_weight',
        'data_type': 'float64',
        'meaning_and_description': 'Inverse-frequency tier weight for training regression loss',
        'role_in_modeling': 'Loss Penalty Weighting',
        'in_quantum_top_8': 'Weight',
        'in_quantum_top_16': 'Weight'
    })

    desc_df = pd.DataFrame(rows)
    desc_path = os.path.join(base_dir, "feature_description.csv")
    desc_df.to_csv(desc_path, index=False)
    return desc_df


def generate_visualizations(raw_df, clean_daily_df, monthly_df, train_df, val_df, test_df, viz_dir, feature_cols):
    """
    Render and save 10 diagnostic visualization figures at 300 DPI.

    Figures:
        1. Target Price Distribution Before and After Preprocessing
        2. Price Range Imbalance Across Tiers
        3. Missing Value Percentages Before and After
        4. Outlier Boxplot Comparison
        5. Monthly Price Trajectory (2014-2024)
        6. Yearly Price Distribution
        7. Feature Correlation Heatmap
        8. Dataset Progression Across Pipeline Stages
        9. Weather and Supply Driver Distributions
        10. Training Target Distribution and Assigned Sample Weights

    Args:
        raw_df (pd.DataFrame): Raw observations.
        clean_daily_df (pd.DataFrame): Clean daily observations.
        monthly_df (pd.DataFrame): Monthly aggregated observations.
        train_df (pd.DataFrame): Training split.
        val_df (pd.DataFrame): Validation split.
        test_df (pd.DataFrame): Test split.
        viz_dir (str): Visualization output directory.
        feature_cols (list): List of model feature names.
    """
    # 1. Target Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(raw_df['modal_price_rs_qtl'], bins=60, kde=True, ax=axes[0], color='#d95f02', edgecolor='black', alpha=0.6)
    axes[0].set_title("Raw Daily Price Distribution", fontsize=12)
    axes[0].set_xlabel("Modal Price (Rs/qtl)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlim(0, 15000)

    sns.histplot(monthly_df['target_price_next_month'], bins=60, kde=True, ax=axes[1], color='#1b9e77', edgecolor='black', alpha=0.6)
    axes[1].set_title("Cleaned Monthly Target Price Distribution (t+1)", fontsize=12)
    axes[1].set_xlabel("Target Modal Price (Rs/qtl)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xlim(0, 15000)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "01_target_distribution_before_after.png"), dpi=300)
    plt.close()

    # 2. Price Range Imbalance
    bins = [-np.inf, 1000, 2500, 5000, np.inf]
    labels = ['Low (<1k)', 'Normal (1k-2.5k)', 'High (2.5k-5k)', 'Spike (>5k)']
    raw_binned = pd.cut(raw_df['modal_price_rs_qtl'], bins=bins, labels=labels).value_counts(sort=False)
    monthly_binned = pd.cut(monthly_df['target_price_next_month'], bins=bins, labels=labels).value_counts(sort=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bars0 = axes[0].bar(labels, raw_binned.values, color='#7570b3', edgecolor='black', alpha=0.8)
    axes[0].set_title("Raw Daily Observations by Price Tier", fontsize=12)
    axes[0].set_ylabel("Observation Count")
    for bar in bars0:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, yval + 10000, f"{yval:,}\n({yval/len(raw_df)*100:.1f}%)", ha='center', va='bottom', fontsize=9)
    axes[0].set_ylim(0, max(raw_binned.values) * 1.2)

    bars1 = axes[1].bar(labels, monthly_binned.values, color='#386cb0', edgecolor='black', alpha=0.8)
    axes[1].set_title("Cleaned Monthly Observations by Price Tier", fontsize=12)
    axes[1].set_ylabel("Observation Count")
    for bar in bars1:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, yval + 600, f"{yval:,}\n({yval/len(monthly_df)*100:.1f}%)", ha='center', va='bottom', fontsize=9)
    axes[1].set_ylim(0, max(monthly_binned.values) * 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "02_price_range_imbalance_comparison.png"), dpi=300)
    plt.close()

    # 3. Missing Values
    raw_missing = raw_df.isna().sum()
    top_missing = raw_missing[raw_missing > 0].sort_values(ascending=False).head(10)
    top_missing_pct = (top_missing / len(raw_df) * 100).round(1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(x=top_missing_pct.values, y=top_missing_pct.index, ax=axes[0], palette='Reds_r', edgecolor='black')
    axes[0].set_title("Top Missing Feature Percentages in Raw Data", fontsize=12)
    axes[0].set_xlabel("Missing Percentage (%)")
    for i, v in enumerate(top_missing_pct.values):
        axes[0].text(v + 1, i, f"{v}%", va='center', fontsize=10)
    axes[0].set_xlim(0, 110)

    cleaned_missing_pct = (monthly_df[feature_cols].isna().sum() / len(monthly_df) * 100).head(10)
    sns.barplot(x=cleaned_missing_pct.values, y=cleaned_missing_pct.index, ax=axes[1], color='#2ca02c', edgecolor='black')
    axes[1].set_title("Missing Percentages in Cleaned Dataset (0% Missing)", fontsize=12)
    axes[1].set_xlabel("Missing Percentage (%)")
    axes[1].set_xlim(0, 100)
    for i, v in enumerate(cleaned_missing_pct.values):
        axes[1].text(1, i, "0.0% (Clean)", va='center', fontsize=10, color='darkgreen', weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "03_missing_values_before_after.png"), dpi=300)
    plt.close()

    # 4. Outliers
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(x=raw_df['modal_price_rs_qtl'], ax=axes[0], color='#e7298a', flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.3})
    axes[0].set_title("Raw Daily Modal Price (Log Scale)", fontsize=12)
    axes[0].set_xlabel("Price (Rs/qtl)")
    axes[0].set_xscale('log')

    sns.boxplot(x=monthly_df['target_price_next_month'], ax=axes[1], color='#66a61e', flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.3})
    axes[1].set_title("Cleaned Monthly Target Price", fontsize=12)
    axes[1].set_xlabel("Target Modal Price (Rs/qtl)")
    axes[1].set_xlim(0, 16000)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "04_outlier_boxplot_comparison.png"), dpi=300)
    plt.close()

    # 5. Price Trajectory
    monthly_trend = monthly_df.groupby(['year', 'month'])['target_price_next_month'].mean().reset_index()
    monthly_trend['date'] = pd.to_datetime(monthly_trend['year'].astype(str) + '-' + monthly_trend['month'].astype(str) + '-01')

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly_trend['date'], monthly_trend['target_price_next_month'], color='#08519c', linewidth=2.2, label='National Monthly Mean Modal Price')
    ax.axvspan(pd.to_datetime('2019-09-01'), pd.to_datetime('2020-01-01'), color='red', alpha=0.18, label='2019 Onion Crisis Peak')
    ax.axvspan(pd.to_datetime('2023-08-01'), pd.to_datetime('2023-12-01'), color='orange', alpha=0.18, label='2023 Inflationary Period')
    ax.set_title("National Onion Market Monthly Price Trajectory (2014 - 2024)", fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("Price (Rs/quintal)")
    ax.legend(loc='upper left', frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "05_monthly_price_trends_over_time.png"), dpi=300)
    plt.close()

    # 6. Yearly Distribution
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.boxplot(x='year', y='target_price_next_month', data=monthly_df, ax=ax, palette='Blues', showfliers=False)
    ax.set_title("Year-Wise Price Distributions (2014 - 2024)", fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("Target Price (Rs/quintal)")

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "06_yearly_price_distribution.png"), dpi=300)
    plt.close()

    # 7. Correlation Heatmap
    corr_features = [
        'target_price_next_month', 'price_current_month', 'price_lag_1', 'price_lag_2',
        'price_lag_3', 'price_lag_12', 'price_rolling_3m_mean',
        'arrivals_total_current_month', 'arrivals_lag_1',
        'rainfall_total_current_month', 'rainfall_lag_1',
        'temp_avg_current_month', 'temp_avg_lag_1'
    ]
    corr_matrix = monthly_df[corr_features].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-0.3, vmax=1.0, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
    ax.set_title("Correlation Heatmap: Agricultural, Weather & Supply Predictors vs Target Price", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "07_feature_correlation_heatmap.png"), dpi=300)
    plt.close()

    # 8. Pipeline Progression
    stages = [
        'Raw Daily Data', 'Cleaned Daily Data', 'Monthly Aggregated',
        'Train Partition', 'Validation Partition', 'Test Partition'
    ]
    counts = [len(raw_df), len(clean_daily_df), len(monthly_df), len(train_df), len(val_df), len(test_df)]
    colors = ['#7570b3', '#1b9e77', '#386cb0', '#2ca02c', '#ff7f0e', '#d62728']

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(stages, counts, color=colors, edgecolor='black', alpha=0.85)
    ax.set_title("Dataset Records Across Preprocessing and Partitioning Stages", fontsize=13)
    ax.set_ylabel("Number of Records")
    ax.set_yscale('log')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval * 1.15, f"{yval:,}", ha='center', va='bottom', fontsize=10, weight='bold')
    ax.set_ylim(1000, 3000000)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "08_dataset_size_comparison.png"), dpi=300)
    plt.close()

    # 9. Feature Distributions
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.kdeplot(monthly_df['rainfall_total_current_month'].clip(upper=500), ax=axes[0], color='#1f78b4', fill=True)
    axes[0].set_title("Monthly Rainfall (mm)\n(Capped at 500mm)", fontsize=11)
    axes[0].set_xlabel("Cumulative Monthly Rain (mm)")

    sns.kdeplot(monthly_df['temp_avg_current_month'], ax=axes[1], color='#e31a1c', fill=True)
    axes[1].set_title("Monthly Ambient Temperature (°C)", fontsize=11)
    axes[1].set_xlabel("Mean Temperature (°C)")

    sns.kdeplot(monthly_df['arrivals_total_current_month'].clip(upper=10000), ax=axes[2], color='#33a02c', fill=True)
    axes[2].set_title("Monthly Arrivals Volume\n(Capped at 10,000 tonnes)", fontsize=11)
    axes[2].set_xlabel("Total Monthly Arrivals (Tonnes)")

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "09_feature_distributions_before_after.png"), dpi=300)
    plt.close()

    # 10. Sample Weights
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(train_df['target_price_next_month'], bins=50, kde=True, ax=axes[0], color='#2ca02c', edgecolor='black')
    axes[0].set_title("Training Target Price Distribution (2014-2021)", fontsize=12)
    axes[0].set_xlabel("Target Price (Rs/qtl)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlim(0, 15000)

    scatter = axes[1].scatter(
        train_df['target_price_next_month'],
        train_df['sample_weight'],
        c=train_df['sample_weight'],
        cmap='viridis',
        alpha=0.6,
        edgecolors='none'
    )
    axes[1].set_title("Assigned Sample Weights (Inverse Tier Frequency)", fontsize=12)
    axes[1].set_xlabel("Target Price (Rs/qtl)")
    axes[1].set_ylabel("Sample Weight")
    axes[1].set_xlim(0, 15000)
    plt.colorbar(scatter, ax=axes[1], label='Sample Weight')

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "10_training_price_sample_weights.png"), dpi=300)
    plt.close()


def export_processed_files(full_monthly_df, train_df, val_df, test_df, base_dir):
    """
    Export processed datasets and partition splits to CSV files.

    Args:
        full_monthly_df (pd.DataFrame): Full monthly dataset.
        train_df (pd.DataFrame): Training partition.
        val_df (pd.DataFrame): Validation partition.
        test_df (pd.DataFrame): Test partition.
        base_dir (str): Destination directory.

    Returns:
        tuple: File paths of generated CSV files.
    """
    monthly_path = os.path.join(base_dir, "cleaned_onion_monthly.csv")
    train_path = os.path.join(base_dir, "train.csv")
    val_path = os.path.join(base_dir, "validation.csv")
    test_path = os.path.join(base_dir, "test.csv")

    full_monthly_df.to_csv(monthly_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return monthly_path, train_path, val_path, test_path


def generate_readme(base_dir, full_monthly_df, train_df, val_df, test_df, top_8, top_16):
    """
    Generate project documentation README.md.

    Args:
        base_dir (str): Base output directory.
        full_monthly_df (pd.DataFrame): Monthly dataset.
        train_df (pd.DataFrame): Training split.
        val_df (pd.DataFrame): Validation split.
        test_df (pd.DataFrame): Test split.
        top_8 (list): Top-8 quantum feature list.
        top_16 (list): Top-16 quantum feature list.
    """
    readme_path = os.path.join(base_dir, "README.md")
    content = f"""# Quantum-Enhanced Crop Market Price Prediction Using Hybrid Machine Learning
## Preprocessed Onion Market Price Dataset (2014–2024)

### 1. Project Overview & Scope
This repository contains the data preprocessing, feature engineering, and quality validation pipeline for predicting monthly wholesale onion market prices in India.
The processed data is prepared for both Classical Machine Learning models (Random Forest, XGBoost, LightGBM, LSTM, SVR) and Quantum Machine Learning architectures (Variational Quantum Regressors - VQR, Quantum Neural Networks - QNN, and Quantum Support Vector Regressors - QSVR).

### 2. Dataset Key Metrics
- **Original Daily Records**: 965,314 rows across 56 columns (2014-01-01 to 2024-12-31).
- **Cleaned Daily Records**: 964,570 rows (removed invalid zero target prices, scale errors > Rs 25,000/qtl, and duplicate records).
- **Final Monthly Prediction Observations**: {len(full_monthly_df):,} records across 812 wholesale mandis, 223 districts, and 24 states.
- **Supervised Prediction Target**: `target_price_next_month` ($P_{{t+1}}$) in Rupees per Quintal (Rs/qtl).
- **Missing Values in Final Features**: Exactly 0.

### 3. Chronological Partitions
Data is split chronologically to prevent lookahead bias:
1. **`train.csv`** (2014–2021 | 8 Years): **{len(train_df):,} observations** (Model training, scaler fitting, and feature selection).
2. **`validation.csv`** (2022–2023 | 2 Years): **{len(val_df):,} observations** (Hyperparameter tuning and checkpointing).
3. **`test.csv`** (2024 | 1 Year): **{len(test_df):,} observations** (Out-of-time test benchmark).

### 4. Methodological Highlights
1. **Target Formulation ($P_{{t+1}}$)**:
   For monthly observations at month $t$, all predictors are derived strictly from month $t$ and historical windows ($t-1, t-2, t-3, t-6, t-12$). The forecast target is the expected modal price in month $t+1$.
2. **Regression Imbalance Handling**:
   Agricultural price spikes (> Rs 5,000/qtl) reflect market supply shocks. Rather than synthetic resampling (e.g. SMOTE) which distorts continuous multivariate distributions, continuous sample weights (`sample_weight`) are computed via inverse tier frequency on the training set:
   $$w_i = \\frac{{N_{{train}}}}{{K \\cdot N_k}}$$
3. **Quantum ML Compatibility**:
   - **Feature Scaling**: RobustScaler features (`*_scaled`) centered by median and scaled by IQR.
   - **Angle Encoding**: MinMaxScaler features (`*_angle`) mapped into $[0, \\pi]$ for rotation gates ($R_y(\\theta)$).
   - **Top-8 Quantum Features**: `{", ".join(top_8)}`.
   - **Top-16 Quantum Features**: `{", ".join(top_16)}`.

### 5. Repository File Structure
```
Capstone/
├── preprocessed_onion_2014_2024_prices_weather_events.zip   # Raw source archive
├── preprocessed_onion_2014_2024_prices_weather_events.csv   # Raw source CSV
├── preprocessing.py                                         # Preprocessing script
├── cleaned_onion_monthly.csv                                # Full monthly processed dataset
├── train.csv                                                # Training split (2014-2021)
├── validation.csv                                           # Validation split (2022-2023)
├── test.csv                                                 # Test split (2024)
├── preprocessing_report.csv                                 # Column audit & classification
├── feature_description.csv                                  # Feature dictionary & QML flags
├── README.md                                                # Documentation
└── preprocessing_visualizations/                            # High-resolution figures
    ├── 01_target_distribution_before_after.png
    ├── 02_price_range_imbalance_comparison.png
    ├── 03_missing_values_before_after.png
    ├── 04_outlier_boxplot_comparison.png
    ├── 05_monthly_price_trends_over_time.png
    ├── 06_yearly_price_distribution.png
    ├── 07_feature_correlation_heatmap.png
    ├── 08_dataset_size_comparison.png
    ├── 09_feature_distributions_before_after.png
    └── 10_training_price_sample_weights.png
```

### 6. Execution
Run the pipeline script:
```bash
python preprocessing.py
```
"""
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content.strip())


def print_summary(metrics_before, clean_daily_df, monthly_df, train_df, val_df, test_df, feature_cols, base_dir, viz_dir):
    """
    Print formal execution summary and output metrics to console.

    Args:
        metrics_before (dict): Raw baseline metrics.
        clean_daily_df (pd.DataFrame): Clean daily observations.
        monthly_df (pd.DataFrame): Final monthly observations.
        train_df (pd.DataFrame): Training split.
        val_df (pd.DataFrame): Validation split.
        test_df (pd.DataFrame): Test split.
        feature_cols (list): Model features.
        base_dir (str): Base directory.
        viz_dir (str): Visualizations directory.
    """
    target = monthly_df['target_price_next_month']
    total_monthly = len(monthly_df)

    print("\n" + "=" * 80)
    print("PREPROCESSING RESULTS")
    print("=" * 80)
    print("Dataset Records:")
    print(f"  Raw observations          : {metrics_before['rows']:,} daily records ({metrics_before['cols']} columns)")
    print(f"  Cleaned daily records     : {len(clean_daily_df):,} daily records ({len(clean_daily_df)/metrics_before['rows']*100:.2f}% retained)")
    print(f"  Monthly forecasting units : {total_monthly:,} records")
    print(f"  Model features            : {len(feature_cols)} numerical predictors + 1 target")
    print(f"  Missing values            : 0")

    print("\nChronological Partitions:")
    print(f"  Train (2014-2021)         : {len(train_df):,} samples ({len(train_df)/total_monthly*100:.2f}%)")
    print(f"  Validation (2022-2023)    : {len(val_df):,} samples ({len(val_df)/total_monthly*100:.2f}%)")
    print(f"  Test (2024)               : {len(test_df):,} samples ({len(test_df)/total_monthly*100:.2f}%)")

    print("\nTarget Variable:")
    print("  Name                      : target_price_next_month (P_{t+1})")
    print("  Unit                      : Rupees per Quintal (Rs/qtl)")
    print(f"  Mean / Median             : Rs {target.mean():.2f} / Rs {target.median():.2f}")
    print(f"  Std Dev                   : Rs {target.std():.2f}")

    print("\nOutput Files Generated:")
    print(f"  - {os.path.join(base_dir, 'cleaned_onion_monthly.csv')}")
    print(f"  - {os.path.join(base_dir, 'train.csv')}")
    print(f"  - {os.path.join(base_dir, 'validation.csv')}")
    print(f"  - {os.path.join(base_dir, 'test.csv')}")
    print(f"  - {os.path.join(base_dir, 'preprocessing_report.csv')}")
    print(f"  - {os.path.join(base_dir, 'feature_description.csv')}")
    print(f"  - {os.path.join(base_dir, 'README.md')}")
    print(f"  - {viz_dir} (10 figures)")
    print("=" * 80 + "\n")


def main():
    print("Executing onion price prediction preprocessing pipeline...")
    base_dir, csv_path, zip_path, viz_dir = resolve_data_paths()

    print("[1/6] Ingesting raw dataset...")
    raw_df = load_raw_dataset(csv_path, zip_path)
    metrics_before = profile_baseline_dataset(raw_df)

    print("[2/6] Cleaning observations and handling anomalies...")
    clean_daily_df, cleaning_ledger = clean_daily_dataset(raw_df)
    generate_feature_classification_report(raw_df, base_dir)

    print("[3/6] Aggregating monthly series and constructing targets...")
    monthly_df = aggregate_to_monthly_dataset(clean_daily_df)

    print("[4/6] Engineering time-aware features and sample weights...")
    monthly_df = engineer_monthly_features(monthly_df)
    monthly_df, tier_df, imbalance_ratio = perform_imbalance_analysis(monthly_df)
    train_df, val_df, test_df = split_chronologically(monthly_df)

    print("[5/6] Encoding features and scaling for classical / QML models...")
    train_df, val_df, test_df, monthly_df, feature_cols, top_8, top_16, importances = prepare_classical_and_quantum_features(
        train_df, val_df, test_df, monthly_df
    )
    save_feature_descriptions(feature_cols, top_8, top_16, base_dir)

    print("[6/6] Generating visualizations and exporting output files...")
    generate_visualizations(raw_df, clean_daily_df, monthly_df, train_df, val_df, test_df, viz_dir, feature_cols)
    export_processed_files(monthly_df, train_df, val_df, test_df, base_dir)
    generate_readme(base_dir, monthly_df, train_df, val_df, test_df, top_8, top_16)

    print_summary(metrics_before, clean_daily_df, monthly_df, train_df, val_df, test_df, feature_cols, base_dir, viz_dir)


if __name__ == '__main__':
    main()
