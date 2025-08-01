import pandas as pd
import numpy as np

from patterns import *
from configuration import *


################################# FEATURES ####################################

def add_hour_column(df):
    """
    Adds an 'hour' column based on the specified datetime column.
    Assumes the column is either a datetime object or a string that can be converted to datetime.
    """
    time_col='time'
    
    # Ensure the time column is datetime dtype
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Extract the hour
    df['hour'] = df[time_col].dt.hour
    
    return df

def add_prev_day_high_low(df):
    """
    Adds PrevDayHigh and PrevDayLow columns based on the previous trading day's high and low.
    Assumes df has 'date', 'high', and 'low' columns, and is sorted by time.
    """
    # Ensure 'date' is datetime if it's not already
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Compute high and low for each day
    daily_high_low = df.groupby('date').agg({'high': 'max', 'low': 'min'}).shift(1)
    daily_high_low.columns = ['PrevDayHigh', 'PrevDayLow']

    # Map previous day's high/low to each row based on the current date
    df = df.merge(daily_high_low, how='left', left_on='date', right_index=True)

    return df

def compute_value_area(prices, volumes, value_area_percent=0.7):
    """
    Computes Value Area High and Low (VAH, VAL) from price and volume arrays.
    Assumes input arrays are from one full day.
    """
    # Round prices to nearest tick size (e.g., 0.25 or 1) for bucketing
    rounded_prices = prices.round(0)  # adjust rounding if needed

    # Create volume profile: total volume per rounded price level
    volume_profile = pd.Series(volumes).groupby(rounded_prices).sum()

    # Sort by volume descending to find where most volume occurred
    sorted_vp = volume_profile.sort_values(ascending=False)

    total_volume = sorted_vp.sum()
    target_volume = total_volume * value_area_percent

    cum_volume = 0
    value_area_prices = []

    for price, vol in sorted_vp.items():
        cum_volume += vol
        value_area_prices.append(price)
        if cum_volume >= target_volume:
            break

    return max(value_area_prices), min(value_area_prices)


def add_prev_value_area(df):
    df = df.copy()
    
    # Ensure 'date' column exists (should already be from import_df)
    if 'date' not in df.columns:
        df['date'] = df.index.date

    df['prev_VA_high'] = np.nan
    df['prev_VA_low'] = np.nan

    grouped = df.groupby('date')
    prev_day_va = {}

    for current_date, group in grouped:
        prev_date = current_date - pd.Timedelta(days=1)
        if prev_date in grouped.groups:
            prev_group = grouped.get_group(prev_date)
            vah, val = compute_value_area(prev_group['close'], prev_group['Volume'])
            prev_day_va[current_date] = (vah, val)

    for date, (vah, val) in prev_day_va.items():
        df.loc[df['date'] == date, 'prev_VA_high'] = vah
        df.loc[df['date'] == date, 'prev_VA_low'] = val

    return df


def add_ma_flags(df, threshold):
    """
    Adds categorical columns for MA50 and MA20 comparisons:
      - 'MA50close', 'MA50high', 'MA50low'
      - 'MA20close', 'MA20high', 'MA20low'

    Each column contains:
      - 'Above' if value > MA + threshold
      - 'Below' if value < MA - threshold
      - 'Near' if within ±threshold of the MA
    """

    def categorize(price, ma):
        diff = price - ma
        if diff > threshold:
            return 'Above'
        elif diff < -threshold:
            return 'Below'
        else:
            return 'Near'

    # MA50 categories
    df['MA50close'] = df.apply(lambda row: categorize(row['close'], row['MA50']), axis=1)
    df['MA50high']  = df.apply(lambda row: categorize(row['high'], row['MA50']), axis=1)
    df['MA50low']   = df.apply(lambda row: categorize(row['low'], row['MA50']), axis=1)

    # MA20 categories
    df['MA20close'] = df.apply(lambda row: categorize(row['close'], row['MA20']), axis=1)
    df['MA20high']  = df.apply(lambda row: categorize(row['high'], row['MA20']), axis=1)
    df['MA20low']   = df.apply(lambda row: categorize(row['low'], row['MA20']), axis=1)

    return df


def add_vwap_flags(df, threshold):
    """
    Adds categorical columns:
      - 'VWAPclose': compares close to VWAP
      - 'VWAPhigh' : compares high to VWAP
      - 'VWAPlow'  : compares low to VWAP
      
    Each column will have one of:
      - 'Above' if price > VWAP + threshold
      - 'Below' if price < VWAP - threshold
      - 'Near'  if within ±threshold of VWAP
    """

    def categorize(value, vwap):
        diff = value - vwap
        if diff > threshold:
            return 'Above'
        elif diff < -threshold:
            return 'Below'
        else:
            return 'Near'

    df['VWAPclose'] = df.apply(lambda row: categorize(row['close'], row['VWAP']), axis=1)
    df['VWAPhigh'] = df.apply(lambda row: categorize(row['high'], row['VWAP']), axis=1)
    df['VWAPlow'] = df.apply(lambda row: categorize(row['low'], row['VWAP']), axis=1)

    return df



def add_macd_flag(df,threshold):
    """
    Adds a single column 'MACD_and_Signal' with values:
        - 'Above' if MACD Signal > MACD + threshold
        - 'Below' if MACD Signal < MACD - threshold
        - 'Near' if they are equal (within a small threshold)
    """

    conditions = [
        (df['MACD Signal'] > df['MACD'] + threshold),
        (df['MACD Signal'] < df['MACD'] - threshold)
    ]
    choices = ['Above', 'Below']
    
    df['MACD_and_Signal'] = np.select(conditions, choices, default='Near')
    return df






########################### SIGNALS - FILTERING ###############################
def add_OR_breakout_flag(df, daily_objects, column_name, check_low):
    """
    Flags candles where:
      - close > OR high
      - and (optionally) low < OR high if check_low is True
    """
    # Create a lookup for OR high by date
    or_high_by_date = {obj.date: obj.or_high for obj in daily_objects}
    
    # Default value for the new column
    df[column_name] = False

    # Apply logic row by row
    for idx, row in df.iterrows():
        date = row['date']
        or_high = or_high_by_date.get(date)
        if or_high is None:
            continue

        # Apply conditions based on check_low flag
        if check_low:
            condition = row['low'] < or_high and row['close'] > or_high
        else:
            condition = row['close'] > or_high

        if condition:
            df.at[idx, column_name] = True

    return df


def add_OR_breakdown_flag(df, daily_objects, column_name, check_high):
    """
    Flags candles where:
      - close < OR low
      - and (optionally) high > OR low if check_high is True
    """
    # Create a mapping from date to OR low
    or_low_map = {obj.date: obj.or_low for obj in daily_objects}

    # Map OR low to each row in the DataFrame
    df['or_low'] = df['date'].map(or_low_map)

    # Apply the condition
    if check_high:
        df[column_name] = (df['high'] > df['or_low']) & (df['close'] < df['or_low'])
    else:
        df[column_name] = df['close'] < df['or_low']

    # Drop the helper column
    df.drop(columns=['or_low'], inplace=True)

    return df



def add_new_day_high_flag(df, column_name):
    """
    Flags candles where:
      - the current candle's high is higher than any previous candle on the same day (new day high)
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for idx in range(len(df)):
        row = df.iloc[idx]
        date = row['date']

        # Get all prior rows for the same day
        prior_rows = df.iloc[:idx]
        prior_day_highs = prior_rows[prior_rows['date'] == date]['high']

        is_new_day_high = len(prior_day_highs) == 0 or row['high'] > prior_day_highs.max()
        if is_new_day_high:
            df.at[idx, column_name] = True

    return df


def add_new_day_low_flag(df, column_name):
    """
    Flags candles where:
      - the current candle's low is lower than any previous candle on the same day (new day low)
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for idx in range(len(df)):
        row = df.iloc[idx]
        date = row['date']

        # Get all prior rows for the same day
        prior_rows = df.iloc[:idx]
        prior_day_lows = prior_rows[prior_rows['date'] == date]['low']

        is_new_day_low = len(prior_day_lows) == 0 or row['low'] < prior_day_lows.min()
        if is_new_day_low:
            df.at[idx, column_name] = True

    return df

def add_prev_day_breakout_flag(df, column_name, check_low):
    """
    Flags candles where:
      - close > previous day high
      - and (optionally) low < previous day high if check_low is True
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for idx, row in df.iterrows():
        prev_day_high = row.get('PrevDayHigh')
        if pd.isna(prev_day_high):
            continue

        if check_low:
            condition = row['low'] < prev_day_high and row['close'] > prev_day_high
        else:
            condition = row['close'] > prev_day_high

        if condition:
            df.at[idx, column_name] = True

    return df

def add_prev_day_breakdown_flag(df, column_name, check_high):
    """
    Flags candles where:
      - close < previous day low
      - and (optionally) high > previous day low if check_high is True
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for idx, row in df.iterrows():
        prev_day_low = row.get('PrevDayLow')
        if pd.isna(prev_day_low):
            continue

        if check_high:
            condition = row['high'] > prev_day_low and row['close'] < prev_day_low
        else:
            condition = row['close'] < prev_day_low

        if condition:
            df.at[idx, column_name] = True

    return df

def add_prev_day_high_reversal_flag(df, column_name):
    """
    Flags candles where:
      - high > previous day high
      - close < previous day high
    Indicates a possible reversal at previous day's high.
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for idx, row in df.iterrows():
        prev_day_high = row.get('PrevDayHigh')
        if pd.isna(prev_day_high):
            continue

        if row['high'] > prev_day_high and row['close'] < prev_day_high:
            df.at[idx, column_name] = True

    return df

def add_prev_day_low_reversal_flag(df, column_name):
    """
    Flags candles where:
      - low < previous day low
      - close > previous day low
    Indicates a possible reversal at previous day's low.
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for idx, row in df.iterrows():
        prev_day_low = row.get('PrevDayLow')
        if pd.isna(prev_day_low):
            continue

        if row['low'] < prev_day_low and row['close'] > prev_day_low:
            df.at[idx, column_name] = True

    return df


def add_prev_va_high_breakout_flag(df, column_name, check_low):
    """
    Flags candles where:
      - close > previous day's VA high
      - and (optionally) low < previous day's VA high if check_low is True
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for idx, row in df.iterrows():
        prev_va_high = row.get('prev_VA_high')
        if pd.isna(prev_va_high):
            continue

        if check_low:
            condition = row['low'] < prev_va_high and row['close'] > prev_va_high
        else:
            condition = row['close'] > prev_va_high

        if condition:
            df.at[idx, column_name] = True

    return df


def add_prev_va_low_breakdown_flag(df, column_name, check_high):
    """
    Flags candles where:
      - close < previous day's VA low
      - and (optionally) high > previous day's VA low if check_high is True
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for idx, row in df.iterrows():
        prev_va_low = row.get('prev_VA_low')
        if pd.isna(prev_va_low):
            continue

        if check_high:
            condition = row['high'] > prev_va_low and row['close'] < prev_va_low
        else:
            condition = row['close'] < prev_va_low

        if condition:
            df.at[idx, column_name] = True

    return df


def add_prev_va_high_reversal_flag(df, column_name):
    """
    Flags candles where:
      - high > previous day's VA high
      - close < previous day's VA high
    Indicates a possible reversal at previous day's value area high.
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for idx, row in df.iterrows():
        prev_va_high = row.get('prev_VA_high')
        if pd.isna(prev_va_high):
            continue

        if row['high'] > prev_va_high and row['close'] < prev_va_high:
            df.at[idx, column_name] = True

    return df


def add_prev_va_low_reversal_flag(df, column_name):
    """
    Flags candles where:
      - low < previous day's VA low
      - close > previous day's VA low
    Indicates a possible reversal at previous day's value area low.
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for idx, row in df.iterrows():
        prev_va_low = row.get('prev_VA_low')
        if pd.isna(prev_va_low):
            continue

        if row['low'] < prev_va_low and row['close'] > prev_va_low:
            df.at[idx, column_name] = True

    return df

def add_or_high_reversal_flag(df, daily_objects, column_name):
    """
    Flags candles where:
      - high > OR high
      - close < OR high
    Indicates a possible reversal at the Opening Range high.
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    # OR high lookup by date
    or_high_by_date = {obj.date: obj.or_high for obj in daily_objects}

    for idx, row in df.iterrows():
        date = row['date']
        or_high = or_high_by_date.get(date)
        if or_high is None:
            continue

        if row['high'] > or_high and row['close'] < or_high:
            df.at[idx, column_name] = True

    return df

def add_or_low_reversal_flag(df, daily_objects, column_name):
    """
    Flags candles where:
      - low < OR low
      - close > OR low
    Indicates a possible reversal at the Opening Range low.
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    # OR low lookup by date
    or_low_by_date = {obj.date: obj.or_low for obj in daily_objects}

    for idx, row in df.iterrows():
        date = row['date']
        or_low = or_low_by_date.get(date)
        if or_low is None:
            continue

        if row['low'] < or_low and row['close'] > or_low:
            df.at[idx, column_name] = True

    return df


def add_bb_reversal_flags(df):
    """
    Adds boolean columns for reversals where:
      - high > BB upper band AND close < BB upper band (possible reversal down)
      - low < BB lower band AND close > BB lower band (possible reversal up)
    """
    upper_cols = ['BBUpper35', 'BBUpper3', 'BBUpper25', 'BBUpper2']
    lower_cols = ['BBLower35', 'BBLower3', 'BBLower25', 'BBLower2']

    # Reversal flags
    for col in upper_cols:
        df[f'ReversalDown_{col}'] = (df['high'] > df[col]) & (df['close'] < df[col])

    for col in lower_cols:
        df[f'ReversalUp_{col}'] = (df['low'] < df[col]) & (df['close'] > df[col])

    return df


def add_bullish_dip_flag(df, column_name, k):
    """
    Flags candles where:
      1. Previous k candles had higher highs and higher lows (trend up),
      2. Current candle makes a lower low than previous candle,
      3. Current candle has a long lower wick.
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for idx in range(k, len(df)):
        window = df.iloc[idx - k:idx]  # previous k candles
        current_row = df.iloc[idx]
        prev_row = df.iloc[idx - 1]

        # Condition 1: Previous k candles show uptrend (higher highs and lows)
        uptrend = all(
            window.iloc[i]['high'] < window.iloc[i + 1]['high'] and
            window.iloc[i]['low'] < window.iloc[i + 1]['low']
            for i in range(k - 1)
        )

        # Condition 2: Current candle makes a lower low than previous
        lower_low = current_row['low'] < prev_row['low']

        # Condition 3: Current candle has long lower wick
        wick_ok = hasLongLowerWick(current_row)

        if uptrend and lower_low and wick_ok:
            df.at[idx, column_name] = True

    return df

def add_bearish_dip_flag(df, column_name, k):
    """
    Flags candles where:
      1. Previous k candles had lower highs and lower lows (trend down),
      2. Current candle makes a higher high than previous candle,
      3. Current candle has a long upper wick.
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for idx in range(k, len(df)):
        window = df.iloc[idx - k:idx]  # previous k candles
        current_row = df.iloc[idx]
        prev_row = df.iloc[idx - 1]

        # Condition 1: Previous k candles show downtrend (lower highs and lows)
        downtrend = all(
            window.iloc[i]['high'] > window.iloc[i + 1]['high'] and
            window.iloc[i]['low'] > window.iloc[i + 1]['low']
            for i in range(k - 1)
        )

        # Condition 2: Current candle makes a higher high than previous
        higher_high = current_row['high'] > prev_row['high']

        # Condition 3: Current candle has long upper wick
        wick_ok = hasLongUpperWick(current_row)

        if downtrend and higher_high and wick_ok:
            df.at[idx, column_name] = True

    return df


############################## Pullbacks ######################################

def add_bullish_pullback_CCSL(df, column_name, sl):
    """
    Flags candles where the next candle's low is <= current candle's low + sl.
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for idx in range(len(df) - 1):  # skip last row to avoid index error
        current_row = df.iloc[idx]
        next_row = df.iloc[idx + 1]

        if next_row['date'] == current_row['date']:
            if next_row['low'] <= current_row['low'] + sl:
                df.at[idx, column_name] = True

    return df

def add_bearish_pullback_CCSL(df, column_name, sl):
    """
    Flags candles where the next candle's high is >= current candle's high - sl.
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for idx in range(len(df) - 1):  # skip last row to avoid index error
        current_row = df.iloc[idx]
        next_row = df.iloc[idx + 1]

        if next_row['date'] == current_row['date']:
            if next_row['high'] >= current_row['high'] - sl:
                df.at[idx, column_name] = True

    return df

def add_bullish_pullback_half_candle(df, column_name):
    """
    Flags candles where the next candle's low is <= 
    the current candle's low + 50% of current candle's range (high - low).
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for idx in range(len(df) - 1):  # skip last row to avoid index error
        current_row = df.iloc[idx]
        next_row = df.iloc[idx + 1]

        if next_row['date'] == current_row['date']:
            half_range_level = current_row['low'] + 0.5 * (current_row['high'] - current_row['low'])
            if next_row['low'] <= half_range_level:
                df.at[idx, column_name] = True

    return df

def add_bearish_pullback_half_candle(df, column_name):
    """
    Flags candles where the next candle's high is >= 
    the current candle's high - 50% of current candle's range (high - low).
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for idx in range(len(df) - 1):  # skip last row to avoid index error
        current_row = df.iloc[idx]
        next_row = df.iloc[idx + 1]

        if next_row['date'] == current_row['date']:
            half_range_level = current_row['high'] - 0.5 * (current_row['high'] - current_row['low'])
            if next_row['high'] >= half_range_level:
                df.at[idx, column_name] = True

    return df






############################ EXECUTIONS - DEPENDENT VARS ######################

def add_next_candle_bullish_flag_v1(df, column_name):
    """
    Flags candles where the next candle is bullish (close > open).
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for i in range(len(df) - 1):  # skip last row
        next_open = df.at[i + 1, 'open']
        next_close = df.at[i + 1, 'close']

        if next_close > next_open:
            df.at[i, column_name] = True

    return df

def add_next_candle_bullish_flag(df, column_name):
    """
    Flags candles where the next candle is bullish (close > open) 
    and occurs on the same day.
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for i in range(len(df) - 1):  # skip last row
        curr_date = df.at[i, 'date']
        next_date = df.at[i + 1, 'date']

        if curr_date == next_date:
            next_open = df.at[i + 1, 'open']
            next_close = df.at[i + 1, 'close']

            if next_close > next_open:
                df.at[i, column_name] = True

    return df



def add_next_candle_bearish_flag_v1(df, column_name):
    """
    Flags candles where the next candle is bearish (close < open).
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for i in range(len(df) - 1):  # skip last row
        next_open = df.at[i + 1, 'open']
        next_close = df.at[i + 1, 'close']

        if next_close < next_open:
            df.at[i, column_name] = True

    return df

def add_next_candle_bearish_flag(df, column_name):
    """
    Flags candles where the next candle is bearish (close < open)
    and occurs on the same day.
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for i in range(len(df) - 1):  # skip last row
        curr_date = df.at[i, 'date']
        next_date = df.at[i + 1, 'date']

        if curr_date == next_date:
            next_open = df.at[i + 1, 'open']
            next_close = df.at[i + 1, 'close']

            if next_close < next_open:
                df.at[i, column_name] = True

    return df


def add_next_candle_higher_high_flag_v1(df, column_name):
    """
    Flags candles where the next candle's high is higher than the current candle's high.
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for i in range(len(df) - 1):  # skip last row
        current_high = df.at[i, 'high']
        next_high = df.at[i + 1, 'high']

        if next_high > current_high:
            df.at[i, column_name] = True

    return df

def add_next_candle_higher_high_flag(df, column_name):
    """
    Flags candles where the next candle's high is higher than the current candle's high
    and occurs on the same day.
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for i in range(len(df) - 1):  # skip last row
        curr_date = df.at[i, 'date']
        next_date = df.at[i + 1, 'date']

        if curr_date == next_date:
            current_high = df.at[i, 'high']
            next_high = df.at[i + 1, 'high']

            if next_high > current_high:
                df.at[i, column_name] = True

    return df


def add_next_candle_lower_low_flag_v1(df, column_name):
    """
    Flags candles where the next candle's low is lower than the current candle's low.
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for i in range(len(df) - 1):  # skip last row
        current_low = df.at[i, 'low']
        next_low = df.at[i + 1, 'low']

        if next_low < current_low:
            df.at[i, column_name] = True

    return df

def add_next_candle_lower_low_flag(df, column_name):
    """
    Flags candles where the next candle's low is lower than the current candle's low
    and occurs on the same day.
    """
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False

    for i in range(len(df) - 1):  # skip last row
        curr_date = df.at[i, 'date']
        next_date = df.at[i + 1, 'date']

        if curr_date == next_date:
            current_low = df.at[i, 'low']
            next_low = df.at[i + 1, 'low']

            if next_low < current_low:
                df.at[i, column_name] = True

    return df


###################### mid candle entry, cc sl, tp - points ###################

def add_next_k_candle_breakout_by_mR_flag(df, k, m, column_name):
    """
    Flags candles where:
      - All next k candle lows are above current candle's low (no breakdown),
      - Any of the next k candle highs reaches at least m * R above the current candle's midpoint,
        where R = entry - low and entry = (high + low) / 2.
    """
    df = df.sort_values('time').reset_index(drop=True)
    n = len(df)
    df[column_name] = False

    for i in range(n - k):  # not enough future candles at the end
        current_low = df.at[i, 'low']
        current_high = df.at[i, 'high']

        entry = (current_high + current_low) / 2
        R = entry - current_low

        # Avoid division by zero or invalid R
        if R <= 0:
            continue

        target_price = entry + m * R

        future_lows = df.loc[i + 1:i + k, 'low']
        future_highs = df.loc[i + 1:i + k, 'high']

        if (future_lows > current_low).all() and (future_highs >= target_price).any():
            df.at[i, column_name] = True

    return df


def add_next_k_candle_breakdown_by_mR_flag(df, k, m, column_name):
    """
    Flags candles where:
      - All next k candle highs are below current candle's high (no breakout),
      - Any of the next k candle lows drops to at least m * R below the current candle's midpoint,
        where R = high - midpoint and entry = (high + low) / 2.
    """
    df = df.sort_values('time').reset_index(drop=True)
    n = len(df)
    df[column_name] = False

    for i in range(n - k):  # not enough future candles at the end
        current_high = df.at[i, 'high']
        current_low = df.at[i, 'low']

        entry = (current_high + current_low) / 2
        R = current_high - entry

        # Avoid division by zero or invalid R
        if R <= 0:
            continue

        target_price = entry - m * R

        future_highs = df.loc[i + 1:i + k, 'high']
        future_lows = df.loc[i + 1:i + k, 'low']

        if (future_highs < current_high).all() and (future_lows <= target_price).any():
            df.at[i, column_name] = True

    return df


def add_next_k_candle_breakout_by_tp_flag(df, k, tp, column_name):
    """
    Flags candles where:
      - All next k candle lows are above current candle's low (no breakdown),
      - Any of the next k candle highs reaches at least `tp` points above the current candle's midpoint.
    """
    df = df.sort_values('time').reset_index(drop=True)
    n = len(df)
    df[column_name] = False

    for i in range(n - k):  # not enough future candles at the end
        current_low = df.at[i, 'low']
        current_high = df.at[i, 'high']

        # Entry midpoint
        entry = (current_high + current_low) / 2
        target_price = entry + tp

        future_lows = df.loc[i + 1:i + k, 'low']
        future_highs = df.loc[i + 1:i + k, 'high']

        if (future_lows > current_low).all() and (future_highs >= target_price).any():
            df.at[i, column_name] = True

    return df


def add_next_k_candle_breakdown_by_tp_flag(df, k, tp, column_name):
    """
    Flags candles where:
      - All next k candle highs are below current candle's high (no fake-out),
      - Any of the next k candle lows drops at least `tp` points below the current candle's midpoint.
    """
    df = df.sort_values('time').reset_index(drop=True)
    n = len(df)
    df[column_name] = False

    for i in range(n - k):  # not enough candles beyond this point
        current_high = df.at[i, 'high']
        current_low = df.at[i, 'low']

        # Midpoint of the current candle (entry reference)
        entry = (current_high + current_low) / 2
        target_price = entry - tp

        future_highs = df.loc[i + 1:i + k, 'high']
        future_lows = df.loc[i + 1:i + k, 'low']

        if (future_highs < current_high).all() and (future_lows <= target_price).any():
            df.at[i, column_name] = True

    return df




############ next k candles BREAKOUT with CC SL=CC and  TP=m*R ################

def add_next_k_candle_breakout_flag(df, k, m, column_name):
    """
    Flags candles where:
      - all next k candle lows are above current low
      - any next k candle high reaches m * R profit (R = close - low)
    
    """
    df = df.sort_values('time').reset_index(drop=True)
    n = len(df)
    df[column_name] = False

    for i in range(n - k):  # last k rows can't have enough future candles
        current_low = df.at[i, 'low']
        current_close = df.at[i, 'close']
        R = current_close - current_low
        target_price = current_close + m * R

        future_lows = df.loc[i+1:i+k, 'low']
        future_highs = df.loc[i+1:i+k, 'high']

        if (future_lows > current_low).all() and (future_highs >= target_price).any():
            df.at[i, column_name] = True

    return df

def add_next_k_candle_breakdown_flag(df, k, m, column_name):
    """
    Flags candles where:
      - all next k candle highs are below current high
      - any next k candle low drops to or below current close - m * R
        where R = high - close
    
    """
    df = df.sort_values('time').reset_index(drop=True)
    n = len(df)
    df[column_name] = False

    for i in range(n - k):  # skip last k rows (not enough future data)
        current_high = df.at[i, 'high']
        current_close = df.at[i, 'close']
        R = current_high - current_close
        target_price = current_close - m * R

        future_highs = df.loc[i+1:i+k, 'high']
        future_lows = df.loc[i+1:i+k, 'low']

        if (future_highs < current_high).all() and (future_lows <= target_price).any():
            df.at[i, column_name] = True

    return df

def add_next_k_candle_breakout_with_pullback_flag(df, column_name, k, m, sl):
    """
    Flags candles where:
      - all next k candle lows are above current low (no stop-out)
      - any next k candle high >= (current low + sl + m * sl)
    """
    df = df.sort_values('time').reset_index(drop=True)
    n = len(df)
    df[column_name] = False

    for i in range(n - k):  # last k candles can't be evaluated
        current_low = df.at[i, 'low']
        target_price = current_low + sl + m * sl

        # Use iloc to ensure exact k rows are selected
        future_lows = df.iloc[i+1:i+1+k]['low']
        future_highs = df.iloc[i+1:i+1+k]['high']

        no_stopout = (future_lows > current_low).all()
        reached_target = (future_highs >= target_price).any()

        if no_stopout and reached_target:
            df.at[i, column_name] = True

    return df


def add_next_k_candle_breakdown_with_pullback_flag(df, column_name, k, m, sl):
    """
    Flags candles where:
      - all next k candle highs are below current high (no stop-out)
      - any next k candle low <= (current high - sl - m * sl)
    """
    df = df.sort_values('time').reset_index(drop=True)
    n = len(df)
    df[column_name] = False

    for i in range(n - k):  # ensure we have k future candles
        current_high = df.at[i, 'high']
        target_price = current_high - sl - m * sl  # profit target

        future_highs = df.iloc[i+1:i+1+k]['high']
        future_lows = df.iloc[i+1:i+1+k]['low']

        no_stopout = (future_highs < current_high).all()
        reached_target = (future_lows <= target_price).any()

        if no_stopout and reached_target:
            df.at[i, column_name] = True

    return df






############ next candle BREAKOUTS with fixed SL and TP #######################

def add_next_bullish_breakout_flag(df,tp_points,sl_points,column_name):
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False  # default

    # Shift necessary columns by -1 for "next candle"
    df['next_open'] = df['open'].shift(-1)
    df['next_close'] = df['close'].shift(-1)
    df['next_low'] = df['low'].shift(-1)
    df['next_date'] = df['date'].shift(-1)

    # Conditions for same-day and breakout pattern
    same_day = df['date'] == df['next_date']
    breakout_condition = (
        (df['next_close'] - df['next_open'] > tp_points) &
        (df['next_open'] - df['next_low'] < sl_points)
    )

    df[column_name] = same_day & breakout_condition

    # Drop helper columns if you don't need them
    df.drop(columns=['next_open', 'next_close', 'next_low', 'next_date'], inplace=True)

    return df

def add_next_bearish_breakdown_flag(df, tp_points, sl_points, column_name):
    df = df.sort_values('time').reset_index(drop=True)
    df[column_name] = False  # default

    # Shift necessary columns by -1 for "next candle"
    df['next_open'] = df['open'].shift(-1)
    df['next_close'] = df['close'].shift(-1)
    df['next_high'] = df['high'].shift(-1)
    df['next_date'] = df['date'].shift(-1)

    # Conditions for same-day and breakdown pattern
    same_day = df['date'] == df['next_date']
    breakdown_condition = (
        (df['next_open'] - df['next_close'] > tp_points) &
        (df['next_high'] - df['next_open'] < sl_points)
    )

    df[column_name] = same_day & breakdown_condition

    # Drop helper columns
    df.drop(columns=['next_open', 'next_close', 'next_high', 'next_date'], inplace=True)

    return df

###############################################################################

def add_features(df, daily_objects):
    # to test
    df = add_hour_column(df)
    df = add_prev_day_high_low(df)  
    df = add_ma_flags(df,10) 
    df = add_vwap_flags(df,10) 
    df = add_macd_flag(df,1.0)

    return df

def add_signals(df, daily_objects):

    df = add_new_day_high_flag(df, "New Day High")
    df = add_new_day_low_flag(df, "New Day Low")
    df = add_OR_breakout_flag(df, daily_objects, "OR Breakout", True)
    df = add_OR_breakdown_flag(df, daily_objects, "OR Breakdown", True)
    df = add_or_high_reversal_flag(df, daily_objects, "OR High Reversal")
    df = add_or_low_reversal_flag(df, daily_objects, "OR Low Reversal")
    df = add_prev_day_breakout_flag(df, "PrevDayHigh Breakout", True)
    df = add_prev_day_breakdown_flag(df, "PrevDayLow Breakdown", True)
    df = add_prev_day_high_reversal_flag(df, "PrevDayHigh Reversal")
    df = add_prev_day_low_reversal_flag(df, "PrevDayLow Reversal")
    # df = add_prev_va_high_breakout_flag(df, "PrevVAH Breakout", True) - to test
    # df = add_prev_va_low_breakdown_flag(df, "PrevVAL Breakdown", True) - to test
    # df = add_prev_va_high_reversal_flag(df, "PrevVAH Reversal") - to test
    # df = add_prev_va_low_reversal_flag(df, "PrevVAL Reversal") - to test

    df = add_bb_reversal_flags(df) # to test
    # df = add_bullish_dip_flag(df, "Bullish Trend Dip", 3) # to test
    # df = add_bearish_dip_flag(df, "Bearish Trend Dip", 3) # to test
    
    return df

def add_pullbacks(df, daily_objects):

    # df = add_bullish_pullback_CCSL(df, "Bullish Pullback 10p to CC", sl)
    # df = add_bullish_pullback_CCSL(df, "Bearish Pullback 10p to CC", sl)
    # df = add_bullish_pullback_half_candle(df, "Bullish Pullback half C")
    # df = add_bearish_pullback_half_candle(df, "Bearish Pullback half C")
    
    return df



def add_executions(df, daily_objects):
    # Add executions - dependent vars
    
    # df = add_next_bullish_breakout_flag(df,20,20,'Bull_EO_1c_SL20_TP20')
    # df = add_next_bearish_breakdown_flag(df,20,20,'Bear_EO_1c_SL20_TP20')
    # df = add_next_bullish_breakout_flag(df,40,20,'Bull_EO_1c_SL20_TP40')
    # df = add_next_bearish_breakdown_flag(df,40,20,'Bear_EO_1c_SL20_TP40')
    
    # df = add_next_k_candle_breakout_flag(df, 1, 1, 'Bull_EO_1c_SLCC_TP1R')
    # df = add_next_k_candle_breakdown_flag(df, 1, 1, 'Bear_EO_1c_SLCC_TP1R')
    # df = add_next_k_candle_breakout_flag(df, 2, 2, 'Bull_EO_2c_SLCC_TP2R')
    # df = add_next_k_candle_breakdown_flag(df, 2, 2, 'Bear_EO_2c_SLCC_TP2R')
    
    # df =  add_next_k_candle_breakout_with_pullback_flag(df, "Bull_ECC10_1c_SLCC_TP1R", 1, 1, sl)
    # df =  add_next_k_candle_breakdown_with_pullback_flag(df, "Bear_ECC10_1c_SLCC_TP1R", 1, 1, sl)
    # df =  add_next_k_candle_breakout_with_pullback_flag(df, "Bull_ECC10_1c_SLCC_TP2R", 1, 2, sl)
    # df =  add_next_k_candle_breakdown_with_pullback_flag(df, "Bear_ECC10_1c_SLCC_TP2R", 1, 2, sl)
    # df =  add_next_k_candle_breakout_with_pullback_flag(df, "Bull_ECC10_1c_SLCC_TP3R", 1, 3, sl)
    # df =  add_next_k_candle_breakdown_with_pullback_flag(df, "Bear_ECC10_1c_SLCC_TP3R", 1, 3, sl)
    
   
    # df = add_next_k_candle_breakout_by_mR_flag(df, 1, 1, "Bull_EMidC_1c_SLCC_TP1R")
    # df = add_next_k_candle_breakdown_by_mR_flag(df, 1, 1, "Bear_EMidC_1c_SLCC_TP1R")
    # df = add_next_k_candle_breakout_by_mR_flag(df, 1, 2, "Bull_EMidC_1c_SLCC_TP2R")
    # df = add_next_k_candle_breakdown_by_mR_flag(df, 1, 2, "Bear_EMidC_1c_SLCC_TP2R")
    # df = add_next_k_candle_breakout_by_mR_flag(df, 2, 1, "Bull_MidCE_2c_SLCC_TP1R")
    # df = add_next_k_candle_breakdown_by_mR_flag(df, 2, 1, "Bear_MidCE_2c_SLCC_TP1R")
    # df = add_next_k_candle_breakout_by_mR_flag(df, 2, 2, "Bull_MidCE_2c_SLCC_TP2R")
    # df = add_next_k_candle_breakdown_by_mR_flag(df, 2, 2, "Bear_MidCE_2c_SLCC_TP2R")
    
    # df = add_next_k_candle_breakout_by_tp_flag(df, 1, 10, "Bull_MidCE_1c_SLCC_TP10")
    # df = add_next_k_candle_breakdown_by_tp_flag(df, 1, 10, "Bear_MidCE_1c_SLCC_TP10")
    # df = add_next_k_candle_breakout_by_tp_flag(df, 1, 20, "Bull_MidCE_1c_SLCC_TP20")
    # df = add_next_k_candle_breakdown_by_tp_flag(df, 1, 20, "Bear_MidCE_1c_SLCC_TP20")
    # df = add_next_k_candle_breakout_by_tp_flag(df, 2, 20, "Bull_MidCE_3c_SLCC_TP30")
    # df = add_next_k_candle_breakdown_by_tp_flag(df, 2, 20, "Bear_MidCE_3c_SLCC_TP30")
    # df = add_next_k_candle_breakout_by_tp_flag(df, 2, 30, "Bull_MidCE_5c_SLCC_TP50")
    # df = add_next_k_candle_breakdown_by_tp_flag(df, 2, 30, "Bear_MidCE_5c_SLCC_TP50")
    
    df = add_next_candle_bullish_flag(df, "Next Bullish")
    df = add_next_candle_bearish_flag(df, "Next Bearish")
    # df = add_next_candle_higher_high_flag(df, "Next HH")
    # df = add_next_candle_lower_low_flag(df, "Next LL")
    
    return df






