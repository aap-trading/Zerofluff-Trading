import pandas as pd

from objects import *
from patterns import *
from configuration import *

def import_df(file_path):
    df = pd.read_csv(file_path, parse_dates=['time'])
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df['time'] = df['time'].dt.tz_convert('US/Mountain')
    df['time'] = df['time'].dt.tz_localize(None)
    df.set_index('time', inplace=True)
    df['date'] = df.index.date
    return df


def add_vix_to_df(df1, vix_df):

    df1 = df1.copy()
    vix_df = vix_df.copy()

    # Handle index-as-time
    if 'time' not in df1.columns and df1.index.name == 'time':
        df1.reset_index(inplace=True)
    if 'time' not in vix_df.columns and vix_df.index.name == 'time':
        vix_df.reset_index(inplace=True)

    if 'time' not in df1.columns or 'time' not in vix_df.columns:
        raise KeyError("'time' column must exist in both dataframes.")

    # Ensure datetime
    df1['time'] = pd.to_datetime(df1['time'])
    vix_df['time'] = pd.to_datetime(vix_df['time'])

    # Rename VIX columns
    vix_df = vix_df.rename(columns={'open': 'vix_open', 'close': 'vix_close'})

    # Exact match merge
    merged = pd.merge(df1, vix_df[['time', 'vix_open', 'vix_close']], on='time', how='left')

    # Restore time index if it was originally set
    merged.set_index('time', inplace=True)

    return merged




def getDailyObjects(df):

    # Create daily objects
    daily_objects = []

    for date, group in df.groupby('date'):
        day_high = group['high'].max()
        day_low = group['low'].min()
        
        opening_range = group.iloc[:num_candles_or]
        or_high = opening_range['high'].max()
        or_low = opening_range['low'].min()
        
        summary = DaySummary(
            date=date,
            day_high=day_high,
            day_low=day_low,
            or_high=or_high,
            or_low=or_low,
            day_data=group
        )
        
        daily_objects.append(summary)
        
    # populate prev_day_high and prev_day_low
    for i in range(1, len(daily_objects)):
        daily_objects[i].prev_day_high = daily_objects[i - 1].day_high
        daily_objects[i].prev_day_low = daily_objects[i - 1].day_low
        
    # Add isStrongBullish, isStrongBearish, isIncreasingOutlier columns
    # Add isBullish, isBearish, isIncreasingVolume, isDecreasingVolume columns
    for day in daily_objects:
        
        first_row = day.day_data.iloc[0]
        day.prev_VAH = first_row['prev_VA_high']
        day.prev_VAL = first_row['prev_VA_low']
        
        df_day = day.day_data.copy()  # work on a copy
        
        is_strong_bullish_col = []
        is_strong_bearish_col = []
        is_outlier_volume_col = []
        
        is_bullish_col = []
        is_bearish_col = []
        has_long_upper_wick_col = []
        has_long_lower_wick_col = []
        is_increasing_volume_col = []
        is_decreasing_volume_col = []
        
        for i in range(len(df_day)):
            row = df_day.iloc[i]
            
            # IsStrongBullish and IsStrongBearish
            is_strong_bullish = isStrongBullish(row,wick_to_body)
            is_strong_bearish = isStrongBearish(row, wick_to_body)
            
            is_strong_bullish_col.append(is_strong_bullish)
            is_strong_bearish_col.append(is_strong_bearish)
            
            # IsBullish and IsBearish
            is_bullish = isBullish(row)
            is_bearish = isBearish(row)
            has_LongUpperWick = hasLongUpperWick(row)
            has_LongLowerWick = hasLongLowerWick(row)
            
            is_bullish_col.append(is_bullish)
            is_bearish_col.append(is_bearish)
            has_long_upper_wick_col.append(has_LongUpperWick)
            has_long_lower_wick_col.append(has_LongLowerWick)
            
            # IsIncreasingOutlierVolume, isIncreasingVolume, isDecreasingVolume
            if i == 0:
                is_outlier = False
                is_inc_vol = False
                is_dec_vol = False
            else:
                prev_row = df_day.iloc[i - 1]
                is_outlier = isIncreasingOutlierVolume(prev_row, row,percentIncrease)
                is_inc_vol = isIncreasingVolume(prev_row, row,percentIncrease)
                is_dec_vol = isDecreasingVolume(prev_row, row,percentDecrease)
            
            is_outlier_volume_col.append(is_outlier)
            is_increasing_volume_col.append(is_inc_vol)
            is_decreasing_volume_col.append(is_dec_vol)

        # Add all new columns
        df_day['IsStrongBullish'] = is_strong_bullish_col
        df_day['IsStrongBearish'] = is_strong_bearish_col
        df_day['IsIncreasingOutlierVolume'] = is_outlier_volume_col
        df_day['IsBullish'] = is_bullish_col
        df_day['IsBearish'] = is_bearish_col
        df_day['IsIncreasingVolume'] = is_increasing_volume_col
        df_day['IsDecreasingVolume'] = is_decreasing_volume_col
        df_day['HasLongUpperWick'] = has_long_upper_wick_col
        df_day['HasLongLowerWick'] = has_long_lower_wick_col

        # Update the object
        day.day_data = df_day
   
    
    return daily_objects


def attach_vix_to_daily_objects(daily_objects, vix_df):

    # Ensure 'Date' is datetime.date and build lookup dict
    vix_df = vix_df.copy()
    vix_df['time'] = pd.to_datetime(vix_df['time']).dt.date
    vix_dict = vix_df.set_index('time')['open'].to_dict()

    # Attach VIX open values
    for day in daily_objects:
        day.vix_open = vix_dict.get(day.date, None)

    return daily_objects


def move_columns_to_end(df, move_columns):
    # Ensure only columns that exist in df are included
    move_columns = [col for col in move_columns if col in df.columns]
    
    # Get the list of remaining columns
    other_columns = [col for col in df.columns if col not in move_columns]
    
    # Reorder DataFrame
    df = df[other_columns + move_columns]
    return df
