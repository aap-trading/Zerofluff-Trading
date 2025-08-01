import pandas as pd
from patterns import *

def buyORB(OR_high, candle, prev_candle, wick_to_body, percentIncrease):
    return (
        #candle['low'] < OR_high and
        candle['close'] > OR_high and
        isStrongBullish(candle, wick_to_body) and
        isIncreasingOutlierVolume(prev_candle, candle, percentIncrease)
    )

def sellORB(OR_low, candle, prev_candle, wick_to_body, percentIncrease):
    return (
        #candle['high'] > OR_low and
        candle['close'] < OR_low and
        isStrongBearish(candle, wick_to_body) and
        isIncreasingOutlierVolume(prev_candle, candle, percentIncrease)
    )

#V1 - checks if candle is bullish
def buyORBfirst(OR_high, candle, prev_candle, df, wick_to_body, vol_inc=1.2):
    current_time = candle.name

    # Filter same-day candles before current time
    same_day = df[df.index.date == current_time.date()]
    earlier_candles = same_day[same_day.index < current_time].copy()

    # Shift volume for comparison (ensure alignment)
    earlier_candles['PrevVolume'] = earlier_candles['Volume'].shift(1)

    # Check if any earlier candle met all 3 conditions
    for idx, row in earlier_candles.iterrows():
        if (
            row['close'] > OR_high and
            row['close'] > row['open'] and
            pd.notna(row['PrevVolume']) and
            row['Volume'] > row['PrevVolume'] * vol_inc
        ):
            return False  # Earlier valid breakout found

    # Check if current candle meets all conditions
    return (
        candle['close'] > OR_high and
        candle['close'] > candle['open'] and
        candle['Volume'] > prev_candle['Volume'] * vol_inc
    )

#V2 - checks if candle is strong bullish
def buyORBfirstV2(OR_high, candle, prev_candle, df, wick_to_body, vol_inc=1.2):
    current_time = candle.name

    # Filter same-day candles before current time
    same_day = df[df.index.date == current_time.date()]
    earlier_candles = same_day[same_day.index < current_time].copy()

    # Shift volume for comparison (ensure alignment)
    earlier_candles['PrevVolume'] = earlier_candles['Volume'].shift(1)

    # Check if any earlier candle met all 3 conditions
    for idx, row in earlier_candles.iterrows():
        if (
            row['close'] > OR_high and
            isStrongBullish(row, wick_to_body) and
            pd.notna(row['PrevVolume']) and
            row['Volume'] > row['PrevVolume'] * vol_inc
        ):
            return False  # Earlier valid breakout found

    # Check if current candle meets all conditions
    return (
        candle['close'] > OR_high and
        isStrongBullish(candle, wick_to_body) and
        candle['Volume'] > prev_candle['Volume'] * vol_inc
    )

#checks if candle is strong bullish and has increasing outlier volume
def buyORBfirst_strict(OR_high, candle, prev_candle, df, wick_to_body, percentIncrease=1.2):
    current_time = candle.name

    # Filter same-day candles before current time
    same_day = df[df.index.date == current_time.date()]
    earlier_candles = same_day[same_day.index < current_time]

    # Loop through earlier candles and check if any prior candle met the entry conditions
    for i in range(1, len(earlier_candles)):
        curr = earlier_candles.iloc[i]
        prev = earlier_candles.iloc[i - 1]

        if (
            curr['close'] > OR_high and
            isStrongBullish(curr, wick_to_body) and
            isIncreasingOutlierVolume(prev, curr, percentIncrease)
        ):
            return False  # Found earlier valid breakout

    # Check current candle
    return (
        candle['close'] > OR_high and
        isStrongBullish(candle, wick_to_body) and
        isIncreasingOutlierVolume(prev_candle, candle, percentIncrease)
    )




def sellORBfirst(OR_low, candle, prev_candle, df, wick_to_body, vol_inc=1.2):
    current_time = candle.name

    # Filter same-day candles before current time
    same_day = df[df.index.date == current_time.date()]
    earlier_candles = same_day[same_day.index < current_time].copy()

    # Compute volume of previous candle for each row
    earlier_candles['PrevVolume'] = earlier_candles['Volume'].shift(1)

    # Check if any earlier candle already met all 3 conditions
    for idx, row in earlier_candles.iterrows():
        if (
            row['close'] < OR_low and
            row['close'] < row['open'] and
            pd.notna(row['PrevVolume']) and
            row['Volume'] > row['PrevVolume'] * vol_inc
        ):
            return False  # Already had a valid sell breakout

    # Check if current candle meets all conditions
    return (
        candle['close'] < OR_low and
        candle['close'] < candle['open'] and
        candle['Volume'] > prev_candle['Volume'] * vol_inc
    )


def sellORBfirst_strict(OR_low, candle, prev_candle, df, wick_to_body, percentIncrease=1.2):
    current_time = candle.name

    # Filter same-day candles before current time
    same_day = df[df.index.date == current_time.date()]
    earlier_candles = same_day[same_day.index < current_time]

    # Check if any earlier candle already met all conditions
    for i in range(1, len(earlier_candles)):
        curr = earlier_candles.iloc[i]
        prev = earlier_candles.iloc[i - 1]

        if (
            curr['close'] < OR_low and
            isStrongBearish(curr, wick_to_body) and
            isIncreasingOutlierVolume(prev, curr, percentIncrease)
        ):
            return False  # Prior valid breakout detected

    # Check if current candle meets all conditions
    return (
        candle['close'] < OR_low and
        isStrongBearish(candle, wick_to_body) and
        isIncreasingOutlierVolume(prev_candle, candle, percentIncrease)
    )





def buyPrevDayLowR(prev_day_low, candle, prev_candle, wick_to_body, percentIncrease, outlierVol):
    volumeCondition = False
    if outlierVol:
       volumeCondition = isIncreasingOutlierVolume(prev_candle, candle, percentIncrease)
    else:
       volumeCondition = isIncreasingVolume(prev_candle, candle, percentIncrease)
    
    return (
        candle['low'] < prev_day_low and
        isStrongBullish(candle, wick_to_body) and
        volumeCondition
    )

def sellPrevDayHighR(prev_day_high, candle, prev_candle, wick_to_body, percentIncrease, outlierVol):
    volumeCondition = False
    if outlierVol:
       volumeCondition = isIncreasingOutlierVolume(prev_candle, candle, percentIncrease)
    else:
       volumeCondition = isIncreasingVolume(prev_candle, candle, percentIncrease)
    
    return (
        candle['high'] > prev_day_high and
        isStrongBearish(candle, wick_to_body) and
        volumeCondition
    )


def buyORBR(OR_low, candle, prev_candle, wick_to_body, percentIncrease, outlierVol):
    volumeCondition = False
    if outlierVol:
       volumeCondition = isIncreasingOutlierVolume(prev_candle, candle, percentIncrease)
    else:
       volumeCondition = isIncreasingVolume(prev_candle, candle, percentIncrease)
    
    return (
        candle['low'] < OR_low and
        #candle['close'] > OR_low and
        isStrongBullish(candle, wick_to_body) and
        volumeCondition
    )

def sellORBR(OR_high, candle, prev_candle, wick_to_body, percentIncrease, outlierVol):
    volumeCondition = False
    if outlierVol:
       volumeCondition = isIncreasingOutlierVolume(prev_candle, candle, percentIncrease)
    else:
       volumeCondition = isIncreasingVolume(prev_candle, candle, percentIncrease)
    
    return (
        candle['high'] > OR_high and
        #candle['close'] < OR_high and
        isStrongBearish(candle, wick_to_body) and
        volumeCondition
    )

def buyBBR(candle, prev_candle, wick_to_body, percentIncrease):
    return (
        candle['low'] < candle['Lower'] and
        isStrongBullish(candle, wick_to_body) and
        isIncreasingOutlierVolume(prev_candle, candle, percentIncrease)
    )

def sellBBR(candle, prev_candle, wick_to_body, percentIncrease):
    return (
        candle['high'] > candle['Upper'] and
        isStrongBearish(candle, wick_to_body) and
        isIncreasingOutlierVolume(prev_candle, candle, percentIncrease)
    )

def buyStrongVolumeAndCandle(candle, prev_candle, wick_to_body, percentIncrease, outlierVol):
    volumeCondition = False
    if outlierVol:
       volumeCondition = isIncreasingOutlierVolume(prev_candle, candle, percentIncrease)
    else:
       volumeCondition = isIncreasingVolume(prev_candle, candle, percentIncrease)
    
    return (
        isStrongBullish(candle, wick_to_body) and
        volumeCondition
    )

def sellStrongVolumeAndCandle(candle, prev_candle, wick_to_body, percentIncrease,outlierVol):
    volumeCondition = False
    if outlierVol:
       volumeCondition = isIncreasingOutlierVolume(prev_candle, candle, percentIncrease)
    else:
       volumeCondition = isIncreasingVolume(prev_candle, candle, percentIncrease)
    
    return (
        isStrongBearish(candle, wick_to_body) and
        volumeCondition
    )

def buyStrongCandle(candle, wick_to_body):
    return (
        isStrongBullish(candle, wick_to_body)      
    )

def sellStrongCandle(candle, wick_to_body):
    return (
        isStrongBearish(candle, wick_to_body)        
    )

def buyStrongCandleVix(candle, wick_to_body,vix_th):
    return (
        isStrongBullish(candle, wick_to_body) and
        candle['vix_close'] < vix_th
    )

def sellStrongCandleVix(candle, wick_to_body, vix_th):
    return (
        isStrongBearish(candle, wick_to_body)  and  
        candle['vix_close'] < vix_th
    )

def buyStrongFirstCandle(candle, wick_to_body, i):
    return (
        isStrongBullish(candle, wick_to_body) and i == 0     
    )

def sellStrongFirstCandle(candle, wick_to_body, i):
    return (
        isStrongBearish(candle, wick_to_body) and i == 0    
    )

def buyStrongVolumeAndCandleR(candle, prev_candle, wick_to_body, percentIncrease):
    return (
        candle['close'] > prev_candle['high'] and
        isBearish(prev_candle) and
        isStrongBullish(candle, wick_to_body) and
        isIncreasingVolume(prev_candle, candle, percentIncrease)
    )

def sellStrongVolumeAndCandleR(candle, prev_candle, wick_to_body, percentIncrease):
    return (
        candle['close'] < prev_candle['low'] and
        isBullish(prev_candle) and
        isStrongBearish(candle, wick_to_body) and
        isIncreasingVolume(prev_candle, candle, percentIncrease)
    )

def buyWeakVolumePullback(candle, prev_candle, wick_to_body, percentDecrease):
    return (
        isStrongBullish(prev_candle,wick_to_body) and
        candle['low'] > prev_candle['low'] and
        isBearish(candle) and
        isDecreasingVolume(prev_candle, candle, percentDecrease)
    )

def sellWeakVolumePullback(candle, prev_candle, wick_to_body, percentDecrease):
    return (
        isStrongBearish(prev_candle, wick_to_body) and
        candle['high'] < prev_candle['high'] and
        isBullish(candle) and
        isDecreasingVolume(prev_candle, candle, percentDecrease)
    )

