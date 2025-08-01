from configuration import *

def isIncreasingVolume(df1, df2, percentIncrease):
    vol1 = df1['Volume']
    vol2 = df2['Volume']
    
    if vol1 == 0:
        return False  # Avoid division by zero
    
    increase = (vol2 - vol1) / vol1
    return increase >= percentIncrease

def isDecreasingVolume(df1, df2, percentDecrease):
    vol1 = df1['Volume']
    vol2 = df2['Volume']
    
    if vol1 == 0:
        return False  # Avoid division by zero
    
    decrease = (vol1 - vol2) / vol1
    return decrease >= percentDecrease

def isIncreasingOutlierVolume(df1, df2, percentIncrease):
    vol2 = df2['Volume']
    vol_ma = df2['Volume MA']
    
    return vol2 > vol_ma and isIncreasingVolume(df1, df2, percentIncrease)

def isBullish(row):
    return row['close'] > row['open']

def isBearish(row):
    return row['close'] < row['open']

def isStrongBullish(row, wick_to_body):
    open_price = row['open']
    close_price = row['close']
    high = row['high']
    low = row['low']
    body = close_price - open_price
    upper_wick = high - close_price
    lower_wick = open_price - low
    body_and_upper_wick = high - open_price
    candle_range = high - low
    
    if close_price <= open_price:
        return False  # Not bullish
    
    if candle_range < min_points_strong_candle:
        return False  # Not strong
    
    # Avoid division by zero
    if body_and_upper_wick == 0:
        return False
    
    if lower_wick / body_and_upper_wick > long_wick_ratio:
        return True
    
    # Avoid division by zero
    if body == 0:
        return False

    # Check if upper wick is small relative to body
    return (upper_wick / body < wick_to_body) 

def isStrongBearish(row, wick_to_body):
    open_price = row['open']
    close_price = row['close']
    high = row['high']
    low = row['low']
    body = open_price - close_price
    upper_wick = high - open_price
    lower_wick = close_price - low
    body_and_lower_wick = open_price - low
    candle_range = high - low

    if close_price >= open_price:
        return False  # Not bearish

    if candle_range < min_points_strong_candle:
        return False  # Not strong

    # Avoid division by zero
    if body_and_lower_wick == 0:
        return False

    if upper_wick / body_and_lower_wick > long_wick_ratio:
        return True

    # Avoid division by zero
    if body == 0:
        return False

    # Check if lower wick is small relative to body
    return (lower_wick / body < wick_to_body)


def hasLongLowerWick(row):
    """
    Returns True if the lower wick is longer than the rest of the candle 
    (body + upper wick), regardless of bullish/bearish candle.
    
    """
    open_price = row['open']
    close_price = row['close']
    high = row['high']
    low = row['low']

    body_top = max(open_price, close_price)
    body_bottom = min(open_price, close_price)

    lower_wick = body_bottom - low
    upper_wick = high - body_top
    body = abs(close_price - open_price)
    
    rest_of_candle = body + upper_wick

    if rest_of_candle == 0:
        return False  # Avoid division by zero

    return (lower_wick / rest_of_candle) > long_wick_ratio


def hasLongUpperWick(row):
    """
    Returns True if the upper wick is longer than the rest of the candle 
    (body + lower wick), regardless of bullish/bearish candle.
    """
    open_price = row['open']
    close_price = row['close']
    high = row['high']
    low = row['low']

    body_top = max(open_price, close_price)
    body_bottom = min(open_price, close_price)

    upper_wick = high - body_top
    lower_wick = body_bottom - low
    body = abs(close_price - open_price)
    
    rest_of_candle = body + lower_wick

    if rest_of_candle == 0:
        return False  # Avoid division by zero

    return (upper_wick / rest_of_candle) > long_wick_ratio



    


