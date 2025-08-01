
# VIX data
vix5_file_path = 'in/VIX5-2021-2025.csv'
vixd_file_path = 'in/VIX_D.csv'

# Price action data
file_path = 'in/NQ-5m-2023-2025-indicators.csv'

# strong candle settings
wick_to_body = 0.9 # shorter wick to body ratio for strong candle def
long_wick_ratio = 1.1 # longer wick to (body+short wick) ratio for strong candle def
min_points_strong_candle = 20 # strong candle range (high to low) - min points 

# volume settings
percentIncrease = 1.2 # 20% increase for increasing volume 
percentDecrease = 1.2 # 20% decrease for decreasing volume

# general settings
trading_hours = {7,8,9,10} # trading_hours 
multiplier = 10 # futures multiplier 
vix_threshold = 30

# setup settings
num_candles_or = 2 # Opening Range - number of candles for the opening range
outlierVol = True 

# execution settings
start_candle = 2 # start candle index
end_candles = 1 # end candles to ignore
num_candles_exit = 30 # if 1, exit at entry candle
tp = 75 # take profit points
sl = 75 # max loss points
sl_extra = 0 # extra points below/above signal candle low/high
entry_pullback_points = 0 # number of pullback points from signal candle close
entry_type_buy = 'next_open' #'next_open', 'pullback_low_sl', 'fixed_pullback'
entry_type_sell = 'next_open' #'next_open', 'pullback_high_sl', 'fixed_pullback'
sl_type_buy = 'fixed' #'fixed', 'signal_candle_low'
sl_type_sell = 'fixed' #'fixed', 'signal_candle_high'
tp_entry = True #True - take profit at entry candle, False - tp after entry candle










