import pandas as pd
import matplotlib.pyplot as plt

from datapreparation import *
from patterns import *
from objects import DaySummary
from setups import *
from execution import *
from analysis import *
from visualise import *
from enrichments import *
from configuration import *

# Upload price action and VIX data
pa_df = import_df(file_path)
vix_5 = import_df(vix5_file_path)
vix_d = pd.read_csv(vixd_file_path)

# Pre-process data
pa_df = add_vix_to_df(pa_df, vix_5)
pa_df = add_prev_value_area(pa_df)
daily_objects = getDailyObjects(pa_df)
daily_objects = attach_vix_to_daily_objects(daily_objects, vix_d)

# Test strategy
trades = testStrategy(daily_objects, buy=True, sell=False, buyStrategy='buyORBfirst', sellStrategy=None)

# Export trades
trades.to_csv('out/unfiltered_trades.csv', index=False)

# Analyse trades
analyseTrades(trades, vix_d)

# Visualise candles and signals - opens 1 browser tab per day!
# for day in daily_objects[1:]: 
#     df_day = day.day_data
#     visualiseCandleSticks(df_day,trades)    


