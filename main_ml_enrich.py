import pandas as pd

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


# Enrich price action data and export
enriched_candles = []

for day in daily_objects[1:]: 
    df_day = day.day_data
    enriched_candles.append(df_day)
    
enriched_candles = [df.reset_index() for df in enriched_candles]
combined_df = pd.concat(enriched_candles, ignore_index=False)

combined_df = add_features(combined_df, daily_objects)
combined_df = add_signals(combined_df, daily_objects)
combined_df = add_executions(combined_df, daily_objects) 
combined_df = add_pullbacks(combined_df, daily_objects)

					 	 	 	
remove_columns = ["time","open","high","low","close","BBBasis","BBUpper35","BBLower35","BBUpper3","BBLower3","BBUpper25",
                  "BBLower25","BBUpper2","BBLower2","MA50","MA20","Volume MA","VWAP","RSI MA","MACD Histogram","MACD","MACD Signal",
                  "date","vix_open","prev_VA_high","prev_VA_low","PrevDayHigh","PrevDayLow","NumUpperBreached","NumLowerBreached"]

df_filtered = combined_df.drop(columns=remove_columns, errors='ignore')
df_filtered.to_csv('ml/in/enriched_candles.csv', index=False)






    


