import pandas as pd
import numpy as np

from visualise import *
from configuration import *



def filter_trades_by_vix_threshold(executed_trades, vix_df, vix_threshold):

    # Ensure date columns are datetime
    executed_trades = executed_trades.copy()
    executed_trades['Trade Date'] = pd.to_datetime(executed_trades['Entry Candle']).dt.date
    vix_df = vix_df.copy()
    vix_df['time'] = pd.to_datetime(vix_df['time']).dt.date

    # Filter VIX to only dates with VIX open > threshold
    high_vix_dates = set(vix_df[vix_df['open'] > vix_threshold]['time'])

    # Remove trades that happened on those dates
    filtered_trades = executed_trades[~executed_trades['Trade Date'].isin(high_vix_dates)]

    return filtered_trades.drop(columns=['Trade Date'])


def remove_overlapping_trades(trades_df):
    # Ensure datetime columns
    trades_df = trades_df.copy()
    # Filter out rows where 'Entry Candle' is already not a valid string/datetime
    trades_df = trades_df[trades_df['Entry Candle'].apply(lambda x: isinstance(x, (str, pd.Timestamp)))]

    trades_df['Entry Candle'] = pd.to_datetime(trades_df['Entry Candle'])
    trades_df['Exit Candle'] = pd.to_datetime(trades_df['Exit Candle'])

    # Sort by Entry Candle to ensure order
    trades_df.sort_values(by='Entry Candle', inplace=True)

    # Add date column for grouping
    trades_df['Trade Date'] = trades_df['Entry Candle'].dt.date

    filtered_trades = []

    for date, group in trades_df.groupby('Trade Date'):
        last_exit = None
        for _, row in group.iterrows():
            if last_exit is None or row['Entry Candle'] >= last_exit:
                filtered_trades.append(row)
                last_exit = row['Exit Candle']
            else:
                # Overlapping trade — skip
                continue

    df = pd.DataFrame(filtered_trades)
    if 'Trade Date' in df.columns:
        df = df.drop(columns=['Trade Date'])
    return df

def summarize_trades(trades_df, multiplier):
    trades_df = trades_df.copy()
    trades_df['Signal Candle'] = pd.to_datetime(trades_df['Signal Candle'])
    trades_df = trades_df.dropna(subset=['Price Difference'])

    # Apply multiplier to PnL
    trades_df['Price Difference'] = trades_df['Price Difference'] * multiplier

    # Basic stats
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['Price Difference'] > 0]
    losing_trades = trades_df[trades_df['Price Difference'] < 0]

    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    avg_pnl = trades_df['Price Difference'].mean()
    avg_win = winning_trades['Price Difference'].mean() if not winning_trades.empty else 0
    avg_loss = losing_trades['Price Difference'].mean() if not losing_trades.empty else 0
    net_pnl = trades_df['Price Difference'].sum()

    summary = {
        'Total Trades': total_trades,
        'Winning Trades': len(winning_trades),
        'Losing Trades': len(losing_trades),
        'Win Rate (%)': round(win_rate * 100, 2),
        'Average PnL': round(avg_pnl, 2),
        'Average Win': round(avg_win, 2),
        'Average Loss': round(avg_loss, 2),
        'Net PnL': round(net_pnl, 2),
    }

    # Signal Type Breakdown
    for signal_type in trades_df['Signal Type'].unique():
        sub_df = trades_df[trades_df['Signal Type'] == signal_type]
        summary[f'{signal_type} Count'] = len(sub_df)
        summary[f'{signal_type} Avg PnL'] = round(sub_df['Price Difference'].mean(), 2)

    # Hourly Breakdown
    trades_df['Hour'] = trades_df['Signal Candle'].dt.hour
    hourly_stats = trades_df.groupby('Hour')['Price Difference'].agg(
        Trades='count',
        AvgPnL='mean',
        WinRate=lambda x: (x > 0).sum() / len(x) * 100,
        TotalPnL='sum'
    ).round(2)

    # Weekly Stats
    trades_df['Week'] = trades_df['Signal Candle'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_stats = trades_df.groupby('Week')['Price Difference'].agg(
        Trades='count',
        AvgPnL='mean',
        WinRate=lambda x: (x > 0).sum() / len(x) * 100,
        TotalPnL='sum'
    ).round(2)
    weekly_stats['CumulativePnL'] = weekly_stats['TotalPnL'].cumsum().round(2)

    # Daily Stats
    trades_df['Date'] = trades_df['Signal Candle'].dt.date
    daily_stats = trades_df.groupby('Date')['Price Difference'].agg(
        Trades='count',
        AvgPnL='mean',
        WinRate=lambda x: (x > 0).sum() / len(x) * 100,
        TotalPnL='sum'
    ).round(2)
    daily_stats['CumulativePnL'] = daily_stats['TotalPnL'].cumsum().round(2)

    # Monthly Stats with negative days and worst day per month
    trades_df['Month'] = trades_df['Signal Candle'].dt.to_period('M').apply(lambda r: r.start_time)
    monthly_stats = trades_df.groupby('Month')['Price Difference'].agg(
        Trades='count',
        AvgPnL='mean',
        WinRate=lambda x: (x > 0).sum() / len(x) * 100,
        TotalPnL='sum'
    ).round(2)

    # Add # of negative days and worst day PnL for each month
    trades_df['Date'] = pd.to_datetime(trades_df['Signal Candle'].dt.date)
    daily_pnl_by_month = trades_df.groupby(['Month', 'Date'])['Price Difference'].sum().reset_index()
    
    negative_days = daily_pnl_by_month[daily_pnl_by_month['Price Difference'] < 0]
    neg_day_counts = negative_days.groupby('Month')['Date'].count()
    worst_days = daily_pnl_by_month.groupby('Month').apply(
        lambda df: df.loc[df['Price Difference'].idxmin()]
    ).reset_index(drop=True)

    monthly_stats['NegativeDays'] = monthly_stats.index.map(neg_day_counts).fillna(0).astype(int)
    monthly_stats['WorstDayPnL'] = monthly_stats.index.map(
        worst_days.set_index('Month')['Price Difference']
    ).round(2)

    # Best and Worst Days
    best_day = daily_stats['TotalPnL'].idxmax() if not daily_stats.empty else None
    worst_day = daily_stats['TotalPnL'].idxmin() if not daily_stats.empty else None
    summary['Best Day'] = str(best_day)
    summary['Best Day PnL'] = round(daily_stats.loc[best_day, 'TotalPnL'], 2) if best_day else 0
    summary['Worst Day'] = str(worst_day)
    summary['Worst Day PnL'] = round(daily_stats.loc[worst_day, 'TotalPnL'], 2) if worst_day else 0

    # Worst Week
    worst_week = weekly_stats['TotalPnL'].idxmin() if not weekly_stats.empty else None
    summary['Worst Week'] = str(worst_week)
    summary['Worst Week PnL'] = round(weekly_stats.loc[worst_week, 'TotalPnL'], 2) if worst_week else 0

    # Worst Month
    worst_month = monthly_stats['TotalPnL'].idxmin() if not monthly_stats.empty else None
    summary['Worst Month'] = str(worst_month)
    summary['Worst Month PnL'] = round(monthly_stats.loc[worst_month, 'TotalPnL'], 2) if worst_month else 0

    return summary, hourly_stats, daily_stats, weekly_stats, monthly_stats


def data_summary_report(df):
    for col in df.columns:
        print(f"Column: {col}")
        series = df[col].dropna()  # exclude NaNs for stats
        
        if pd.api.types.is_bool_dtype(series):
            # Boolean: % True
            true_pct = series.mean() * 100
            print(f"  Type: Boolean")
            print(f"  % True: {true_pct:.2f}%")
        
        elif pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
            # Categorical: % distribution
            print(f"  Type: Categorical")
            counts = series.value_counts(normalize=True) * 100
            for cat_val, pct in counts.items():
                print(f"    {cat_val}: {pct:.2f}%")
        
        elif pd.api.types.is_numeric_dtype(series):
            # Numeric: histogram with 7 bins
            print(f"  Type: Numerical")
            counts, bin_edges = np.histogram(series, bins=7)
            for i in range(len(counts)):
                print(f"    Bin {i+1} [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}): {counts[i]}")
        
        else:
            print(f"  Type: Other/Unknown - skipped")
        
        print("-" * 40)
        

def data_summary_comparison_report(df1, df2, name1, name2):
    assert list(df1.columns) == list(df2.columns), "DataFrames must have same columns"

    for col in df1.columns:
        print(f"\nColumn: {col}")
        s1 = df1[col].dropna()
        s2 = df2[col].dropna()

        if pd.api.types.is_bool_dtype(s1):
            print("  Type: Boolean")
            pct1 = s1.mean() * 100
            pct2 = s2.mean() * 100
            print(f"    % True in {name1}: {pct1:.2f}%")
            print(f"    % True in {name2}: {pct2:.2f}%")

        elif pd.api.types.is_categorical_dtype(s1) or pd.api.types.is_object_dtype(s1):
            print("  Type: Categorical")
            vc1 = s1.value_counts(normalize=True) * 100
            vc2 = s2.value_counts(normalize=True) * 100
            all_categories = set(vc1.index).union(set(vc2.index))
            for cat in sorted(all_categories):
                p1 = vc1.get(cat, 0.0)
                p2 = vc2.get(cat, 0.0)
                print(f"    {cat}: {name1}: {p1:.2f}%, {name2}: {p2:.2f}%")

        elif pd.api.types.is_numeric_dtype(s1):
            print("  Type: Numerical")
            combined = pd.concat([s1, s2])
            bins = np.histogram_bin_edges(combined, bins=3)
            h1, _ = np.histogram(s1, bins=bins)
            h2, _ = np.histogram(s2, bins=bins)
            for i in range(len(h1)):
                print(f"    Bin {i+1} [{bins[i]:.2f}, {bins[i+1]:.2f}): {name1}: {h1[i]}, {name2}: {h2[i]}")

        else:
            print("  Type: Other/Unknown - skipped")

        print("-" * 60)


def filter_first_trade_per_day(df):
    # Ensure Signal Candle is datetime
    df['Signal Candle'] = pd.to_datetime(df['Signal Candle'])

    # Sort to ensure the first trade of the day is actually first
    df_sorted = df.sort_values('Signal Candle')

    # Group by date and take the first trade per day
    first_trades = df_sorted.groupby(df_sorted['Signal Candle'].dt.date, as_index=False).first()

    return first_trades



def filter_trades_by_entry_time(df, start_time_str, end_time_str):
    """
    Filters trades to include only those with Entry Candle time between start_time and end_time (inclusive).
    
    """
    # Ensure Entry Candle is datetime
    df = df.copy()
    df['Entry Candle'] = pd.to_datetime(df['Entry Candle'])

    # Extract just the time part
    entry_times = df['Entry Candle'].dt.time

    # Convert strings to time objects
    start_time = pd.to_datetime(start_time_str).time()
    end_time = pd.to_datetime(end_time_str).time()

    # Apply filter
    mask = (entry_times >= start_time) & (entry_times <= end_time)
    return df[mask].reset_index(drop=True)


def calculate_pnl_targets(df, profit_target=3000, loss_limit=-2000):
    """
    For each month's first trading day, calculate:
    - Days to reach cumulative profit_target (across months if needed)
    - Days to reach cumulative loss_limit (if not preceded by hitting profit target)
    - If profit is hit first, 'Days to -2000' is 0

    Returns a DataFrame with:
    - 'Month Start'
    - 'Days to +3000'
    - 'Days to -2000'
    """
    df = df.copy()

    # Ensure proper date formatting
    if 'Date' not in df.columns:
        if df.index.name == 'Date':
            df.reset_index(inplace=True)
        else:
            raise KeyError("'Date' column not found in DataFrame.")

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Identify first trading day of each month
    df['Month'] = df['Date'].dt.to_period('M')
    first_day_indices = df.groupby('Month').head(1).index

    results = []

    for idx in first_day_indices:
        start_row = df.iloc[idx]
        start_date = start_row['Date']
        cumulative = 0
        days_to_profit = 0
        days_to_loss = 0
        profit_hit_index = None
        loss_hit_index = None

        for offset, row in enumerate(df.iloc[idx:].itertuples(index=False), start=1):
            cumulative += row.TotalPnL

            if profit_hit_index is None and cumulative >= profit_target:
                days_to_profit = offset
                profit_hit_index = offset

            if loss_hit_index is None and cumulative <= loss_limit:
                days_to_loss = offset
                loss_hit_index = offset

            if profit_hit_index is not None and loss_hit_index is not None:
                break

        # Override loss if profit came first
        if profit_hit_index is not None and (loss_hit_index is None or profit_hit_index < loss_hit_index):
            days_to_loss = 0

        results.append({
            'Month Start': start_date.date(),
            'Days to +3000': days_to_profit,
            'Days to -2000': days_to_loss
        })

    return pd.DataFrame(results)


def analyseTrades(trades, vix_d):
    if  trades.empty:
        print('No trades')
    else:
        filtered_trades = trades[trades['Signal Type'].isin(['Buy', 'Sell'])] #executed trades
        filtered_trades = remove_overlapping_trades(filtered_trades) #non-overlapping trades
        filtered_trades = filter_trades_by_vix_threshold(filtered_trades, vix_d, vix_threshold) #filter by vix threshold
        filtered_trades = filter_first_trade_per_day(filtered_trades) #only consider first trade each day
        filtered_trades = filter_trades_by_entry_time(filtered_trades, "07:35:00", "09:30:00") #filter trades by start time
        
        
        if  filtered_trades.empty:
            print('No trades')
        else:
            summary, hourly_stats, daily_stats, weekly_stats, monthly_stats = summarize_trades(filtered_trades,multiplier)
            
            filtered_trades.to_csv('out/trades_summary.csv', index=False)
            
            print("=== Summary ===")
            df_summary = pd.DataFrame.from_dict(summary, orient='index', columns=['Value'])
            df_summary.to_csv('out/summary.csv', index=True)
            for k, v in summary.items():
                print(f"{k}: {v}")
            
            print("\n=== Hourly Breakdown ===")
            print(hourly_stats)
            
            print("\n=== Monthly Breakdown ===")
            print(monthly_stats)
            monthly_stats.to_csv('out/monthly_stats.csv', index=True)
            
            #print("\n=== Weekly Breakdown ===")
            #print(weekly_stats)
            weekly_stats.to_csv('out/weekly_stats.csv', index=True)
            
            # print("\n=== Daily Breakdown ===")
            # print(daily_stats)
            daily_stats.to_csv('out/daily_stats.csv', index=True)
            plot_cumulative_pnl(daily_stats)
            
            day_to_target_df = calculate_pnl_targets(daily_stats, profit_target=3000, loss_limit=-2000)
            day_to_target_df.to_csv('out/day_to_target.csv', index=False)




