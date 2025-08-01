import pandas as pd
from objects import *
from setups import *
from configuration import *


def executeBuyTrade(df_day, i, candle, trades,entry_type,sl_type,tp,sl,sl_extra,entry_pullback_points,num_candles_exit,tp_entry):
    """
    Configurable Buy trade executor.
    """
    if i + 1 >= len(df_day):
        return trades  # Can't process without next candle

    entry_candle = df_day.iloc[i + 1]

    # Determine entry price
    if entry_type == 'next_open':
        entry_price = entry_candle['open']
    elif entry_type == 'pullback_low_sl':
        entry_price = candle['low'] + sl
        if entry_candle['low'] > entry_price:
            new_trade = pd.DataFrame([{
                'Signal Type': 'BuyNoEntry',
                'Signal Candle': candle.name,
                'Entry Candle': pd.NaT,
                'Exit Candle': pd.NaT,
                'Entry Price': float('nan'),
                'Exit Price': float('nan'),
                'Price Difference': float('nan')
            }])
            trades = pd.concat([trades, new_trade], ignore_index=True)
            return trades
    elif entry_type == 'fixed_pullback':
        entry_price = entry_candle['open'] - entry_pullback_points
        if entry_price < candle['low']:
            entry_price = candle['low']
        if entry_candle['low'] > entry_price:
            new_trade = pd.DataFrame([{
                'Signal Type': 'BuyNoEntry',
                'Signal Candle': candle.name,
                'Entry Candle': pd.NaT,
                'Exit Candle': pd.NaT,
                'Entry Price': float('nan'),
                'Exit Price': float('nan'),
                'Price Difference': float('nan')
            }])
            trades = pd.concat([trades, new_trade], ignore_index=True)
            return trades
    else:
        raise ValueError("Invalid entry_type")

    # Determine stop loss
    if sl_type == 'fixed':
        stop_price = entry_price - sl
    elif sl_type == 'signal_candle_low':
        stop_price = candle['low'] - sl_extra
    else:
        raise ValueError("Invalid sl_type")

    # Determine take profit
    take_profit = entry_price + tp if tp is not None else None

    # Define range for exit loop
    max_exit_index = min(i + num_candles_exit, len(df_day) - 1)
    exit_price = None
    exit_candle_time = None
    pnl = None

    for j in range(i + 1, max_exit_index + 1):
        current_candle = df_day.iloc[j]

        # Always check stop loss from entry candle
        if current_candle['low'] < stop_price:
            exit_price = stop_price
            pnl = round(exit_price - entry_price, 2)
            exit_candle_time = current_candle.name
            break

        # Check take profit condition based on tp_entry flag
        if take_profit is not None:
            if (tp_entry and j == i + 1 and current_candle['high'] >= take_profit) or \
               (j > i + 1 and current_candle['high'] >= take_profit):
                exit_price = take_profit
                pnl = round(take_profit - entry_price, 2)
                exit_candle_time = current_candle.name
                break

    # If no exit triggered, close at last allowed candle
    if pnl is None:
        final_candle = df_day.iloc[max_exit_index]
        exit_price = final_candle['close']
        exit_candle_time = final_candle.name
        pnl = round(exit_price - entry_price, 2)

    trades.loc[len(trades)] = [
        'Buy',
        candle.name,
        entry_candle.name,
        exit_candle_time,
        entry_price,
        exit_price,
        pnl
    ]

    return trades

def executeSellTrade(df_day, i, candle, trades,entry_type,sl_type,tp,sl,sl_extra,entry_pullback_points,num_candles_exit,tp_entry):
    """
    Configurable Sell trade executor.
    """
    if i + 1 >= len(df_day):
        return trades  # Can't process without next candle

    entry_candle = df_day.iloc[i + 1]

    # Determine entry price
    if entry_type == 'next_open':
        entry_price = entry_candle['open']
    elif entry_type == 'pullback_high_sl':
        entry_price = candle['high'] - sl
        if entry_candle['high'] < entry_price:
            new_trade = pd.DataFrame([{
                'Signal Type': 'SellNoEntry',
                'Signal Candle': candle.name,
                'Entry Candle': pd.NaT,
                'Exit Candle': pd.NaT,
                'Entry Price': float('nan'),
                'Exit Price': float('nan'),
                'Price Difference': float('nan')
            }])
            trades = pd.concat([trades, new_trade], ignore_index=True)
            return trades
    elif entry_type == 'fixed_pullback':
        entry_price = entry_candle['open'] + entry_pullback_points
        if entry_price > candle['high']:
            entry_price = candle['high']
        if entry_candle['high'] < entry_price:
            new_trade = pd.DataFrame([{
                'Signal Type': 'SellNoEntry',
                'Signal Candle': candle.name,
                'Entry Candle': pd.NaT,
                'Exit Candle': pd.NaT,
                'Entry Price': float('nan'),
                'Exit Price': float('nan'),
                'Price Difference': float('nan')
            }])
            trades = pd.concat([trades, new_trade], ignore_index=True)
            return trades
    else:
        raise ValueError("Invalid entry_type")

    # Determine stop loss
    if sl_type == 'fixed':
        stop_price = entry_price + sl
    elif sl_type == 'signal_candle_high':
        stop_price = candle['high'] + sl_extra
    else:
        raise ValueError("Invalid sl_type")

    # Determine take profit
    take_profit = entry_price - tp if tp is not None else None

    # Define range for exit loop
    max_exit_index = min(i + num_candles_exit, len(df_day) - 1)
    exit_price = None
    exit_candle_time = None
    pnl = None

    for j in range(i + 1, max_exit_index + 1):
        current_candle = df_day.iloc[j]

        # Always check stop loss starting from entry candle
        if current_candle['high'] > stop_price:
            exit_price = stop_price
            pnl = round(entry_price - exit_price, 2)
            exit_candle_time = current_candle.name
            break

        # Take profit condition, controlled by tp_entry
        if take_profit is not None:
            if (tp_entry and j == i + 1 and current_candle['low'] <= take_profit) or \
               (j > i + 1 and current_candle['low'] <= take_profit):
                exit_price = take_profit
                pnl = round(entry_price - take_profit, 2)
                exit_candle_time = current_candle.name
                break

    # If no exit triggered, close at last allowed candle
    if pnl is None:
        final_candle = df_day.iloc[max_exit_index]
        exit_price = final_candle['close']
        exit_candle_time = final_candle.name
        pnl = round(entry_price - exit_price, 2)

    trades.loc[len(trades)] = [
        'Sell',
        candle.name,
        entry_candle.name,
        exit_candle_time,
        entry_price,
        exit_price,
        pnl
    ]

    return trades


def testStrategy(daily_objects, buy=True, sell=False, buyStrategy=None, sellStrategy=None):
    # Define trades df
    trades = pd.DataFrame({
        'Signal Type': pd.Series(dtype='str'),
        'Signal Candle': pd.Series(dtype='datetime64[ns]'),
        'Entry Candle': pd.Series(dtype='datetime64[ns]'),
        'Exit Candle': pd.Series(dtype='datetime64[ns]'),
        'Entry Price': pd.Series(dtype='float'),
        'Exit Price': pd.Series(dtype='float'),
        'Price Difference': pd.Series(dtype='float')
    })

    for day in daily_objects[1:]:
        df_day = day.day_data
        or_high = day.or_high
        or_low = day.or_low
        prev_day_high = day.prev_day_high
        prev_day_low = day.prev_day_low

        for i in range(start_candle, len(df_day) - end_candles):
            candle = df_day.iloc[i]
            prev_candle = df_day.iloc[i - 1] if i > 0 else None
            candle_hour = candle.name.hour if hasattr(candle.name, 'hour') else pd.to_datetime(candle['time']).hour

            if candle_hour in trading_hours:

                # ---------------- BUY LOGIC ----------------
                if buy and buyStrategy:
                    signal = False

                    if buyStrategy == 'buyBBR':
                        signal = buyBBR(candle, prev_candle, wick_to_body, percentIncrease)

                    elif buyStrategy == 'buyORB':
                        signal = buyORB(or_high, candle, prev_candle, wick_to_body, percentIncrease)

                    elif buyStrategy == 'buyORBfirst':
                        signal = buyORBfirst(or_high, candle, prev_candle, df_day, wick_to_body, percentIncrease)

                    elif buyStrategy == 'buyPrevDayLowR':
                        signal = buyPrevDayLowR(prev_day_low, candle, prev_candle, wick_to_body, percentIncrease, outlierVol)

                    elif buyStrategy == 'buyStrongVolumeAndCandle':
                        signal = buyStrongVolumeAndCandle(candle, prev_candle, wick_to_body, percentIncrease, outlierVol)

                    elif buyStrategy == 'buyStrongVolumeAndCandleR':
                        signal = buyStrongVolumeAndCandleR(candle, prev_candle, wick_to_body, percentIncrease)

                    elif buyStrategy == 'buyWeakVolumePullback':
                        signal = buyWeakVolumePullback(candle, prev_candle, wick_to_body, percentDecrease)

                    elif buyStrategy == 'buyStrongCandle':
                        signal = buyStrongCandle(candle, wick_to_body)

                    elif buyStrategy == 'buyStrongCandleVix':
                        signal = buyStrongCandleVix(candle, wick_to_body, vix_threshold)

                    elif buyStrategy == 'buyStrongFirstCandle':
                        signal = buyStrongFirstCandle(candle, wick_to_body, i)

                    if signal:
                        trades = executeBuyTrade(df_day, i, candle, trades,
                                                 entry_type_buy, sl_type_buy, tp, sl, sl_extra,
                                                 entry_pullback_points, num_candles_exit, tp_entry)

                # ---------------- SELL LOGIC ----------------
                if sell and sellStrategy:
                    signal = False

                    if sellStrategy == 'sellBBR':
                        signal = sellBBR(candle, prev_candle, wick_to_body, percentIncrease)

                    elif sellStrategy == 'sellORB':
                        signal = sellORB(or_low, candle, prev_candle, wick_to_body, percentIncrease)

                    elif sellStrategy == 'sellORBfirst':
                        signal = sellORBfirst(or_low, candle, prev_candle, df_day, wick_to_body, percentIncrease)

                    elif sellStrategy == 'sellPrevDayHighR':
                        signal = sellPrevDayHighR(prev_day_high, candle, prev_candle, wick_to_body, percentIncrease, outlierVol)

                    elif sellStrategy == 'sellStrongVolumeAndCandle':
                        signal = sellStrongVolumeAndCandle(candle, prev_candle, wick_to_body, percentIncrease, outlierVol)

                    elif sellStrategy == 'sellStrongVolumeAndCandleR':
                        signal = sellStrongVolumeAndCandleR(candle, prev_candle, wick_to_body, percentIncrease)

                    elif sellStrategy == 'sellWeakVolumePullback':
                        signal = sellWeakVolumePullback(candle, prev_candle, wick_to_body, percentDecrease)

                    elif sellStrategy == 'sellStrongCandle':
                        signal = sellStrongCandle(candle, wick_to_body)

                    elif sellStrategy == 'sellStrongCandleVix':
                        signal = sellStrongCandleVix(candle, wick_to_body, vix_threshold)

                    elif sellStrategy == 'sellStrongFirstCandle':
                        signal = sellStrongFirstCandle(candle, wick_to_body, i)

                    if signal:
                        trades = executeSellTrade(df_day, i, candle, trades,
                                                  entry_type_sell, sl_type_sell, tp, sl, sl_extra,
                                                  entry_pullback_points, num_candles_exit, tp_entry)

    return trades


