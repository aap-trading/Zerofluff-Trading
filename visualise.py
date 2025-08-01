import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
import matplotlib.pyplot as plt

# Set renderer to open in browser (important for Spyder)
pio.renderers.default = "browser"

def visualiseCandleSticks(df, trades_df):
    import numpy as np

    # Create figure
    fig = go.Figure()

    # Strong Bullish Candles (Dark Green)
    strong_bullish = df[(df['close'] > df['open']) & df['IsStrongBullish']]
    fig.add_trace(go.Candlestick(
        x=strong_bullish.index,
        open=strong_bullish['open'],
        high=strong_bullish['high'],
        low=strong_bullish['low'],
        close=strong_bullish['close'],
        increasing_line_color='darkgreen',
        decreasing_line_color='darkgreen',
        showlegend=False
    ))

    # Normal Bullish Candles (Light Green)
    normal_bullish = df[(df['close'] > df['open']) & ~df['IsStrongBullish']]
    fig.add_trace(go.Candlestick(
        x=normal_bullish.index,
        open=normal_bullish['open'],
        high=normal_bullish['high'],
        low=normal_bullish['low'],
        close=normal_bullish['close'],
        increasing_line_color='lightgreen',
        decreasing_line_color='lightgreen',
        showlegend=False
    ))

    # Strong Bearish Candles (Dark Red)
    strong_bearish = df[(df['close'] < df['open']) & df['IsStrongBearish']]
    fig.add_trace(go.Candlestick(
        x=strong_bearish.index,
        open=strong_bearish['open'],
        high=strong_bearish['high'],
        low=strong_bearish['low'],
        close=strong_bearish['close'],
        increasing_line_color='darkred',
        decreasing_line_color='darkred',
        showlegend=False
    ))

    # Normal Bearish Candles (Light Red)
    normal_bearish = df[(df['close'] < df['open']) & ~df['IsStrongBearish']]
    fig.add_trace(go.Candlestick(
        x=normal_bearish.index,
        open=normal_bearish['open'],
        high=normal_bearish['high'],
        low=normal_bearish['low'],
        close=normal_bearish['close'],
        increasing_line_color='salmon',
        decreasing_line_color='salmon',
        showlegend=False
    ))

    # Volume bar colors
    volume_colors = ['orange' if is_outlier else 'steelblue' for is_outlier in df['IsIncreasingOutlierVolume']]

    # Add volume bars
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        marker_color=volume_colors,
        name='Volume',
        yaxis='y2',
        opacity=0.6
    ))

    # Add volume moving average
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Volume MA'],
        mode='lines',
        name='Volume MA',
        line=dict(color='purple', width=1.5),
        yaxis='y2'
    ))

    # Add triangle markers for all signals
    for _, row in trades_df.iterrows():
        time = pd.to_datetime(row['Signal Candle'])
        if time not in df.index:
            continue

        color = 'green' if row['Signal Type'] in ['Buy', 'BuyNoEntry'] else 'red'
        symbol = 'triangle-up' if color == 'green' else 'triangle-down'
        price = df.loc[time]['low'] if color == 'green' else df.loc[time]['high']

        fig.add_trace(go.Scatter(
            x=[time],
            y=[price],
            mode='markers',
            marker=dict(color=color, size=12, symbol=symbol),
            name=row['Signal Type'],
            showlegend=False
        ))

    # Add Entry and Exit Price lines for completed trades
    completed_trades = trades_df[trades_df['Signal Type'].isin(['Buy', 'Sell'])]

    for _, trade in completed_trades.iterrows():
        entry_time = pd.to_datetime(trade['Entry Candle'])
        exit_time = pd.to_datetime(trade['Exit Candle'])

        # Entry line (white)
        if pd.notna(entry_time) and pd.notna(trade['Entry Price']) and entry_time in df.index:
            fig.add_shape(
                type="line",
                x0=entry_time, y0=trade['Entry Price'],
                x1=entry_time, y1=trade['Entry Price'],
                line=dict(color="white", width=2),
                xref='x', yref='y'
            )

        # Exit line (blue)
        if pd.notna(exit_time) and pd.notna(trade['Exit Price']) and exit_time in df.index:
            fig.add_shape(
                type="line",
                x0=exit_time, y0=trade['Exit Price'],
                x1=exit_time, y1=trade['Exit Price'],
                line=dict(color="blue", width=2),
                xref='x', yref='y'
            )

    # Layout settings
    fig.update_layout(
        title='Candlestick Chart with Trades and Volume',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Price'),
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        height=700,
        hovermode='x unified'
    )

    fig.show()


def plot_cumulative_pnl(df):
    """
    Plots the Cumulative PnL over time, assuming 'Date' and 'CumulativePnL' columns.
    Will try to reset index if 'Date' is not a column.
    """
    df = df.copy()

    # Try to recover 'Date' if it's in the index
    if 'Date' not in df.columns:
        if df.index.name == 'Date':
            df = df.reset_index()
        else:
            raise KeyError("'Date' column not found. Make sure your DataFrame has a 'Date' column or index.")

    # Confirm required columns exist
    if 'CumulativePnL' not in df.columns:
        raise KeyError("'CumulativePnL' column not found in DataFrame.")

    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['CumulativePnL'], marker='o', linestyle='-', color='blue', label='Cumulative PnL')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title('Cumulative PnL Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
