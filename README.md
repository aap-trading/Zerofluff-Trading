# 📊 Strategy Backtesting Framework

A modular Python framework for testing intraday trading strategies with configurable signal generation, entry methods, and risk management rules.

## 🔧 Extensible Design

The framework supports easy plug-in of new strategies, indicators, and exit logic for fast experimentation and research.

---

## 🧠 Supported Strategy Setups

### 🔼 Buy Setups
- **ORB / ORB-first / ORB-strict**: Breakout above Opening Range with strong candle and volume confirmation. Variants restrict to first breakout or apply stricter filters.
- **ORBR**: Reversal from OR low with bullish momentum and volume.
- **Prev Day Low R**: Reversal near previous day’s low with strong candle and rising volume.
- **BBR**: Bounce from lower Bollinger Band with volume.
- **Strong Volume & Candle**: Strong bullish candle with increasing volume.
- **Strong Candle / VIX**: Bullish candle alone or filtered by low VIX.
- **First Candle**: Strong first candle of session.
- **Weak Volume Pullback**: Pullback after strong bullish candle with low volume.

### 🔽 Sell Setups
- **ORB / ORB-first / ORB-strict**: Breakdown below Opening Range with strong bearish candle and volume confirmation. Variants restrict to first signal or apply stricter filters.
- **ORBR**: Reversal from OR high with bearish momentum and volume.
- **Prev Day High R**: Reversal near previous day’s high with strong candle and rising volume.
- **BBR**: Reversal from upper Bollinger Band with volume.
- **Strong Volume & Candle**: Strong bearish candle with increasing volume.
- **Strong Candle / VIX**: Bearish candle alone or filtered by low VIX.
- **First Candle**: Strong first candle of session.
- **Weak Volume Pullback**: Pullback after strong bearish candle with low volume.

---

## ⚙️ Execution Logic

- **Entry Types**: Next open, pullback from candle high/low, fixed-point pullback.
- **Stop Loss**: Fixed or based on signal candle’s low/high.
- **Take Profit**: Configurable level; triggers on entry or subsequent candles.
- **Exit Rules**: Stop loss, take profit, or timeout (number of candles). All positions close by day-end.

---

## 📈 Trade Analysis

The analysis pipeline includes filtering executed trades based on various conditions: excluding overlapping trades, removing trades on high-volatility (VIX) days, and selecting only the first valid trade within a specific intraday window. Summary statistics are generated across hourly, daily, weekly, and monthly periods, including cumulative profit and loss (PnL), win rate, and extreme day metrics. Additional metrics estimate how long it takes each month to hit profit or loss thresholds. Visualizations and CSV outputs support further inspection.

---

<p style="font-size: 0.9em; color: gray;">
<b>Disclaimer:</b> This framework is for educational and research purposes only. It does not constitute financial advice. No warranties are provided. Use at your own risk.
</p>
