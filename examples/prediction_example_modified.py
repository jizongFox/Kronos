import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


def plot_prediction(kline_df, pred_df, pred_std=None):
    pred_df.index = kline_df.index[-pred_df.shape[0] :]
    sr_close = kline_df["close"]
    sr_pred_close = pred_df["close"]
    sr_close.name = "Ground Truth"
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df["volume"]
    sr_pred_volume = pred_df["volume"]
    sr_volume.name = "Ground Truth"
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(
        close_df["Ground Truth"], label="Ground Truth", color="blue", linewidth=1.5
    )
    ax1.plot(close_df["Prediction"], label="Prediction", color="red", linewidth=1.5)
    # Uncertainty band for close over the prediction horizon only: mean ± 2*std
    if pred_std is not None and "close" in pred_std.columns:
        mean_close = pd.Series(pred_df["close"].to_numpy(), index=pred_df.index)
        std_close = pd.Series(pred_std["close"].to_numpy(), index=pred_df.index)
        upper = mean_close + 2.0 * std_close
        lower = mean_close - 2.0 * std_close
        ax1.fill_between(
            pred_df.index, lower, upper, color="red", alpha=0.2, label="±2σ"
        )
    ax1.set_ylabel("Close Price", fontsize=14)
    ax1.legend(loc="lower left", fontsize=12)
    ax1.grid(True)

    ax2.plot(
        volume_df["Ground Truth"], label="Ground Truth", color="blue", linewidth=1.5
    )
    ax2.plot(volume_df["Prediction"], label="Prediction", color="red", linewidth=1.5)
    # Uncertainty band for volume over the prediction horizon only: mean ± 2*std
    if pred_std is not None and "volume" in pred_std.columns:
        mean_vol = pd.Series(pred_df["volume"].to_numpy(), index=pred_df.index)
        std_vol = pd.Series(pred_std["volume"].to_numpy(), index=pred_df.index)
        v_upper = mean_vol + 2.0 * std_vol
        v_lower = mean_vol - 2.0 * std_vol
        ax2.fill_between(
            pred_df.index, v_lower, v_upper, color="red", alpha=0.2, label="±2σ"
        )
    ax2.set_ylabel("Volume", fontsize=14)
    ax2.legend(loc="upper left", fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_prediction2(kline_df, pred_df, pred_std=None):
    kline_df.set_index("timestamps", inplace=True)
    # pred_df.set_index('timestamps', inplace=True)
    sr_close = kline_df["close"]
    sr_pred_close = pred_df["close"]
    sr_close.name = "Ground Truth"
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df["volume"]
    sr_pred_volume = pred_df["volume"]
    sr_volume.name = "Ground Truth"
    sr_pred_volume.name = "Prediction"

    # close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    # volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(sr_close, label="Ground Truth", color="blue", linewidth=1.5)
    ax1.plot(sr_pred_close, label="Prediction", color="red", linewidth=1.5)
    # Uncertainty band for close over the prediction horizon only: mean ± 2*std
    if pred_std is not None and "close" in pred_std.columns:
        mean_close = pd.Series(pred_df["close"].to_numpy(), index=pred_df.index)
        std_close = pd.Series(pred_std["close"].to_numpy(), index=pred_df.index)
        upper = mean_close + 2.0 * std_close
        lower = mean_close - 2.0 * std_close
        ax1.fill_between(
            pred_df.index, lower, upper, color="red", alpha=0.2, label="±2σ"
        )
    ax1.set_ylabel("Close Price", fontsize=14)
    ax1.legend(loc="lower left", fontsize=12)
    ax1.grid(True)

    ax2.plot(sr_volume, label="Ground Truth", color="blue", linewidth=1.5)
    ax2.plot(sr_pred_volume, label="Prediction", color="red", linewidth=1.5)
    # Uncertainty band for volume over the prediction horizon only: mean ± 2*std
    if pred_std is not None and "volume" in pred_std.columns:
        mean_vol = pd.Series(pred_df["volume"].to_numpy(), index=pred_df.index)
        std_vol = pd.Series(pred_std["volume"].to_numpy(), index=pred_df.index)
        v_upper = mean_vol + 2.0 * std_vol
        v_lower = mean_vol - 2.0 * std_vol
        ax2.fill_between(
            pred_df.index, v_lower, v_upper, color="red", alpha=0.2, label="±2σ"
        )
    ax2.set_ylabel("Volume", fontsize=14)
    ax2.legend(loc="upper left", fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# 1. Load Model and Tokenizer
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")

# 2. Instantiate Predictor
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)

# 3. Prepare Data
# df = pd.read_csv("./data/XSHG_5min_600977.csv")
# df['timestamps'] = pd.to_datetime(df['timestamps'])
"""
columns = ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']

timestamps    datetime64[ns]
open                 float64
high                 float64
low                  float64
close                float64
volume               float64
amount               float64
dtype: object
"""
import yfinance as yf

df2 = yf.download(
    "btc-usd",
    start="2025-07-23",
    interval="1h",
    auto_adjust=True,
    progress=False,
    multi_level_index=False,
)
assert df2.shape[0] > 0, "No data found."
# Convert to target schema
df2 = df2.rename(
    columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
)
# df2["volume"] = df2["volume"] / 10000000

# timestamps as timezone-naive datetime64[ns] (convert to UTC if tz-aware)
_idx = df2.index
df2["timestamps"] = _idx

# compute amount and reorder columns
df2["amount"] = df2["close"] * df2["volume"]
df2 = df2[["timestamps", "open", "high", "low", "close", "volume", "amount"]]

df = df2

lookback = 360
pred_len = 24
df = df.iloc[-(lookback):, :]
df.reset_index(drop=True, inplace=True)

x_df = df.loc[:lookback, ["open", "high", "low", "close", "volume", "amount"]]
x_timestamp = df.loc[:lookback, "timestamps"]
# y_timestamp = df.loc[lookback:lookback + pred_len - 1, 'timestamps']
y_timestamp_start = x_timestamp.iloc[-1] + pd.Timedelta(hours=1)
y_timestamp_end = y_timestamp_start + pd.Timedelta(hours=pred_len - 1)
y_timestamp = pd.Series(
    pd.date_range(start=y_timestamp_start, end=y_timestamp_end, freq="h")
)

# y_timestamp = df.loc[lookback:, 'timestamps']

# 4. Make Prediction
preds = []
for _ in range(30):
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.95,
        sample_count=1,
        verbose=True,
    )
    preds.append(pred_df)

preds_mean = pd.concat(preds).groupby(level=0).mean()
pred_std = pd.concat(preds).groupby(level=0).std()
pred_df = preds_mean

# 5. Visualize Results
print("Forecasted Data Head:")
print(pred_df.head())

# Combine historical and forecasted data for plotting
kline_df = df.loc[:lookback]

# visualize
plot_prediction2(kline_df, pred_df)
