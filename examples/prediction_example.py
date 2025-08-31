import matplotlib.pyplot as plt
import pandas as pd

from kronos.model import Kronos, KronosTokenizer, KronosPredictor


def plot_prediction(kline_df, pred_df):
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
    ax1.set_ylabel("Close Price", fontsize=14)
    ax1.legend(loc="lower left", fontsize=12)
    ax1.grid(True)

    ax2.plot(
        volume_df["Ground Truth"], label="Ground Truth", color="blue", linewidth=1.5
    )
    ax2.plot(volume_df["Prediction"], label="Prediction", color="red", linewidth=1.5)
    ax2.set_ylabel("Volume", fontsize=14)
    ax2.legend(loc="upper left", fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# 1. Load Model and Tokenizer
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

# 2. Instantiate Predictor
predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=2048)
#

import yfinance as yf

df2 = yf.download(
    "voo",
    start="2020-07-23",
    interval="1d",
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
df.reset_index(drop=True, inplace=True)
lookback = 512
pred_len = 120

x_df = df.iloc[-(lookback + pred_len) : -pred_len]
x_df = x_df.loc[:, ["open", "high", "low", "close", "volume", "amount"]]
x_timestamp = df.iloc[-(lookback + pred_len) : -pred_len].loc[:, "timestamps"]
# x_timestamp = df.loc[: lookback - 1, "timestamps"]
y_timestamp = df.iloc[-(pred_len):].loc[:, "timestamps"]


# 4. Make Prediction
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True,
)

# 5. Visualize Results
print("Forecasted Data Head:")
print(pred_df.head())

# Combine historical and forecasted data for plotting
kline_df = x_df

# visualize
plot_prediction(kline_df, pred_df)
