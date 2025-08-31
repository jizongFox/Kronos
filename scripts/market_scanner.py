# %%
import copy
import typing as t
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots
from torch.utils.data import Dataset
from tqdm import tqdm

from kronos.model import KronosPredictor, KronosTokenizer, Kronos

warnings.filterwarnings("ignore", category=FutureWarning)

qlib_cn_dir = Path("~/.qlib/qlib_data/cn_data")


class MarketData(Dataset[t.Tuple[str, pd.DataFrame]]):
    def __init__(self, data_dir: Path = qlib_cn_dir):
        super().__init__()
        self.data_dir = data_dir
        self.instruments = self._get_instruments(self.data_dir)
        self.day_range = self._get_day_range(self.data_dir)

    def __len__(self):
        return len(self.instruments)

    def __getitem__(self, idx) -> t.Tuple[str, pd.DataFrame]:
        instrument_name = self.instruments.index[idx]
        return (instrument_name, self.get_feature(instrument_name))

    def _get_instruments(
        self, data_dir: Path, today: pd.Timestamp = pd.Timestamp("2025-08-29")
    ):
        instruments = pd.read_table(data_dir / "instruments" / "all.txt", sep="\t")
        instruments.columns = ["code", "start", "end"]
        instruments.start = pd.to_datetime(instruments.start)
        instruments.end = pd.to_datetime(instruments.end)
        instruments.set_index("code", inplace=True)
        validate_mask = (instruments.end) >= today
        instruments = instruments[validate_mask]

        duration = instruments.end - instruments.start
        validate_mask = duration.dt.days > 400
        instruments = instruments[validate_mask]

        instruments["code"] = instruments.index
        instruments.drop_duplicates(inplace=True)
        instruments.set_index("code", inplace=True)

        return instruments

    def _get_day_range(self, data_dir: Path):
        day_range = pd.read_table(
            data_dir / "calendars" / "day.txt", sep="\t", header=None
        ).squeeze(axis=1)
        day_range = pd.to_datetime(day_range)
        return day_range

    def get_feature(self, instrument_name: str) -> pd.DataFrame:
        start = self.instruments.loc[instrument_name, ["start"]].iloc[0]
        end = self.instruments.loc[instrument_name, ["end"]].iloc[-1]
        _day_range = self.day_range[(self.day_range > start) & (self.day_range <= end)]
        _ins_folder = qlib_cn_dir / "features" / instrument_name.lower()

        columns = ["open", "high", "low", "close", "volume", "amount"]
        data = []
        for attr in columns:
            _file_path = _ins_folder / f"{attr}.day.bin"
            _file = np.fromfile(_file_path.expanduser(), dtype=np.float32)
            _file = _file[~np.isnan(_file)]
            data.append(_file)
        data_pd = np.stack(data, axis=1)
        common_length = min(len(data_pd), len(_day_range)) - 10
        data_pd = data_pd[-common_length:]
        data_pd = pd.DataFrame(
            data_pd, index=_day_range[-common_length:], columns=columns
        )
        data_pd.index.name = "date"
        return data_pd


class Runner:
    def __init__(
        self,
        data: MarketData,
        model_name: t.Literal["mini", "small", "base"] = "base",
        lookback: int = 512,
        pred_len: int = 120,
        save_result_path: Path | None = None,
    ):
        self.data = data
        self.lookback = lookback
        self.pred_len = pred_len
        self.data = data
        self.save_result_path = save_result_path
        # 1. Load Model and Tokenizer
        if model_name == "mini":
            tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
            max_context = 2048

        else:
            tokenizer = KronosTokenizer.from_pretrained(
                "NeoQuasar/Kronos-Tokenizer-base"
            )
            max_context = 512

        model = Kronos.from_pretrained(f"NeoQuasar/Kronos-{model_name}")

        # 2. Instantiate Predictor
        self.predictor = KronosPredictor(
            model, tokenizer, device="cuda", max_context=max_context
        )

    def run_scanner(
        self,
    ):
        test_date_end = pd.Timestamp("2025-08-29")
        test_date_start = pd.Timestamp("2025-05-01")
        test_day_range = pd.date_range(test_date_start, test_date_end)

        choose_dates = test_day_range[::3]
        market_data = self.data
        eval_results = []
        for split_date in choose_dates:
            logger.info(f"Scanning instruments at {split_date}")

            for i in tqdm(
                range(len(market_data)), desc="scanning instruments", disable=True
            ):
                name, k_line = market_data[i]
                train_df = k_line[k_line.index < split_date]
                test_df = k_line[k_line.index >= split_date]
                if len(train_df) < 100:
                    logger.debug(
                        f"Instrument {name} has less than 100 training samples"
                    )
                    continue

                eval_result = self.predict(name, train_df, test_df)
                eval_result.update({"test_date": split_date})
                eval_results.append(eval_result)

                if self.save_result_path is not None and (i + 1) % 30 == 0:
                    self._save_csv(eval_results, self.save_result_path)

        if self.save_result_path is not None:
            self._save_csv(eval_results, self.save_result_path)
        return pd.DataFrame(eval_results)

    def predict(
        self,
        instrument_name: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):
        x_train, (x_timestamp, y_timestamp) = self._process_data(train_df, test_df)

        pred_df = self.predictor.predict(
            df=x_train,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp.iloc[:8],
            pred_len=8,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=True,
        )
        # self._plot_test_pred(test, pred_df, title=f"Instrument {i}")
        eval_result = self._compute_metrics(pred_df, test_df)
        eval_result.update(
            {
                "instrument_name": instrument_name,
            }
        )
        return eval_result

    def _process_data(self, tra_data: pd.DataFrame, test_data: pd.DataFrame):
        tra_data = copy.deepcopy(tra_data)
        x_timestamp = pd.Series(tra_data.index)
        x_timestamp = x_timestamp.iloc[-self.lookback :]

        tra_data.reset_index(drop=True, inplace=True)
        tra_data = tra_data.iloc[-self.lookback :, :]
        y_timestamp = pd.Series(test_data.index)
        return tra_data, (x_timestamp, y_timestamp)

    def _compute_metrics(self, pred_df, test_df):
        max_return_period = min(len(test_df), 7)
        result_dict = {}
        for return_period in range(1, max_return_period + 1):

            pred_return = (
                pred_df.close.pct_change(periods=return_period)
                .shift(-return_period)
                .dropna()
            )
            gt_return = (
                test_df.close.pct_change(periods=return_period)
                .shift(-return_period)
                .dropna()
            )
            result_dict.update(
                {
                    f"pred_return_{return_period}_days": float(pred_return.iloc[0]),
                    f"gt_return_{return_period}_days": float(gt_return.iloc[0]),
                }
            )
        return result_dict

    def _plot_test_pred(self, test: pd.DataFrame, pred_df: pd.DataFrame, title: str):
        """Plot test and prediction candlesticks in upper and lower subplots.

        Args:
            test: Ground truth OHLC dataframe with columns [open, high, low, close].
            pred_df: Predicted OHLC dataframe with columns [open, high, low, close].
            title: Figure title.
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
        )

        # Upper subplot: Ground Truth candlestick
        fig.add_trace(
            go.Candlestick(
                x=test.index,
                open=test["open"],
                high=test["high"],
                low=test["low"],
                close=test["close"],
                name="Ground Truth",
            ),
            row=1,
            col=1,
        )

        # Lower subplot: Prediction candlestick
        fig.add_trace(
            go.Candlestick(
                x=pred_df.index,
                open=pred_df["open"],
                high=pred_df["high"],
                low=pred_df["low"],
                close=pred_df["close"],
                name="Prediction",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title_text=title,
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
        )

        fig.show()

    @staticmethod
    def _save_csv(result_dict: t.List[t.Dict[str, t.Any]], save_path: Path):
        _save_file = pd.DataFrame(result_dict)
        if "instrument_name" in _save_file.columns:
            _save_file.set_index("instrument_name", inplace=True)
        _save_file.to_csv(save_path)


data = MarketData()
print(f"Total instruments: {len(data)}")
runner = Runner(data, save_result_path=Path("market_scan.csv"))
scan_result = runner.run_scanner()
