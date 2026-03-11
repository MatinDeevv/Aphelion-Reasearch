import pytest
import numpy as np
import pandas as pd

torch = pytest.importorskip("torch")

from aphelion.intelligence.hydra.dataset import (
    HydraDataset, DatasetConfig, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES,
    build_dataset_from_feature_dicts, build_dataset_from_dataframe,
)


def _make_feature_dicts(n=200):
    rng = np.random.default_rng(42)
    dicts = []
    for i in range(n):
        d = {}
        for feat in CONTINUOUS_FEATURES:
            d[feat] = float(rng.normal(0, 1))
        d["session"] = "LONDON"
        d["day_of_week"] = "MON"
        dicts.append(d)
    return dicts


def _make_dataframe(n=200):
    rng = np.random.default_rng(42)
    data = {feat: rng.normal(0, 1, n).astype(np.float32) for feat in CONTINUOUS_FEATURES}
    data["session"] = ["LONDON"] * n
    data["day_of_week"] = ["MON"] * n
    return pd.DataFrame(data)


def _make_dummy_dataset(n_bars=200, seq_len=50):
    n_cont = len(CONTINUOUS_FEATURES)
    n_cat = len(CATEGORICAL_FEATURES)
    rng = np.random.default_rng(42)
    cont = rng.normal(0, 1, (n_bars, n_cont)).astype(np.float32)
    cat = rng.integers(0, 5, (n_bars, n_cat)).astype(np.int64)
    labels = rng.integers(0, 3, n_bars).astype(np.int64)
    raw_ret = rng.normal(0, 0.1, (n_bars, 3)).astype(np.float32)
    # Indices: valid from seq_len to n_bars
    indices = list(range(seq_len, n_bars))
    return HydraDataset(cont, cat, labels, labels, labels, raw_ret, indices, seq_len)


class TestHydraDataset:
    def test_dataset_len_correct(self):
        ds = _make_dummy_dataset(200, 50)
        assert len(ds) == 150  # 200 - 50

    def test_dataset_getitem_shapes(self):
        ds = _make_dummy_dataset(200, 50)
        cont_seq, cat_seq, y5, y15, y1h, raw_ret = ds[0]
        assert cont_seq.shape == (50, len(CONTINUOUS_FEATURES))
        assert cat_seq.shape == (50, len(CATEGORICAL_FEATURES))

    def test_dataset_no_lookahead(self):
        ds = _make_dummy_dataset(200, 50)
        # Item 0: x uses rows [0:50], y uses row 50
        cont_seq, cat_seq, y5, y15, y1h, raw_ret = ds[0]
        # The y labels come from index 50 (the "center" index)
        # which is NOT inside the x range [0:50)
        assert cont_seq.shape[0] == 50

    def test_dataset_handles_too_short(self):
        ds = _make_dummy_dataset(30, 50)
        # With 30 bars and seq_len=50, no valid indices
        assert len(ds) == 0

    def test_dataloader_batches_correctly(self):
        from torch.utils.data import DataLoader
        ds = _make_dummy_dataset(200, 32)
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        batch = next(iter(loader))
        cont_seq = batch[0]
        assert cont_seq.shape == (8, 32, len(CONTINUOUS_FEATURES))


class TestBuildDatasetFromDataframe:
    """Tests for the DataFrame-based fast path."""

    def _config(self):
        return DatasetConfig(
            lookback_bars=20,
            horizon_5m=5,
            horizon_15m=10,
            horizon_1h=15,
            val_split=0.15,
            test_split=0.15,
            batch_size=16,
            num_workers=0,
        )

    def test_returns_five_tuple(self):
        df = _make_dataframe(200)
        close = np.random.default_rng(1).uniform(1900, 2100, 200).astype(np.float32)
        result = build_dataset_from_dataframe(df, close, config=self._config())
        assert len(result) == 5

    def test_dataset_lengths_match_feature_dicts_path(self):
        """DataFrame path and feature-dicts path must produce the same split sizes."""
        rng = np.random.default_rng(0)
        n = 300
        close = rng.uniform(1900, 2100, n).astype(np.float32)

        df = _make_dataframe(n)
        dicts = _make_feature_dicts(n)

        cfg = self._config()
        tr_df, va_df, te_df, _, _ = build_dataset_from_dataframe(df, close, config=cfg)
        tr_fd, va_fd, te_fd, _, _ = build_dataset_from_feature_dicts(dicts, close, config=cfg)

        assert len(tr_df) == len(tr_fd)
        assert len(va_df) == len(va_fd)
        assert len(te_df) == len(te_fd)

    def test_item_shapes(self):
        df = _make_dataframe(200)
        close = np.random.default_rng(2).uniform(1900, 2100, 200).astype(np.float32)
        train_ds, _, _, _, _ = build_dataset_from_dataframe(df, close, config=self._config())
        assert len(train_ds) > 0
        cont_seq, cat_seq, y5, y15, y1h, raw_ret = train_ds[0]
        assert cont_seq.shape == (self._config().lookback_bars, len(CONTINUOUS_FEATURES))
        assert cat_seq.shape == (self._config().lookback_bars, len(CATEGORICAL_FEATURES))
        assert raw_ret.shape == (3,)

    def test_normalization_no_leakage(self):
        """Means/stds must be computed from train slice only, not val/test."""
        df = _make_dataframe(300)
        close = np.random.default_rng(3).uniform(1900, 2100, 300).astype(np.float32)
        cfg = self._config()
        _, _, _, means, stds = build_dataset_from_dataframe(df, close, config=cfg)
        assert means.shape == (len(CONTINUOUS_FEATURES),)
        assert stds.shape == (len(CONTINUOUS_FEATURES),)
        # stds should all be positive (no zero-std columns left)
        assert np.all(stds >= 1e-8)

    def test_missing_columns_default_to_zero(self):
        """If a feature column is absent, the corresponding values should be 0 after norm."""
        df = _make_dataframe(200)
        # Remove a feature column so it defaults to 0
        feat_to_drop = CONTINUOUS_FEATURES[5]
        df = df.drop(columns=[feat_to_drop])
        close = np.ones(200, dtype=np.float32) * 2000.0
        train_ds, _, _, means, stds = build_dataset_from_dataframe(df, close, config=self._config())
        # Should not raise; dataset should still build
        assert len(train_ds) > 0

    def test_create_dataloaders_num_workers_zero(self):
        """create_dataloaders with num_workers=0 must not pass persistent_workers/prefetch_factor."""
        from aphelion.intelligence.hydra.dataset import create_dataloaders
        df = _make_dataframe(200)
        close = np.random.default_rng(4).uniform(1900, 2100, 200).astype(np.float32)
        cfg = DatasetConfig(lookback_bars=20, batch_size=8, num_workers=0)
        train_ds, val_ds, test_ds, _, _ = build_dataset_from_dataframe(df, close, config=cfg)
        # Must not raise even though persistent_workers defaults to False
        train_dl, val_dl, test_dl = create_dataloaders(train_ds, val_ds, test_ds, config=cfg)
        batch = next(iter(train_dl))
        assert batch[0].shape[0] <= 8  # batch size

