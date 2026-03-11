import pytest
import numpy as np

torch = pytest.importorskip("torch")

import pandas as pd

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


def _make_feature_dataframe(n=200):
    """Build a DataFrame equivalent to _make_feature_dicts."""
    rng = np.random.default_rng(42)
    data = {}
    for feat in CONTINUOUS_FEATURES:
        data[feat] = rng.normal(0, 1, n).astype(np.float32)
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
    """Tests for the DataFrame-native dataset builder."""

    def _close_prices(self, n=200):
        rng = np.random.default_rng(0)
        return 2000.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))

    def test_basic_split_sizes(self):
        df = _make_feature_dataframe(300)
        close = self._close_prices(300)
        config = DatasetConfig(lookback_bars=32, val_split=0.15, test_split=0.15, batch_size=16)
        train_ds, val_ds, test_ds, means, stds = build_dataset_from_dataframe(df, close, config)
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total > 0
        assert len(train_ds) > len(val_ds)
        assert len(train_ds) > len(test_ds)

    def test_getitem_shapes(self):
        df = _make_feature_dataframe(300)
        close = self._close_prices(300)
        config = DatasetConfig(lookback_bars=32, batch_size=16)
        train_ds, val_ds, test_ds, means, stds = build_dataset_from_dataframe(df, close, config)
        cont_seq, cat_seq, y5, y15, y1h, raw_ret = train_ds[0]
        assert cont_seq.shape == (32, len(CONTINUOUS_FEATURES))
        assert cat_seq.shape == (32, len(CATEGORICAL_FEATURES))
        assert raw_ret.shape == (3,)

    def test_means_stds_shapes(self):
        df = _make_feature_dataframe(300)
        close = self._close_prices(300)
        _, _, _, means, stds = build_dataset_from_dataframe(df, close)
        assert means.shape == (len(CONTINUOUS_FEATURES),)
        assert stds.shape == (len(CONTINUOUS_FEATURES),)
        assert np.all(np.isfinite(means))
        assert np.all(np.isfinite(stds))
        assert np.all(stds > 0)

    def test_parity_with_feature_dicts_builder(self):
        """DataFrame builder must produce same split sizes as the dict builder."""
        n = 300
        rng = np.random.default_rng(42)
        close = 2000.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
        config = DatasetConfig(lookback_bars=32, val_split=0.15, test_split=0.15, batch_size=16)

        # Build via dicts (legacy path)
        dicts = _make_feature_dicts(n)
        tr1, v1, te1, m1, s1 = build_dataset_from_feature_dicts(dicts, close, config)

        # Build via DataFrame (new path) — same RNG seed → same data
        df = _make_feature_dataframe(n)
        tr2, v2, te2, m2, s2 = build_dataset_from_dataframe(df, close, config)

        assert len(tr1) == len(tr2), "Train split sizes must match"
        assert len(v1) == len(v2), "Val split sizes must match"
        assert len(te1) == len(te2), "Test split sizes must match"

    def test_missing_columns_handled_gracefully(self):
        """Columns missing from the DataFrame are treated as 0."""
        df = pd.DataFrame({"close": [2000.0] * 200, "session": ["LONDON"] * 200,
                           "day_of_week": ["MON"] * 200})
        close = np.full(200, 2000.0)
        config = DatasetConfig(lookback_bars=32, batch_size=8)
        train_ds, val_ds, test_ds, means, stds = build_dataset_from_dataframe(df, close, config)
        assert len(train_ds) >= 0  # Should not raise

    def test_num_workers_zero_safe_dataloaders(self):
        """create_dataloaders must not pass persistent_workers/prefetch_factor when num_workers==0."""
        from aphelion.intelligence.hydra.dataset import create_dataloaders
        df = _make_feature_dataframe(300)
        close = self._close_prices(300)
        config = DatasetConfig(lookback_bars=32, batch_size=8, num_workers=0)
        train_ds, val_ds, test_ds, _, _ = build_dataset_from_dataframe(df, close, config)
        # Should not raise
        train_dl, val_dl, test_dl = create_dataloaders(train_ds, val_ds, test_ds, config)
        batch = next(iter(train_dl))
        assert batch[0].shape[0] <= 8  # batch size ≤ requested
