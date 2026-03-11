import pytest
import numpy as np
import pandas as pd

torch = pytest.importorskip("torch")

from aphelion.intelligence.hydra.dataset import (
    HydraDataset, DatasetConfig, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES,
    build_dataset_from_feature_dicts, build_dataset_from_dataframe,
    create_dataloaders,
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
    """Build a DataFrame equivalent to _make_feature_dicts for testing."""
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
    def test_returns_three_datasets_and_stats(self):
        df = _make_feature_dataframe(200)
        rng = np.random.default_rng(1)
        close = np.cumsum(rng.normal(0, 0.5, 200)) + 2000.0
        cfg = DatasetConfig(lookback_bars=32, batch_size=8)
        result = build_dataset_from_dataframe(df, close, cfg)
        assert len(result) == 5
        train_ds, val_ds, test_ds, means, stds = result
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert means.shape == (len(CONTINUOUS_FEATURES),)
        assert stds.shape == (len(CONTINUOUS_FEATURES),)

    def test_getitem_shape_matches_config(self):
        df = _make_feature_dataframe(200)
        rng = np.random.default_rng(2)
        close = np.cumsum(rng.normal(0, 0.5, 200)) + 2000.0
        cfg = DatasetConfig(lookback_bars=32, batch_size=8)
        train_ds, _, _, _, _ = build_dataset_from_dataframe(df, close, cfg)
        cont_seq, cat_seq, y5, y15, y1h, raw_ret = train_ds[0]
        assert cont_seq.shape == (32, len(CONTINUOUS_FEATURES))
        assert cat_seq.shape == (32, len(CATEGORICAL_FEATURES))
        assert raw_ret.shape == (3,)

    def test_matches_dict_builder_output_shape(self):
        """DataFrame builder should produce same-shaped outputs as dict builder."""
        rng = np.random.default_rng(0)
        n = 200
        close = np.cumsum(rng.normal(0, 0.5, n)) + 2000.0
        cfg = DatasetConfig(lookback_bars=32, batch_size=8, val_split=0.15, test_split=0.15)

        df = _make_feature_dataframe(n)
        # Override close column to match close_prices arg
        df["close"] = close
        fds = _make_feature_dicts(n)
        for i, d in enumerate(fds):
            d["close"] = float(close[i])

        train_df, val_df, test_df, _, _ = build_dataset_from_dataframe(df, close, cfg)
        train_fd, val_fd, test_fd, _, _ = build_dataset_from_feature_dicts(fds, close, cfg)

        assert len(train_df) == len(train_fd)
        assert len(val_df) == len(val_fd)
        assert len(test_df) == len(test_fd)

    def test_missing_columns_default_to_zero(self):
        """DataFrame with only close/session/day_of_week should not crash."""
        n = 200
        rng = np.random.default_rng(3)
        close = np.cumsum(rng.normal(0, 0.5, n)) + 2000.0
        df = pd.DataFrame({
            "close": close,
            "session": ["LONDON"] * n,
            "day_of_week": ["MON"] * n,
        })
        cfg = DatasetConfig(lookback_bars=32, batch_size=8)
        train_ds, _, _, means, stds = build_dataset_from_dataframe(df, close, cfg)
        assert len(train_ds) > 0
        assert np.isfinite(means).all()


class TestCreateDataloadersConfig:
    def test_num_workers_zero_no_persistent_workers(self):
        """With num_workers=0, persistent_workers must not be passed (PyTorch restriction)."""
        df = _make_feature_dataframe(200)
        rng = np.random.default_rng(4)
        close = np.cumsum(rng.normal(0, 0.5, 200)) + 2000.0
        cfg = DatasetConfig(lookback_bars=32, batch_size=8, num_workers=0)
        train_ds, val_ds, test_ds, _, _ = build_dataset_from_dataframe(df, close, cfg)
        # Should not raise
        tl, vl, tel = create_dataloaders(train_ds, val_ds, test_ds, cfg)
        assert tl.num_workers == 0

    def test_dataset_config_has_persistent_workers_and_prefetch_factor(self):
        cfg = DatasetConfig()
        assert hasattr(cfg, "persistent_workers")
        assert hasattr(cfg, "prefetch_factor")
        assert cfg.prefetch_factor == 2
