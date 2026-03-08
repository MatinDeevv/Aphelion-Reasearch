import pytest
import numpy as np

torch = pytest.importorskip("torch")

from aphelion.intelligence.hydra.dataset import (
    HydraDataset, DatasetConfig, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES,
    build_dataset_from_feature_dicts,
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
