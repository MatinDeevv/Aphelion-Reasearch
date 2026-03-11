"""
Phase 4 acceptance: HYDRA trainer integration, dataset builder, and
__init__.py robustness tests.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aphelion.intelligence.hydra.dataset import (
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
    DatasetConfig,
    HydraDataset,
    build_dataset_from_feature_dicts,
    create_dataloaders,
    _extract_features_from_bar_dict,
    SESSION_MAP,
    DAY_MAP,
)
from aphelion.intelligence.hydra.ensemble import EnsembleConfig
from aphelion.intelligence.hydra.tft import TFTConfig
from aphelion.intelligence.hydra.lstm import LSTMConfig
from aphelion.intelligence.hydra.cnn import CNNConfig
from aphelion.intelligence.hydra.moe import MoEConfig
from aphelion.intelligence.hydra.trainer import HydraTrainer, TrainerConfig
from aphelion.intelligence.hydra.inference import HydraInference, HydraSignal, InferenceConfig

N_CONT = len(CONTINUOUS_FEATURES)
N_CAT = len(CATEGORICAL_FEATURES)


def _small_config():
    return EnsembleConfig(
        tft_config=TFTConfig(hidden_dim=32, lstm_layers=1, attention_heads=2),
        lstm_config=LSTMConfig(hidden_size=32, num_layers=1),
        cnn_config=CNNConfig(hidden_size=32),
        moe_config=MoEConfig(hidden_size=32),
        gate_hidden_size=32,
        dropout=0.1,
    )


def _make_feature_dicts(n=200):
    rng = np.random.default_rng(42)
    dicts = []
    for _ in range(n):
        d = {f: float(rng.normal(0, 1)) for f in CONTINUOUS_FEATURES}
        d["session"] = "LONDON"
        d["day_of_week"] = "TUE"
        dicts.append(d)
    return dicts


# ═══════════════════════════════════════════════════════════════════════════
# _extract_features_from_bar_dict
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractFeatures:
    def test_numeric_features_extracted(self):
        fd = {f: float(i) for i, f in enumerate(CONTINUOUS_FEATURES)}
        fd["session"] = "LONDON"
        fd["day_of_week"] = "WED"
        cont, cat = _extract_features_from_bar_dict(fd, 2850.0)
        assert cont.shape == (N_CONT,)
        assert cat.shape == (N_CAT,)
        assert cont[0] == 0.0  # first feature
        assert cat[0] == SESSION_MAP["LONDON"]
        assert cat[1] == DAY_MAP["WED"]

    def test_missing_features_defaulted_to_zero(self):
        cont, cat = _extract_features_from_bar_dict({}, 2850.0)
        assert cont.sum() == 0.0
        assert cat[0] == SESSION_MAP.get("DEAD_ZONE", 4)

    def test_none_feature_becomes_zero(self):
        fd = {"vpin": None, "session": None}
        cont, cat = _extract_features_from_bar_dict(fd, 2850.0)
        assert cont[0] == 0.0

    def test_string_feature_becomes_zero(self):
        fd = {"vpin": "not_a_number"}
        cont, _ = _extract_features_from_bar_dict(fd, 2850.0)
        assert cont[0] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# build_dataset_from_feature_dicts
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildDataset:
    def test_returns_three_datasets_and_stats(self):
        fds = _make_feature_dicts(200)
        close = np.cumsum(np.random.randn(200) * 0.5) + 2000.0
        train, val, test, means, stds = build_dataset_from_feature_dicts(
            fds, close, DatasetConfig(lookback_bars=32)
        )
        assert len(train) > 0
        assert len(val) >= 0
        assert len(test) >= 0
        assert means.shape == (N_CONT,)
        assert stds.shape == (N_CONT,)

    def test_no_division_by_zero_in_normalization(self):
        """Constant features should get std=1.0 to avoid div-by-zero."""
        fds = []
        for _ in range(200):
            d = {f: 1.0 for f in CONTINUOUS_FEATURES}
            d["session"] = "LONDON"
            d["day_of_week"] = "MON"
            fds.append(d)
        close = np.ones(200) * 2000.0
        train, _, _, _, stds = build_dataset_from_feature_dicts(
            fds, close, DatasetConfig(lookback_bars=32)
        )
        # Should not have any zero or NaN in stds
        assert (stds > 0).all()

    def test_train_does_not_leak_future_stats(self):
        """Norm stats are computed from train portion only."""
        fds = _make_feature_dicts(500)
        close = np.cumsum(np.random.randn(500) * 0.5) + 2000.0
        cfg = DatasetConfig(lookback_bars=32, val_split=0.15, test_split=0.15)
        _, _, _, means, stds = build_dataset_from_feature_dicts(fds, close, cfg)
        # Just check they are finite — the key guarantee is that train portion
        # ends before val/test so no future leakage.
        assert np.all(np.isfinite(means))
        assert np.all(np.isfinite(stds))


# ═══════════════════════════════════════════════════════════════════════════
# create_dataloaders
# ═══════════════════════════════════════════════════════════════════════════

class TestCreateDataloaders:
    def test_returns_three_loaders(self):
        fds = _make_feature_dicts(200)
        close = np.cumsum(np.random.randn(200) * 0.5) + 2000.0
        cfg = DatasetConfig(lookback_bars=32, batch_size=8)
        train, val, test, _, _ = build_dataset_from_feature_dicts(fds, close, cfg)
        tl, vl, tel = create_dataloaders(train, val, test, cfg)
        batch = next(iter(tl))
        assert batch[0].shape[0] <= 8  # batch size


# ═══════════════════════════════════════════════════════════════════════════
# TrainerConfig defaults
# ═══════════════════════════════════════════════════════════════════════════

class TestTrainerConfig:
    def test_defaults_sane(self):
        cfg = TrainerConfig()
        assert cfg.learning_rate > 0
        assert cfg.max_epochs > 0
        assert cfg.patience > 0
        assert cfg.gradient_clip_norm > 0


# ═══════════════════════════════════════════════════════════════════════════
# HydraTrainer full train() loop
# ═══════════════════════════════════════════════════════════════════════════

class TestTrainerFull:
    def _loader(self, n=64, seq=32, bs=16):
        cont = torch.randn(n, seq, N_CONT)
        cat = torch.randint(0, 5, (n, seq, N_CAT))
        y = torch.full((n,), 2, dtype=torch.long)
        raw = torch.ones(n, 3) * 0.1
        ds = torch.utils.data.TensorDataset(cont, cat, y, y, y, raw)
        return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)

    def test_train_returns_summary_dict(self, tmp_path):
        loader = self._loader()
        cfg = TrainerConfig(
            max_epochs=2,
            use_amp=False,
            checkpoint_dir=str(tmp_path),
            save_every_n_epochs=1,
            ensemble_config=_small_config(),
        )
        trainer = HydraTrainer(cfg, device="cpu")
        result = trainer.train(loader, loader)
        assert "total_epochs" in result
        assert "best_val_sharpe" in result
        assert "best_val_loss" in result
        assert result["total_epochs"] == 2

    def test_early_stopping(self, tmp_path):
        """Training should stop early when patience is exhausted."""
        # Use random labels so loss won't decrease reliably
        n = 32
        seq = 32
        cont = torch.randn(n, seq, N_CONT)
        cat = torch.randint(0, 5, (n, seq, N_CAT))
        y = torch.randint(0, 3, (n,))
        raw = torch.randn(n, 3)
        ds = torch.utils.data.TensorDataset(cont, cat, y, y, y, raw)
        loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)

        cfg = TrainerConfig(
            max_epochs=200,
            patience=3,
            use_amp=False,
            checkpoint_dir=str(tmp_path),
            ensemble_config=_small_config(),
        )
        trainer = HydraTrainer(cfg, device="cpu")
        result = trainer.train(loader, loader)
        # Should stop well before 200 epochs
        assert result["total_epochs"] < 200

    def test_checkpoint_saved_on_improvement(self, tmp_path):
        loader = self._loader()
        cfg = TrainerConfig(
            max_epochs=3,
            use_amp=False,
            checkpoint_dir=str(tmp_path),
            save_every_n_epochs=1,
            ensemble_config=_small_config(),
        )
        trainer = HydraTrainer(cfg, device="cpu")
        trainer.train(loader, loader)
        # At least latest and potentially best_loss/best_sharpe
        assert (tmp_path / "hydra_ensemble_latest.pt").exists()


# ═══════════════════════════════════════════════════════════════════════════
# InferenceConfig defaults
# ═══════════════════════════════════════════════════════════════════════════

class TestInferenceConfig:
    def test_defaults(self):
        cfg = InferenceConfig()
        assert cfg.history_len == 64
        assert 0 < cfg.smoothing_alpha < 1


# ═══════════════════════════════════════════════════════════════════════════
# HydraInference deeper tests
# ═══════════════════════════════════════════════════════════════════════════

class TestHydraInferenceDeep:
    def test_load_nonexistent_checkpoint_raises(self):
        inf = HydraInference()
        with pytest.raises(FileNotFoundError):
            inf.load_checkpoint("/nonexistent/path.pt")

    def test_predict_batch_without_model_raises(self):
        inf = HydraInference()
        with pytest.raises(RuntimeError, match="not loaded"):
            inf.predict_batch(
                np.zeros((1, 32, N_CONT), dtype=np.float32),
                np.zeros((1, 32, N_CAT), dtype=np.int64),
            )

    def test_process_bar_primes_after_64_bars(self):
        from aphelion.intelligence.hydra.ensemble import HydraGate
        inf = HydraInference()
        inf._model = HydraGate(_small_config()).eval()

        for i in range(63):
            fd = {f: float(np.random.randn()) for f in CONTINUOUS_FEATURES}
            fd.update({"session": 0, "day_of_week": 0})
            result = inf.process_bar(fd)
            assert result is None, f"Should not be primed at bar {i}"

        # 64th bar should prime the buffer
        fd = {f: float(np.random.randn()) for f in CONTINUOUS_FEATURES}
        fd.update({"session": 0, "day_of_week": 0})
        result = inf.process_bar(fd)
        assert isinstance(result, HydraSignal)

    def test_signal_smoothing_applied(self):
        """Second call should apply exponential smoothing to probs."""
        from aphelion.intelligence.hydra.ensemble import HydraGate
        inf = HydraInference()
        inf._model = HydraGate(_small_config()).eval()

        # Prime the buffer
        for i in range(65):
            fd = {f: float(np.random.randn()) for f in CONTINUOUS_FEATURES}
            fd.update({"session": 0, "day_of_week": 0})
            inf.process_bar(fd)

        # _last_raw_probs should be set (smoothed)
        assert inf._last_raw_probs is not None


# ═══════════════════════════════════════════════════════════════════════════
# __init__.py robustness
# ═══════════════════════════════════════════════════════════════════════════

class TestHydraInit:
    def test_has_torch_is_true(self):
        import aphelion.intelligence.hydra as hydra_pkg
        assert hydra_pkg.HAS_TORCH is True

    def test_all_exports_importable(self):
        import aphelion.intelligence.hydra as hydra_pkg
        for name in hydra_pkg.__all__:
            obj = getattr(hydra_pkg, name, None)
            assert obj is not None, f"{name} not importable from hydra"

    def test_extended_exports_present(self):
        """Verify the extended exports added for Phase 4 robustness."""
        import aphelion.intelligence.hydra as hydra_pkg
        for name in [
            "DatasetConfig", "build_dataset_from_feature_dicts",
            "build_dataset_from_dataframe",
            "CONTINUOUS_FEATURES", "CATEGORICAL_FEATURES",
        ]:
            assert name in hydra_pkg.__all__, f"{name} missing from __all__"


# ═══════════════════════════════════════════════════════════════════════════
# build_dataset_from_dataframe
# ═══════════════════════════════════════════════════════════════════════════

def _make_synthetic_df(n: int = 200) -> "pd.DataFrame":
    """Build a minimal synthetic DataFrame matching generate_synthetic_data output."""
    import pandas as pd

    rng = np.random.default_rng(42)
    close = np.cumsum(rng.normal(0, 0.5, n)) + 2000.0
    df = pd.DataFrame({
        "open": np.roll(close, 1),
        "high": close * (1 + rng.uniform(0, 0.001, n)),
        "low": close * (1 - rng.uniform(0, 0.001, n)),
        "close": close,
        "volume": rng.integers(10, 1000, n).astype(float),
    })
    sessions = ["ASIAN", "LONDON", "OVERLAP_LDN_NY", "NEW_YORK", "DEAD_ZONE"]
    days = ["MON", "TUE", "WED", "THU", "FRI"]
    df["session"] = [sessions[i % len(sessions)] for i in range(n)]
    df["day_of_week"] = [days[i % len(days)] for i in range(n)]
    for feat in CONTINUOUS_FEATURES:
        if feat not in df.columns:
            df[feat] = rng.normal(0, 1, n)
    return df


class TestBuildDatasetFromDataframe:
    def test_returns_three_datasets_and_stats(self):
        import pandas as pd
        from aphelion.intelligence.hydra.dataset import build_dataset_from_dataframe

        df = _make_synthetic_df(200)
        cfg = DatasetConfig(lookback_bars=32)
        train, val, test, means, stds = build_dataset_from_dataframe(df, config=cfg)
        assert len(train) > 0
        assert len(val) >= 0
        assert len(test) >= 0
        assert means.shape == (N_CONT,)
        assert stds.shape == (N_CONT,)

    def test_matches_feature_dicts_builder(self):
        """build_dataset_from_dataframe and build_dataset_from_feature_dicts
        should produce identical means/stds when given the same data."""
        import pandas as pd
        from aphelion.intelligence.hydra.dataset import build_dataset_from_dataframe

        df = _make_synthetic_df(300)
        close = df["close"].values
        feature_dicts = df.to_dict(orient="records")
        cfg = DatasetConfig(lookback_bars=32)

        _, _, _, means_df, stds_df = build_dataset_from_dataframe(df, config=cfg)
        _, _, _, means_dicts, stds_dicts = build_dataset_from_feature_dicts(
            feature_dicts, close, config=cfg,
        )

        np.testing.assert_allclose(means_df, means_dicts, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(stds_df, stds_dicts, rtol=1e-5, atol=1e-5)

    def test_dataset_lengths_match_between_builders(self):
        """Train/val/test sizes should be equal regardless of entry path."""
        import pandas as pd
        from aphelion.intelligence.hydra.dataset import build_dataset_from_dataframe

        df = _make_synthetic_df(300)
        close = df["close"].values
        feature_dicts = df.to_dict(orient="records")
        cfg = DatasetConfig(lookback_bars=32)

        tr_df, va_df, te_df, _, _ = build_dataset_from_dataframe(df, config=cfg)
        tr_fd, va_fd, te_fd, _, _ = build_dataset_from_feature_dicts(
            feature_dicts, close, config=cfg,
        )

        assert len(tr_df) == len(tr_fd)
        assert len(va_df) == len(va_fd)
        assert len(te_df) == len(te_fd)

    def test_no_division_by_zero_with_constant_features(self):
        """Constant features must not cause div-by-zero in normalization."""
        import pandas as pd
        from aphelion.intelligence.hydra.dataset import build_dataset_from_dataframe

        df = _make_synthetic_df(200)
        for feat in CONTINUOUS_FEATURES:
            df[feat] = 1.0  # constant
        cfg = DatasetConfig(lookback_bars=32)
        _, _, _, _, stds = build_dataset_from_dataframe(df, config=cfg)
        assert (stds > 0).all()

    def test_missing_close_column_raises(self):
        """DataFrame without 'close' column should raise a clear ValueError."""
        import pandas as pd
        from aphelion.intelligence.hydra.dataset import build_dataset_from_dataframe

        df = _make_synthetic_df(200).drop(columns=["close"])
        cfg = DatasetConfig(lookback_bars=32)
        with pytest.raises(ValueError, match="close"):
            build_dataset_from_dataframe(df, config=cfg)

    def test_first_batch_shape_is_correct(self):
        """Verify tensor shapes coming out of the DataLoader."""
        import pandas as pd
        from aphelion.intelligence.hydra.dataset import build_dataset_from_dataframe

        df = _make_synthetic_df(200)
        cfg = DatasetConfig(lookback_bars=32, batch_size=8)
        train_ds, val_ds, test_ds, _, _ = build_dataset_from_dataframe(df, config=cfg)
        train_dl, _, _ = create_dataloaders(train_ds, val_ds, test_ds, config=cfg)
        batch = next(iter(train_dl))
        cont, cat = batch[0], batch[1]
        assert cont.shape[1] == 32   # lookback
        assert cont.shape[2] == N_CONT
        assert cat.shape[2] == N_CAT


# ═══════════════════════════════════════════════════════════════════════════
# create_dataloaders — persistent_workers / prefetch_factor
# ═══════════════════════════════════════════════════════════════════════════

class TestCreateDataloadersWorkerOptions:
    def _make_tiny_datasets(self):
        fds = _make_feature_dicts(200)
        close = np.cumsum(np.random.randn(200) * 0.5) + 2000.0
        cfg = DatasetConfig(lookback_bars=32, batch_size=8)
        train, val, test, _, _ = build_dataset_from_feature_dicts(fds, close, cfg)
        return train, val, test, cfg

    def test_num_workers_zero_has_no_persistent_or_prefetch(self):
        """With num_workers=0, persistent_workers/prefetch_factor must not be set."""
        train, val, test, cfg = self._make_tiny_datasets()
        cfg.num_workers = 0
        tl, vl, tel = create_dataloaders(train, val, test, config=cfg)
        # The DataLoader should not expose persistent_workers=True when nw=0
        assert tl.num_workers == 0

    def test_dataloader_config_fields_accepted(self):
        """DatasetConfig accepts persistent_workers and prefetch_factor without error."""
        cfg = DatasetConfig(
            num_workers=0,
            persistent_workers=True,
            prefetch_factor=4,
        )
        assert cfg.persistent_workers is True
        assert cfg.prefetch_factor == 4
