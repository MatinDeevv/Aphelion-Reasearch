"""
APHELION HYDRA Dataset
PyTorch Dataset for TFT training on XAU/USD bar features.
Sliding window over 60+ features → multi-horizon directional targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from aphelion.core.data_layer import Bar


# ─── Configuration ───────────────────────────────────────────────────────────

# Features used by TFT — ordered for consistent tensor construction.
# These match FeatureEngine output keys.
CONTINUOUS_FEATURES: list[str] = [
    # Microstructure
    "vpin", "ofi", "tick_entropy", "hawkes_buy", "hawkes_sell",
    "bid_ask_spread", "quote_depth",
    # Market structure
    "nearest_ob_distance", "nearest_fvg_distance",
    "nearest_swing_high_distance", "nearest_swing_low_distance",
    "volume_imbalance_strength",
    # Volume profile
    "volume_delta", "cumulative_delta", "poc_distance",
    "vah_distance", "val_distance", "absorption_score",
    # VWAP
    "vwap", "vwap_upper_1", "vwap_lower_1",
    "vwap_upper_2", "vwap_lower_2", "price_vs_vwap",
    # Technicals
    "atr", "rsi", "bb_width", "bb_percentile",
    "ema_20", "ema_50", "ema_cross",
    # MTF
    "mtf_alignment_score", "mtf_weighted_alignment",
    # Cointegration
    "max_spread_zscore",
    # OHLCV (raw, normalized later)
    "open", "high", "low", "close", "volume",
    # Advanced microstructure (v8)
    "hawkes_flow_acceleration",
    "hawkes_branching_ratio",
    "tsrv_volatility",
    "tsrv_noise_ratio",
    "toxicity_index",
    # Signature transform features
    "sig1_price",
    "sig1_volume",
    "sig1_spread",
    "sig2_price_volume",
    "sig2_price_spread",
    "sig2_volume_spread",
    "sig_levy_pv",
    "sig_l2_frobenius",
    # Cross-impact features
    "cross_impact_pred_return",
    "cross_impact_signal",
    "cross_impact_strength",
    "cross_impact_beta_dxy",
    "cross_impact_beta_tlt",
    "cross_impact_beta_xagusd",
]

CATEGORICAL_FEATURES: list[str] = [
    "session",          # ASIAN/LONDON/NY/OVERLAP/DEAD_ZONE → 0-4
    "day_of_week",      # MON-FRI → 0-4
]

# Session encoding
SESSION_MAP: dict[str, int] = {
    "ASIAN": 0, "LONDON": 1, "OVERLAP_LDN_NY": 2,
    "NEW_YORK": 3, "DEAD_ZONE": 4,
}
DAY_MAP: dict[str, int] = {
    "MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6,
}


@dataclass
class DatasetConfig:
    """Configuration for HYDRA dataset creation."""
    lookback_bars: int = 64           # Input sequence length
    horizon_5m: int = 5               # 5-bar horizon (5 min on M1)
    horizon_15m: int = 15             # 15-bar horizon
    horizon_1h: int = 60              # 60-bar horizon
    direction_threshold_pct: float = 0.05  # +/- 0.05% = directional move
    val_split: float = 0.15           # Validation fraction
    test_split: float = 0.15          # Test fraction
    batch_size: int = 256
    num_workers: int = 0              # 0 for Windows compatibility


def _extract_features_from_bar_dict(
    features: dict,
    close_price: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract continuous and categorical feature vectors from
    a FeatureEngine output dict.
    """
    cont = np.zeros(len(CONTINUOUS_FEATURES), dtype=np.float32)
    for i, key in enumerate(CONTINUOUS_FEATURES):
        val = features.get(key, 0.0)
        if isinstance(val, (int, float)):
            cont[i] = float(val)
        elif isinstance(val, str):
            cont[i] = 0.0
        elif val is None:
            cont[i] = 0.0
        else:
            cont[i] = float(val)

    cat = np.zeros(len(CATEGORICAL_FEATURES), dtype=np.int64)
    session_val = features.get("session", "DEAD_ZONE")
    if isinstance(session_val, str):
        cat[0] = SESSION_MAP.get(session_val, 4)
    day_val = features.get("day_of_week", "MON")
    if isinstance(day_val, str):
        cat[1] = DAY_MAP.get(day_val, 0)

    return cont, cat


def build_dataset_from_feature_dicts(
    feature_dicts: list[dict],
    close_prices: np.ndarray,
    config: Optional[DatasetConfig] = None,
) -> tuple:
    """
    Build train/val/test datasets from a list of feature dicts
    (one per bar, from FeatureEngine.on_bar()).

    Args:
        feature_dicts: List of feature dicts, one per bar (chronological).
        close_prices: Array of close prices corresponding to each bar.
        config: Dataset configuration.

    Returns:
        (train_dataset, val_dataset, test_dataset, feature_means, feature_stds)
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required. Install with: pip install -e '.[ml]'")

    config = config or DatasetConfig()
    n_bars = len(feature_dicts)
    n_cont = len(CONTINUOUS_FEATURES)
    n_cat = len(CATEGORICAL_FEATURES)
    max_horizon = max(config.horizon_5m, config.horizon_15m, config.horizon_1h)

    # Extract all features into matrices
    cont_matrix = np.zeros((n_bars, n_cont), dtype=np.float32)
    cat_matrix = np.zeros((n_bars, n_cat), dtype=np.int64)

    for i, fd in enumerate(feature_dicts):
        cont, cat = _extract_features_from_bar_dict(fd, close_prices[i])
        cont_matrix[i] = cont
        cat_matrix[i] = cat

    # Normalize continuous features (z-score)
    # Use only training portion for stats to prevent leakage
    n_usable = n_bars - max_horizon
    train_end = int(n_usable * (1 - config.val_split - config.test_split))

    feature_means = np.mean(cont_matrix[:train_end], axis=0)
    feature_stds = np.std(cont_matrix[:train_end], axis=0)
    feature_stds[feature_stds < 1e-8] = 1.0  # Avoid division by zero

    cont_matrix = (cont_matrix - feature_means) / feature_stds

    # Build direction labels for each horizon
    # Label: 0=SHORT, 1=FLAT, 2=LONG
    returns_5m = np.zeros(n_bars, dtype=np.float32)
    returns_15m = np.zeros(n_bars, dtype=np.float32)
    returns_1h = np.zeros(n_bars, dtype=np.float32)

    for i in range(n_bars - max_horizon):
        p = close_prices[i]
        if p > 0:
            returns_5m[i] = (close_prices[min(i + config.horizon_5m, n_bars - 1)] - p) / p * 100
            returns_15m[i] = (close_prices[min(i + config.horizon_15m, n_bars - 1)] - p) / p * 100
            returns_1h[i] = (close_prices[min(i + config.horizon_1h, n_bars - 1)] - p) / p * 100

    threshold = config.direction_threshold_pct

    def classify(ret: np.ndarray) -> np.ndarray:
        labels = np.ones(len(ret), dtype=np.int64)  # FLAT=1
        labels[ret > threshold] = 2   # LONG
        labels[ret < -threshold] = 0  # SHORT
        return labels

    labels_5m = classify(returns_5m)
    labels_15m = classify(returns_15m)
    labels_1h = classify(returns_1h)

    # Raw return values for quantile regression targets
    raw_returns = np.stack([returns_5m, returns_15m, returns_1h], axis=1)  # (n_bars, 3)

    # Create sliding windows
    valid_start = config.lookback_bars
    valid_end = n_bars - max_horizon

    indices = list(range(valid_start, valid_end))
    train_split_idx = int(len(indices) * (1 - config.val_split - config.test_split))
    val_split_idx = int(len(indices) * (1 - config.test_split))

    train_indices = indices[:train_split_idx]
    val_indices = indices[train_split_idx:val_split_idx]
    test_indices = indices[val_split_idx:]

    train_ds = HydraDataset(
        cont_matrix, cat_matrix, labels_5m, labels_15m, labels_1h,
        raw_returns, train_indices, config.lookback_bars,
    )
    val_ds = HydraDataset(
        cont_matrix, cat_matrix, labels_5m, labels_15m, labels_1h,
        raw_returns, val_indices, config.lookback_bars,
    )
    test_ds = HydraDataset(
        cont_matrix, cat_matrix, labels_5m, labels_15m, labels_1h,
        raw_returns, test_indices, config.lookback_bars,
    )

    return train_ds, val_ds, test_ds, feature_means, feature_stds


class HydraDataset:
    """
    PyTorch-compatible dataset for TFT training.
    Returns (continuous_seq, categorical_seq, labels_5m, labels_15m, labels_1h, raw_returns).
    """

    def __init__(
        self,
        cont_matrix: np.ndarray,
        cat_matrix: np.ndarray,
        labels_5m: np.ndarray,
        labels_15m: np.ndarray,
        labels_1h: np.ndarray,
        raw_returns: np.ndarray,
        indices: list[int],
        lookback: int,
    ):
        self._cont = cont_matrix
        self._cat = cat_matrix
        self._labels_5m = labels_5m
        self._labels_15m = labels_15m
        self._labels_1h = labels_1h
        self._raw_returns = raw_returns
        self._indices = indices
        self._lookback = lookback

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        center = self._indices[idx]
        start = center - self._lookback

        cont_seq = torch.from_numpy(self._cont[start:center])  # (lookback, n_cont)
        cat_seq = torch.from_numpy(self._cat[start:center])    # (lookback, n_cat)
        label_5m = torch.tensor(self._labels_5m[center], dtype=torch.long)
        label_15m = torch.tensor(self._labels_15m[center], dtype=torch.long)
        label_1h = torch.tensor(self._labels_1h[center], dtype=torch.long)
        raw_ret = torch.from_numpy(self._raw_returns[center])  # (3,)

        return cont_seq, cat_seq, label_5m, label_15m, label_1h, raw_ret


def create_dataloaders(
    train_ds, val_ds, test_ds,
    config: Optional[DatasetConfig] = None,
) -> tuple:
    """Create train/val/test DataLoaders."""
    if not HAS_TORCH:
        raise ImportError("PyTorch required")

    config = config or DatasetConfig()
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
