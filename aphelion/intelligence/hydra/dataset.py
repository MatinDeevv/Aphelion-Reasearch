"""
APHELION HYDRA Dataset
PyTorch Dataset for TFT training on XAU/USD bar features.
Sliding window over 60+ features → multi-horizon directional targets.
"""

from __future__ import annotations

import datetime
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
            v = float(val)
            cont[i] = v if np.isfinite(v) else 0.0
        elif isinstance(val, str):
            cont[i] = 0.0
        elif val is None:
            cont[i] = 0.0
        else:
            v = float(val)
            cont[i] = v if np.isfinite(v) else 0.0

    cat = np.zeros(len(CATEGORICAL_FEATURES), dtype=np.int64)
    session_val = features.get("session", "DEAD_ZONE")
    if isinstance(session_val, str):
        cat[0] = SESSION_MAP.get(session_val, 4)
    day_val = features.get("day_of_week", "MON")
    if isinstance(day_val, str):
        cat[1] = DAY_MAP.get(day_val, 0)

    return cont, cat


def _ts() -> str:
    """Return a compact HH:MM:SS timestamp for progress prints."""
    return datetime.datetime.now().strftime("%H:%M:%S")


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

    print(f"  [{_ts()}] Building dataset from {n_bars:,} bars "
          f"({n_cont} continuous + {n_cat} categorical features)...")

    # OPTIMIZED: Vectorized feature extraction using pandas DataFrame
    # Avoids 100K+ Python dict lookups per feature
    import pandas as pd
    if isinstance(feature_dicts, list) and len(feature_dicts) > 0:
        print(f"  [{_ts()}] Converting {n_bars:,} feature dicts → DataFrame...")
        df_features = pd.DataFrame(feature_dicts)
        print(f"  [{_ts()}] DataFrame ready ({df_features.shape[0]:,} rows × {df_features.shape[1]} cols)")

        # Extract continuous features — vectorized
        print(f"  [{_ts()}] Extracting continuous features ({n_cont} cols)...")
        cont_matrix = np.zeros((n_bars, n_cont), dtype=np.float32)
        for i, key in enumerate(CONTINUOUS_FEATURES):
            if key in df_features.columns:
                col = pd.to_numeric(df_features[key], errors='coerce').values.astype(np.float32)
                # Replace NaN/Inf with 0
                bad_mask = ~np.isfinite(col)
                if bad_mask.any():
                    col[bad_mask] = 0.0
                cont_matrix[:, i] = col

        # Extract categorical features — vectorized
        print(f"  [{_ts()}] Extracting categorical features ({n_cat} cols)...")
        cat_matrix = np.zeros((n_bars, n_cat), dtype=np.int64)
        if 'session' in df_features.columns:
            cat_matrix[:, 0] = df_features['session'].map(
                lambda x: SESSION_MAP.get(x, 4) if isinstance(x, str) else 4
            ).values
        if 'day_of_week' in df_features.columns:
            cat_matrix[:, 1] = df_features['day_of_week'].map(
                lambda x: DAY_MAP.get(x, 0) if isinstance(x, str) else 0
            ).values
    else:
        cont_matrix = np.zeros((n_bars, n_cont), dtype=np.float32)
        cat_matrix = np.zeros((n_bars, n_cat), dtype=np.int64)

    # Normalize continuous features (z-score)
    # Use only training portion for stats to prevent leakage
    n_usable = n_bars - max_horizon
    train_end = int(n_usable * (1 - config.val_split - config.test_split))

    print(f"  [{_ts()}] Normalizing features (z-score, train window = {train_end:,} bars)...")
    feature_means = np.mean(cont_matrix[:train_end], axis=0)
    feature_stds = np.std(cont_matrix[:train_end], axis=0)
    feature_stds[feature_stds < 1e-8] = 1.0  # Avoid division by zero

    cont_matrix = (cont_matrix - feature_means) / feature_stds

    # Belt-and-suspenders: clean any remaining NaN/Inf after normalization
    nan_mask = ~np.isfinite(cont_matrix)
    if nan_mask.any():
        n_bad = nan_mask.sum()
        print(f"  [{_ts()}] ⚠ Cleaning {n_bad} NaN/Inf values in normalized features")
        cont_matrix = np.nan_to_num(cont_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Build direction labels for each horizon using vectorized numpy ops.
    # Replaces the former pure-Python loop which was O(n_bars) and froze on
    # large datasets (100K+ bars could take several minutes).
    print(f"  [{_ts()}] Computing returns for {n_usable:,} bars "
          f"(horizons: 5m/{config.horizon_5m}, 15m/{config.horizon_15m}, "
          f"1h/{config.horizon_1h})...")

    idx = np.arange(n_usable)
    prices = close_prices[:n_usable]
    if prices.dtype != np.float64:
        prices = prices.astype(np.float64)
    safe_mask = prices > 0
    safe_prices = np.where(safe_mask, prices, 1.0)

    idx_5m = np.minimum(idx + config.horizon_5m, n_bars - 1)
    idx_15m = np.minimum(idx + config.horizon_15m, n_bars - 1)
    idx_1h = np.minimum(idx + config.horizon_1h, n_bars - 1)

    returns_5m = np.zeros(n_bars, dtype=np.float32)
    returns_15m = np.zeros(n_bars, dtype=np.float32)
    returns_1h = np.zeros(n_bars, dtype=np.float32)

    returns_5m[:n_usable] = np.where(
        safe_mask, (close_prices[idx_5m] - prices) / safe_prices * 100, 0.0
    ).astype(np.float32)
    returns_15m[:n_usable] = np.where(
        safe_mask, (close_prices[idx_15m] - prices) / safe_prices * 100, 0.0
    ).astype(np.float32)
    returns_1h[:n_usable] = np.where(
        safe_mask, (close_prices[idx_1h] - prices) / safe_prices * 100, 0.0
    ).astype(np.float32)

    print(f"  [{_ts()}] Classifying direction labels (threshold ±{config.direction_threshold_pct}%)...")
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

    print(f"  [{_ts()}] Assembling PyTorch datasets "
          f"(train={len(train_indices):,}, val={len(val_indices):,}, "
          f"test={len(test_indices):,})...")

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

    print(f"  [{_ts()}] Dataset ready ✓")
    return train_ds, val_ds, test_ds, feature_means, feature_stds


class HydraDataset:
    """
    OPTIMIZED PyTorch-compatible dataset for TFT training.
    Pre-converts numpy arrays to contiguous tensors at construction time
    to avoid per-sample torch.from_numpy() overhead.
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
        if not HAS_TORCH:
            raise ImportError("PyTorch required")
        # Pre-convert to tensors — massive speedup in __getitem__
        self._cont = torch.from_numpy(np.ascontiguousarray(cont_matrix)).float()
        self._cat = torch.from_numpy(np.ascontiguousarray(cat_matrix)).long()
        self._labels_5m = torch.from_numpy(np.ascontiguousarray(labels_5m)).long()
        self._labels_15m = torch.from_numpy(np.ascontiguousarray(labels_15m)).long()
        self._labels_1h = torch.from_numpy(np.ascontiguousarray(labels_1h)).long()
        self._raw_returns = torch.from_numpy(np.ascontiguousarray(raw_returns)).float()
        self._indices = indices
        self._lookback = lookback

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        center = self._indices[idx]
        start = center - self._lookback

        # Tensor slicing — no numpy conversion needed, zero-copy
        return (
            self._cont[start:center],           # (lookback, n_cont)
            self._cat[start:center],             # (lookback, n_cat)
            self._labels_5m[center],
            self._labels_15m[center],
            self._labels_1h[center],
            self._raw_returns[center],           # (3,)
        )


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
