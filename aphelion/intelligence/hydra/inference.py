"""
APHELION HYDRA Inference Engine (Ensemble Edition)
Replaces TFT inference with the Full Ensemble `HydraGate`.
Handles feature buffering, batch inference, signal smoothing,
and confidence calibration across all 4 sub-models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from aphelion.intelligence.hydra.dataset import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES
from aphelion.intelligence.hydra.ensemble import HydraGate, EnsembleConfig

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for HYDRA inference."""
    checkpoint_path: str = ""
    device: str = ""
    smoothing_alpha: float = 0.3
    history_len: int = 64
# FIXED: Added InferenceConfig and is_actionable/horizon_agreement to HydraSignal


@dataclass
class HydraSignal:
    """Output prediction from the HYDRA Ensemble."""
    direction: int         # 1=LONG, -1=SHORT, 0=FLAT
    confidence: float      # 0.0 to 1.0 (calibrated probability)
    uncertainty: float     # Aleatoric uncertainty from model
    
    # Probabilities across horizons [5m, 15m, 1h] for LONG/FLAT/SHORT
    probs_long: list[float]
    probs_short: list[float]
    
    # Output from the routing network representing the current regime
    regime_weights: dict[str, float]
    # Output from the master ensemble gate tracking which model is trusted
    gate_weights: dict[str, float]

    timestamp_ms: int = 0

    @property
    def is_actionable(self) -> bool:
        """Signal is actionable if direction != 0 and confidence > 0.55."""
        return self.direction != 0 and self.confidence > 0.55

    @property
    def horizon_agreement(self) -> float:
        """Fraction of horizons that agree on direction (0-1)."""
        # Compute from probs_long and probs_short
        dirs = []
        for pl, ps in zip(self.probs_long, self.probs_short):
            if pl > ps and pl > (1 - pl - ps):
                dirs.append(1)
            elif ps > pl and ps > (1 - pl - ps):
                dirs.append(-1)
            else:
                dirs.append(0)
        from collections import Counter
        most_common = Counter(dirs).most_common(1)[0][1]
        return most_common / 3.0


class HydraInference:
    """Live and backtest inference wrapper for HYDRA Gate."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for HYDRA inference.")

        self._device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Buffer for the 64-bar sliding window
        self._history_len = 64
        self._n_cont = len(CONTINUOUS_FEATURES)
        self._n_cat = len(CATEGORICAL_FEATURES)
        
        self._cont_buffer = np.zeros((self._history_len, self._n_cont), dtype=np.float32)
        self._cat_buffer = np.zeros((self._history_len, self._n_cat), dtype=np.int64)
        self._buffer_idx = 0
        self._is_primed = False

        self._model: Optional[HydraGate] = None
        self._config: Optional[EnsembleConfig] = None
        
        # Signal smoothing
        self._last_raw_probs = None
        self._smoothing_alpha = 0.3

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, path: str) -> None:
        """Load trained HydraGate ensemble."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=self._device)
        self._config = ckpt.get("ensemble_config", EnsembleConfig())
        
        self._model = HydraGate(self._config).to(self._device)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()
        logger.info("HYDRA Ensemble loaded from %s on %s", path, self._device)

    def process_bar(self, features: dict) -> Optional[HydraSignal]:
        """
        Process a single bar of features (live streaming).
        Returns HydraSignal if buffer is primed, else None.
        """
        cont_vals = [float(features.get(f, 0.0)) for f in CONTINUOUS_FEATURES]
        cat_vals = [int(features.get(f, 0)) for f in CATEGORICAL_FEATURES]

        self._cont_buffer[:-1] = self._cont_buffer[1:]
        self._cat_buffer[:-1] = self._cat_buffer[1:]
        
        self._cont_buffer[-1] = cont_vals
        self._cat_buffer[-1] = cat_vals

        if not self._is_primed:
            self._buffer_idx += 1
            if self._buffer_idx >= self._history_len:
                self._is_primed = True
            else:
                return None

        # Prepare tensors
        cont_t = torch.tensor(self._cont_buffer, dtype=torch.float32).unsqueeze(0).to(self._device)
        cat_t = torch.tensor(self._cat_buffer, dtype=torch.long).unsqueeze(0).to(self._device)

        return self._predict_single(cont_t, cat_t, features.get("timestamp_ms", 0))

    def predict_batch(self, cont_batch: np.ndarray, cat_batch: np.ndarray) -> list[HydraSignal]:
        """
        Process a pre-windowed batch (backtesting).
        cont_batch: (B, 64, N_CONT)
        cat_batch: (B, 64, N_CAT)
        """
        if self._model is None:
            raise RuntimeError("Model not loaded yet.")

        cont_t = torch.tensor(cont_batch, dtype=torch.float32).to(self._device)
        cat_t = torch.tensor(cat_batch, dtype=torch.long).to(self._device)

        with torch.no_grad():
            outputs = self._model(cont_t, cat_t)

        return self._format_batch_outputs(outputs)

    @torch.no_grad()
    def _predict_single(self, cont_t: torch.Tensor, cat_t: torch.Tensor, ts_ms: int) -> HydraSignal:
        if self._model is None:
            raise RuntimeError("Model not loaded yet.")

        outputs = self._model(cont_t, cat_t)
        signals = self._format_batch_outputs(outputs, [ts_ms])
        return signals[0]

    def _format_batch_outputs(self, outputs: dict, timestamps: Optional[list[int]] = None) -> list[HydraSignal]:
        signals = []
        batch_size = outputs["probs_1h"].shape[0]

        probs_1h = outputs["probs_1h"].cpu().numpy()
        probs_15m = outputs["probs_15m"].cpu().numpy()
        probs_5m = outputs["probs_5m"].cpu().numpy()
        uncertainty = outputs["uncertainty"].cpu().numpy().flatten()
        
        gate_weights = outputs.get("gate_attention_weights", torch.zeros(batch_size, 1, 4)).cpu().numpy()
        moe_weights = outputs.get("moe_routing_weights", torch.zeros(batch_size, 4)).cpu().numpy()

        for i in range(batch_size):
            p1 = probs_1h[i] # [SHORT, FLAT, LONG]

            if self._last_raw_probs is None:
                self._last_raw_probs = p1
            else:
                # Exponential smoothing of probabilities
                self._last_raw_probs = (
                    self._smoothing_alpha * p1 + 
                    (1 - self._smoothing_alpha) * self._last_raw_probs
                )
                p1 = self._last_raw_probs

            direction = 0
            confidence = float(p1[1]) # Default flat
            
            p_short, p_flat, p_long = p1[0], p1[1], p1[2]

            if p_long > p_flat and p_long > p_short:
                direction = 1
                confidence = float(p_long)
            elif p_short > p_flat and p_short > p_long:
                direction = -1
                confidence = float(p_short)

            gw = gate_weights[i, 0]
            mw = moe_weights[i]

            sig = HydraSignal(
                direction=direction,
                confidence=confidence,
                uncertainty=float(uncertainty[i]),
                probs_long=[float(probs_5m[i, 2]), float(probs_15m[i, 2]), float(p_long)],
                probs_short=[float(probs_5m[i, 0]), float(probs_15m[i, 0]), float(p_short)],
                regime_weights={
                    "TREND": float(mw[0]),
                    "RANGE": float(mw[1]),
                    "VOL_EXP": float(mw[2]),
                    "NEWS": float(mw[3]),
                },
                gate_weights={
                    "TFT": float(gw[0]),
                    "LSTM": float(gw[1]),
                    "CNN": float(gw[2]),
                    "MoE": float(gw[3]),
                },
                timestamp_ms=timestamps[i] if timestamps else 0
            )
            signals.append(sig)

        return signals

    def reset(self):
        """Reset sequence buffers (e.g., between trading sessions)."""
        self._cont_buffer.fill(0)
        self._cat_buffer.fill(0)
        self._buffer_idx = 0
        self._is_primed = False
        self._last_raw_probs = None
