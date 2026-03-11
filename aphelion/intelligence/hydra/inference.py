"""
APHELION HYDRA Inference Engine (Ensemble Edition)

Supports live/batch inference, probability smoothing, and adversarial
robustness gating that automatically downweights confidence near a learned
adversarial boundary.
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

from aphelion.intelligence.hydra.dataset import (
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
    SESSION_MAP,
    DAY_MAP,
)
from aphelion.intelligence.hydra.ensemble import HydraGate, EnsembleConfig
from aphelion.intelligence.hydra.adversarial import (
    AdversarialFeaturePerturbationDetector,
    AdversarialConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for HYDRA inference."""

    checkpoint_path: str = ""
    device: str = ""
    smoothing_alpha: float = 0.3
    history_len: int = 64

    # Adversarial robustness gate
    adversarial_enabled: bool = True
    adversarial_epsilon: float = 0.20
    adversarial_step_size: float = 0.05
    adversarial_steps: int = 4
    adversarial_risk_penalty: float = 0.65


@dataclass
class HydraSignal:
    """Output prediction from the HYDRA Ensemble."""

    direction: int
    confidence: float
    uncertainty: float

    probs_long: list[float]
    probs_short: list[float]

    regime_weights: dict[str, float]
    gate_weights: dict[str, float]

    # Robustness metadata
    raw_confidence: float = 0.0
    robust_confidence: float = 0.0
    adversarial_risk: float = 0.0
    adversarial_boundary: float = 0.0
    adversarial_confidence_drop: float = 0.0

    timestamp_ms: int = 0

    @property
    def is_actionable(self) -> bool:
        return self.direction != 0 and self.confidence > 0.55

    @property
    def horizon_agreement(self) -> float:
        dirs = []
        for pl, ps in zip(self.probs_long, self.probs_short):
            p_flat = 1.0 - pl - ps
            if pl > ps and pl > p_flat:
                dirs.append(1)
            elif ps > pl and ps > p_flat:
                dirs.append(-1)
            else:
                dirs.append(0)

        from collections import Counter

        most_common = Counter(dirs).most_common(1)[0][1]
        return most_common / 3.0


class HydraInference:
    """Live and backtest inference wrapper for HYDRA Gate."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[InferenceConfig] = None,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for HYDRA inference.")

        self._runtime_cfg = config or InferenceConfig()
        if checkpoint_path:
            self._runtime_cfg.checkpoint_path = checkpoint_path
        if device:
            self._runtime_cfg.device = device

        self._device = torch.device(
            self._runtime_cfg.device
            if self._runtime_cfg.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._history_len = int(self._runtime_cfg.history_len)
        self._cont_feature_names: list[str] = list(CONTINUOUS_FEATURES)
        self._cat_feature_names: list[str] = list(CATEGORICAL_FEATURES)

        self._n_cont = len(self._cont_feature_names)
        self._n_cat = len(self._cat_feature_names)

        self._cont_buffer = np.zeros((self._history_len, self._n_cont), dtype=np.float32)
        self._cat_buffer = np.zeros((self._history_len, self._n_cat), dtype=np.int64)
        self._buffer_idx = 0
        self._is_primed = False

        self._model: Optional[HydraGate] = None
        self._config: Optional[EnsembleConfig] = None

        self._last_raw_probs = None
        self._smoothing_alpha = float(self._runtime_cfg.smoothing_alpha)

        self._adversarial_detector = AdversarialFeaturePerturbationDetector(
            AdversarialConfig(
                enabled=bool(self._runtime_cfg.adversarial_enabled),
                epsilon=float(self._runtime_cfg.adversarial_epsilon),
                step_size=float(self._runtime_cfg.adversarial_step_size),
                steps=int(self._runtime_cfg.adversarial_steps),
                risk_penalty=float(self._runtime_cfg.adversarial_risk_penalty),
            )
        )

        if self._runtime_cfg.checkpoint_path:
            self.load_checkpoint(self._runtime_cfg.checkpoint_path)

    def _reinit_buffers(self) -> None:
        self._cont_buffer = np.zeros((self._history_len, self._n_cont), dtype=np.float32)
        self._cat_buffer = np.zeros((self._history_len, self._n_cat), dtype=np.int64)
        self._buffer_idx = 0
        self._is_primed = False
        self._last_raw_probs = None

    def load_checkpoint(self, path: str) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        self._config = ckpt.get("ensemble_config", EnsembleConfig())

        # Keep backward compatibility with checkpoints trained on older feature counts.
        tft_cfg = self._config.tft_config
        expected_n_cont = int(getattr(tft_cfg, "n_continuous", len(CONTINUOUS_FEATURES)))
        expected_n_cat = int(getattr(tft_cfg, "n_categorical", len(CATEGORICAL_FEATURES)))
        expected_lookback = int(getattr(tft_cfg, "lookback", self._history_len))

        self._cont_feature_names = list(CONTINUOUS_FEATURES[:expected_n_cont])
        if expected_n_cont > len(self._cont_feature_names):
            self._cont_feature_names.extend(
                [f"_pad_cont_{i}" for i in range(expected_n_cont - len(self._cont_feature_names))]
            )

        self._cat_feature_names = list(CATEGORICAL_FEATURES[:expected_n_cat])
        if expected_n_cat > len(self._cat_feature_names):
            self._cat_feature_names.extend(
                [f"_pad_cat_{i}" for i in range(expected_n_cat - len(self._cat_feature_names))]
            )

        self._history_len = expected_lookback
        self._n_cont = len(self._cont_feature_names)
        self._n_cat = len(self._cat_feature_names)
        self._reinit_buffers()

        self._model = HydraGate(self._config).to(self._device)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()
        logger.info("HYDRA Ensemble loaded from %s on %s", path, self._device)

    def _encode_cat(self, name: str, value) -> int:
        if name.startswith("_pad_cat_"):
            return 0
        if name == "session":
            if isinstance(value, str):
                return int(SESSION_MAP.get(value, 4))
            try:
                return int(value)
            except Exception:
                return 4
        if name == "day_of_week":
            if isinstance(value, str):
                return int(DAY_MAP.get(value, 0))
            try:
                return int(value)
            except Exception:
                return 0
        try:
            return int(value)
        except Exception:
            return 0

    def process_bar(self, features: dict) -> Optional[HydraSignal]:
        cont_vals = [float(features.get(f, 0.0)) for f in self._cont_feature_names]
        cat_vals = [self._encode_cat(f, features.get(f, 0)) for f in self._cat_feature_names]

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

        cont_t = torch.tensor(self._cont_buffer, dtype=torch.float32).unsqueeze(0).to(self._device)
        cat_t = torch.tensor(self._cat_buffer, dtype=torch.long).unsqueeze(0).to(self._device)

        return self._predict_single(cont_t, cat_t, int(features.get("timestamp_ms", 0)))

    def predict_batch(self, cont_batch: np.ndarray, cat_batch: np.ndarray) -> list[HydraSignal]:
        if self._model is None:
            raise RuntimeError("Model not loaded yet.")

        cont_t = torch.tensor(cont_batch, dtype=torch.float32).to(self._device)
        cat_t = torch.tensor(cat_batch, dtype=torch.long).to(self._device)

        with torch.no_grad():
            outputs = self._model(cont_t, cat_t)

        return self._format_batch_outputs(outputs)

    def _predict_single(self, cont_t: torch.Tensor, cat_t: torch.Tensor, ts_ms: int) -> HydraSignal:
        if self._model is None:
            raise RuntimeError("Model not loaded yet.")

        with torch.no_grad():
            outputs = self._model(cont_t, cat_t)
        signal = self._format_batch_outputs(outputs, [ts_ms])[0]

        # Adversarial robustness gate: downweight confidence near vulnerable boundaries.
        assessment = self._adversarial_detector.assess(self._model, cont_t, cat_t, base_outputs=outputs)
        if assessment.raw_confidence > 0:
            raw_conf = float(signal.confidence)
            ratio = assessment.robust_confidence / max(assessment.raw_confidence, 1e-9)
            robust_conf = max(0.0, min(1.0, raw_conf * ratio))

            signal.raw_confidence = raw_conf
            signal.robust_confidence = robust_conf
            signal.confidence = robust_conf
            signal.adversarial_risk = float(assessment.risk_score)
            signal.adversarial_boundary = float(assessment.boundary_distance)
            signal.adversarial_confidence_drop = float(assessment.confidence_drop)
        else:
            signal.raw_confidence = float(signal.confidence)
            signal.robust_confidence = float(signal.confidence)

        return signal

    def _format_batch_outputs(self, outputs: dict, timestamps: Optional[list[int]] = None) -> list[HydraSignal]:
        signals = []
        batch_size = outputs["probs_1h"].shape[0]

        probs_1h = outputs["probs_1h"].detach().cpu().float().numpy()
        probs_15m = outputs["probs_15m"].detach().cpu().float().numpy()
        probs_5m = outputs["probs_5m"].detach().cpu().float().numpy()
        uncertainty = outputs["uncertainty"].detach().cpu().float().numpy().flatten()

        gate_weights = outputs.get("gate_attention_weights", torch.zeros(batch_size, 1, 6)).detach().cpu().float().numpy()
        moe_weights = outputs.get("moe_routing_weights", torch.zeros(batch_size, 4)).detach().cpu().float().numpy()

        for i in range(batch_size):
            p1 = probs_1h[i]

            if self._last_raw_probs is None:
                self._last_raw_probs = p1
            else:
                self._last_raw_probs = (
                    self._smoothing_alpha * p1 + (1.0 - self._smoothing_alpha) * self._last_raw_probs
                )
                p1 = self._last_raw_probs

            p_short, p_flat, p_long = float(p1[0]), float(p1[1]), float(p1[2])
            direction = 0
            confidence = p_flat

            if p_long > p_flat and p_long > p_short:
                direction = 1
                confidence = p_long
            elif p_short > p_flat and p_short > p_long:
                direction = -1
                confidence = p_short

            gw = gate_weights[i, 0]
            mw = moe_weights[i]

            sig = HydraSignal(
                direction=direction,
                confidence=float(confidence),
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
                    "TFT": float(gw[0]) if len(gw) > 0 else 0.0,
                    "LSTM": float(gw[1]) if len(gw) > 1 else 0.0,
                    "CNN": float(gw[2]) if len(gw) > 2 else 0.0,
                    "MoE": float(gw[3]) if len(gw) > 3 else 0.0,
                    "TCN": float(gw[4]) if len(gw) > 4 else 0.0,
                    "Transformer": float(gw[5]) if len(gw) > 5 else 0.0,
                },
                raw_confidence=float(confidence),
                robust_confidence=float(confidence),
                timestamp_ms=timestamps[i] if timestamps else 0,
            )
            signals.append(sig)

        return signals

    def reset(self):
        self._reinit_buffers()
