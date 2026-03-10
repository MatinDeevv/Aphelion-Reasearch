"""
APHELION HYDRA — Neural Intelligence Core (Phase 4+7: Full Ensemble)
TFT + LSTM + CNN + MoE ensemble for multi-horizon XAU/USD direction prediction.

Export policy:
  - If PyTorch is installed: all model, dataset, trainer, inference, and strategy
    classes are exported.  ``HAS_TORCH`` is ``True``.
  - If PyTorch is *not* installed: only ``HAS_TORCH = False`` is exported and a
    clear warning is emitted so callers know *why* the module is inert.
  - Any unexpected import error (broken install, version mismatch) is logged at
    WARNING level rather than silently swallowed, so developers can diagnose it.
"""

from __future__ import annotations

import logging as _logging

_logger = _logging.getLogger(__name__)

# ── Attempt to import the full HYDRA API ──────────────────────────────────

try:
    from aphelion.intelligence.hydra.tft import (
        TemporalFusionTransformer,
        TFTConfig,
    )
    from aphelion.intelligence.hydra.lstm import HydraLSTM, LSTMConfig
    from aphelion.intelligence.hydra.cnn import HydraCNN, CNNConfig
    from aphelion.intelligence.hydra.moe import HydraMoE, MoEConfig
    from aphelion.intelligence.hydra.tcn import HydraTCN, TCNConfig
    from aphelion.intelligence.hydra.transformer import HydraTransformer, TransformerConfig
    from aphelion.intelligence.hydra.ensemble import HydraGate, EnsembleConfig

    from aphelion.intelligence.hydra.dataset import (
        HydraDataset,
        DatasetConfig,
        create_dataloaders,
        build_dataset_from_feature_dicts,
        CONTINUOUS_FEATURES,
        CATEGORICAL_FEATURES,
    )
    from aphelion.intelligence.hydra.trainer import (
        HydraTrainer,
        TrainerConfig,
    )
    from aphelion.intelligence.hydra.inference import (
        HydraInference,
        InferenceConfig,
        HydraSignal,
    )
    from aphelion.intelligence.hydra.strategy import (
        HydraStrategy,
        StrategyConfig,
    )
    from aphelion.intelligence.hydra.calibration import (
        IsotonicCalibrator,
        DisagreementDetector,
        DynamicEnsembleWeights,
        CalibrationResult,
    )
    from aphelion.intelligence.hydra.xgb_model import (
        HydraXGBoost,
        HydraRandomForest,
        HydraTreeEnsemble,
        TreeModelConfig,
        TreePrediction,
    )

    HAS_TORCH = True

    __all__ = [
        # Models
        "TemporalFusionTransformer", "TFTConfig",
        "HydraLSTM", "LSTMConfig",
        "HydraCNN", "CNNConfig",
        "HydraMoE", "MoEConfig",
        "HydraTCN", "TCNConfig",
        "HydraTransformer", "TransformerConfig",
        "HydraGate", "EnsembleConfig",
        # Tree models
        "HydraXGBoost", "HydraRandomForest", "HydraTreeEnsemble",
        "TreeModelConfig", "TreePrediction",
        # Calibration & Disagreement
        "IsotonicCalibrator", "DisagreementDetector",
        "DynamicEnsembleWeights", "CalibrationResult",
        # Data
        "HydraDataset", "DatasetConfig",
        "create_dataloaders", "build_dataset_from_feature_dicts",
        "CONTINUOUS_FEATURES", "CATEGORICAL_FEATURES",
        # Training
        "HydraTrainer", "TrainerConfig",
        # Inference & Strategy
        "HydraInference", "InferenceConfig", "HydraSignal",
        "HydraStrategy", "StrategyConfig",
        # Availability flag
        "HAS_TORCH",
    ]

except ModuleNotFoundError:
    # PyTorch is not installed — this is the expected non-ML path.
    HAS_TORCH = False
    __all__ = ["HAS_TORCH"]
    _logger.debug(
        "PyTorch not installed — HYDRA module inert.  "
        "Install with: pip install -e '.[ml]'"
    )

except Exception as _exc:
    # Unexpected error (e.g. version mismatch, broken C extension).
    # Log loudly so developers can diagnose it instead of getting a
    # mysterious empty __all__.
    HAS_TORCH = False
    __all__ = ["HAS_TORCH"]
    _logger.warning(
        "HYDRA import failed unexpectedly: %s — module will be inert.",
        _exc,
        exc_info=True,
    )
