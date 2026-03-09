"""
APHELION HYDRA — Neural Intelligence Core (Phase 4: TFT v1)
Temporal Fusion Transformer for multi-horizon XAU/USD direction prediction.
"""

try:
    from aphelion.intelligence.hydra.tft import (
        TemporalFusionTransformer,
        TFTConfig,
    )
    from aphelion.intelligence.hydra.lstm import HydraLSTM, LSTMConfig
    from aphelion.intelligence.hydra.cnn import HydraCNN, CNNConfig
    from aphelion.intelligence.hydra.moe import HydraMoE, MoEConfig
    from aphelion.intelligence.hydra.ensemble import HydraGate, EnsembleConfig
    
    from aphelion.intelligence.hydra.dataset import (
        HydraDataset,
        create_dataloaders,
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

    HAS_TORCH = True

    __all__ = [
        "TemporalFusionTransformer",
        "TFTConfig",
        "HydraLSTM",
        "LSTMConfig",
        "HydraCNN",
        "CNNConfig",
        "HydraMoE",
        "MoEConfig",
        "HydraGate",
        "EnsembleConfig",
        "HydraDataset",
        "create_dataloaders",
        "HydraTrainer",
        "TrainerConfig",
        "HydraInference",
        "InferenceConfig",
        "HydraSignal",
        "HydraStrategy",
        "StrategyConfig",
        "HAS_TORCH",
    ]
except ModuleNotFoundError:
    # PyTorch not installed — module not available
    HAS_TORCH = False
    __all__ = ["HAS_TORCH"]
