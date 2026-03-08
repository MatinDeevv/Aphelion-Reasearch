"""
APHELION HYDRA Trainer (Ensemble Edition)
Trains the Full Ensemble (Gate + TFT + LSTM + CNN + MoE) end-to-end.
Implements Auxiliary Losses to force sub-models to learn useful independent representations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from contextlib import nullcontext

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from aphelion.intelligence.hydra.ensemble import EnsembleConfig, HydraGate

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Training configuration for the Full Ensemble."""
    learning_rate: float = 5e-4    # Lower LR for giant ensemble
    weight_decay: float = 1e-4
    max_epochs: int = 150
    gradient_clip_norm: float = 1.0

    warmup_epochs: int = 5
    cosine_t_max: int = 30

    # Loss weights
    # Main ensemble outputs
    classification_loss_weight: float = 1.0
    quantile_loss_weight: float = 0.3
    
    # Auxiliary losses to guide sub-models independently
    aux_loss_weight: float = 0.2

    # Focal loss configs
    focal_gamma: float = 2.0
    focal_alpha: list[float] = field(default_factory=lambda: [1.5, 0.5, 1.5])

    patience: int = 20
    min_delta: float = 0.01

    use_amp: bool = True
    checkpoint_dir: str = "models/hydra"
    save_every_n_epochs: int = 5

    ensemble_config: EnsembleConfig = field(default_factory=EnsembleConfig)


if HAS_TORCH:
    class FocalLoss(nn.Module):
        def __init__(self, gamma: float = 2.0, alpha: Optional[list[float]] = None):
            super().__init__()
            self.gamma = gamma
            if alpha is not None:
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = None

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            ce_loss = F.cross_entropy(logits, targets, reduction="none")
            pt = torch.exp(-ce_loss)
            focal = ((1 - pt) ** self.gamma) * ce_loss

            if self.alpha is not None:
                alpha = self.alpha.to(logits.device)
                alpha_t = alpha[targets]
                focal = alpha_t * focal

            return focal.mean()

    class QuantileLoss(nn.Module):
        def __init__(self, quantiles: list[float]):
            super().__init__()
            self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))

        def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            losses = []
            for i, q in enumerate(self.quantiles):
                errors = targets[:, i] - predictions[:, i]
                losses.append(torch.max(q * errors, (q - 1) * errors))
            return torch.stack(losses, dim=1).mean()

    class HydraTrainer:
        """Trainer for the full HYDRA Ensemble."""

        def __init__(
            self,
            config: Optional[TrainerConfig] = None,
            device: Optional[str] = None,
        ):
            self._config = config or TrainerConfig()
            cfg = self._config

            if device:
                self._device = torch.device(device)
            else:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self._model = HydraGate(cfg.ensemble_config).to(self._device)
            logger.info(
                "HYDRA Full Ensemble initialized: %s parameters on %s",
                f"{self._model.count_parameters():,}", self._device,
            )

            self._optimizer = torch.optim.AdamW(
                self._model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )

            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self._optimizer, T_0=cfg.cosine_t_max, T_mult=2,
            )

            self._focal = FocalLoss(cfg.focal_gamma, cfg.focal_alpha)
            self._quantile = QuantileLoss(cfg.ensemble_config.tft_config.quantile_targets)
            amp_enabled = cfg.use_amp and self._device.type == "cuda"
            self._scaler = self._create_grad_scaler(enabled=amp_enabled)

            self._best_val_sharpe = -float("inf")
            self._best_val_loss = float("inf")
            self._epochs_no_improve = 0
            self._epoch = 0

            self._train_history: list[dict] = []
            self._val_history: list[dict] = []

        @property
        def model(self) -> HydraGate:
            return self._model

        @staticmethod
        def _create_grad_scaler(enabled: bool):
            try:
                return torch.amp.GradScaler("cuda", enabled=enabled)
            except (AttributeError, TypeError):
                return torch.cuda.amp.GradScaler(enabled=enabled)

        def _autocast_context(self):
            if not self._config.use_amp or self._device.type != "cuda":
                return nullcontext()

            try:
                return torch.amp.autocast("cuda", enabled=True)
            except (AttributeError, TypeError):
                return torch.cuda.amp.autocast(enabled=True)

        def train(self, train_loader, val_loader) -> dict:
            cfg = self._config
            logger.info("Starting HYDRA Ensemble training: %d epochs max", cfg.max_epochs)

            for epoch in range(cfg.max_epochs):
                self._epoch = epoch
                t0 = time.time()

                train_metrics = self._train_epoch(train_loader)
                self._train_history.append(train_metrics)

                val_metrics = self._validate(val_loader)
                self._val_history.append(val_metrics)

                self._scheduler.step()
                lr = self._optimizer.param_groups[0]["lr"]

                elapsed = time.time() - t0
                logger.info(
                    "Epoch %d/%d — loss=%.4f/%.4f acc=%.1f%% sharpe=%.2f lr=%.6f (%.1fs)",
                    epoch + 1, cfg.max_epochs,
                    train_metrics["loss"], val_metrics["loss"],
                    val_metrics["accuracy"] * 100,
                    val_metrics.get("sharpe_proxy", 0.0),
                    lr, elapsed,
                )

                improved = self._check_improvement(val_metrics)

                if (epoch + 1) % cfg.save_every_n_epochs == 0:
                    self._save_checkpoint("latest")

                if self._epochs_no_improve >= cfg.patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch + 1, cfg.patience,
                    )
                    break

            return {
                "total_epochs": self._epoch + 1,
                "best_val_sharpe": self._best_val_sharpe,
                "best_val_loss": self._best_val_loss,
                "final_train_loss": self._train_history[-1]["loss"],
                "final_val_loss": self._val_history[-1]["loss"],
                "model_params": self._model.count_parameters(),
            }

        def _compute_aux_losses(self, aux_logits_list: list[torch.Tensor], targets: list[torch.Tensor]) -> torch.Tensor:
            """Compute loss for auxiliary heads [y5m, y15m, y1h]"""
            loss = 0.0
            for logits, target in zip(aux_logits_list, targets):
                loss += self._focal(logits, target)
            return loss / 3.0

        def _train_epoch(self, loader) -> dict:
            self._model.train()
            cfg = self._config

            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            n_batches = 0

            for batch in loader:
                cont, cat, y5m, y15m, y1h, raw_ret = [b.to(self._device) for b in batch]
                targets = [y5m, y15m, y1h]

                self._optimizer.zero_grad(set_to_none=True)

                with self._autocast_context():
                    outputs = self._model(cont, cat)

                    # 1. Main Ensemble Classification Loss (Focal)
                    loss_cls = (
                        self._focal(outputs["logits_5m"], y5m) +
                        self._focal(outputs["logits_15m"], y15m) +
                        self._focal(outputs["logits_1h"], y1h)
                    ) / 3.0

                    # 2. Main Ensemble Quantile Loss
                    loss_q = (
                        self._quantile(outputs["quantiles_5m"], raw_ret[:, 0:1].expand(-1, 3)) +
                        self._quantile(outputs["quantiles_15m"], raw_ret[:, 1:2].expand(-1, 3)) +
                        self._quantile(outputs["quantiles_1h"], raw_ret[:, 2:3].expand(-1, 3))
                    ) / 3.0

                    # 3. Auxiliary Sub-Model Losses (Forces sub-models to be predictive independent of gate)
                    loss_tft_aux = self._compute_aux_losses(outputs["tft_logits"], targets)
                    loss_lstm_aux = self._compute_aux_losses(outputs["lstm_logits"], targets)
                    loss_cnn_aux = self._compute_aux_losses(outputs["cnn_logits"], targets)
                    loss_moe_aux = self._compute_aux_losses(outputs["moe_logits"], targets)

                    total_aux_loss = (loss_tft_aux + loss_lstm_aux + loss_cnn_aux + loss_moe_aux) / 4.0

                    # Combined Objective Function
                    loss = (
                        (cfg.classification_loss_weight * loss_cls) +
                        (cfg.quantile_loss_weight * loss_q) +
                        (cfg.aux_loss_weight * total_aux_loss)
                    )

                self._scaler.scale(loss).backward()
                self._scaler.unscale_(self._optimizer)
                nn.utils.clip_grad_norm_(self._model.parameters(), cfg.gradient_clip_norm)
                self._scaler.step(self._optimizer)
                self._scaler.update()

                total_loss += loss.item()
                preds = outputs["logits_1h"].argmax(dim=-1)
                total_correct += (preds == y1h).sum().item()
                total_samples += y1h.shape[0]
                n_batches += 1

            return {
                "loss": total_loss / max(n_batches, 1),
                "accuracy": total_correct / max(total_samples, 1),
            }

        @torch.no_grad()
        def _validate(self, loader) -> dict:
            self._model.eval()
            cfg = self._config

            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            n_batches = 0
            all_preds = []

            for batch in loader:
                cont, cat, y5m, y15m, y1h, raw_ret = [b.to(self._device) for b in batch]

                with self._autocast_context():
                    outputs = self._model(cont, cat)

                    loss_cls = (
                        self._focal(outputs["logits_5m"], y5m) +
                        self._focal(outputs["logits_15m"], y15m) +
                        self._focal(outputs["logits_1h"], y1h)
                    ) / 3.0

                    loss_q = (
                        self._quantile(outputs["quantiles_5m"], raw_ret[:, 0:1].expand(-1, 3)) +
                        self._quantile(outputs["quantiles_15m"], raw_ret[:, 1:2].expand(-1, 3)) +
                        self._quantile(outputs["quantiles_1h"], raw_ret[:, 2:3].expand(-1, 3))
                    ) / 3.0

                    loss = cfg.classification_loss_weight * loss_cls + cfg.quantile_loss_weight * loss_q

                total_loss += loss.item()
                preds = outputs["logits_1h"].argmax(dim=-1)
                total_correct += (preds == y1h).sum().item()
                total_samples += y1h.shape[0]

                pred_dir = preds.float() - 1.0  # SHORT=-1, FLAT=0, LONG=1
                actual_ret = raw_ret[:, 2]      # 1h returns
                strategy_ret = pred_dir * actual_ret
                all_preds.extend(strategy_ret.cpu().numpy().tolist())

                n_batches += 1

            all_preds_arr = np.array(all_preds)
            sharpe_proxy = 0.0
            if len(all_preds_arr) > 1 and np.std(all_preds_arr) > 0:
                sharpe_proxy = float(np.mean(all_preds_arr) / np.std(all_preds_arr) * np.sqrt(252 * 24))

            return {
                "loss": total_loss / max(n_batches, 1),
                "accuracy": total_correct / max(total_samples, 1),
                "sharpe_proxy": sharpe_proxy,
            }

        def _check_improvement(self, val_metrics: dict) -> bool:
            cfg = self._config
            sharpe = val_metrics.get("sharpe_proxy", 0.0)
            loss = val_metrics["loss"]
            improved = False

            if sharpe > self._best_val_sharpe + cfg.min_delta:
                self._best_val_sharpe = sharpe
                self._save_checkpoint("best_sharpe")
                self._epochs_no_improve = 0
                improved = True

            if loss < self._best_val_loss - cfg.min_delta:
                self._best_val_loss = loss
                self._save_checkpoint("best_loss")
                if not improved:
                    self._epochs_no_improve = 0
                improved = True

            if not improved:
                self._epochs_no_improve += 1

            return improved

        def _save_checkpoint(self, tag: str) -> Path:
            ckpt_dir = Path(self._config.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            path = ckpt_dir / f"hydra_ensemble_{tag}.pt"

            torch.save({
                "epoch": self._epoch,
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "scheduler_state_dict": self._scheduler.state_dict(),
                "scaler_state_dict": self._scaler.state_dict(),
                "best_val_sharpe": self._best_val_sharpe,
                "best_val_loss": self._best_val_loss,
                "ensemble_config": self._config.ensemble_config,
                "train_history": self._train_history,
                "val_history": self._val_history,
            }, path)

            logger.info("Checkpoint saved: %s (epoch %d)", path, self._epoch + 1)
            return path

        def load_checkpoint(self, path: str) -> None:
            ckpt = torch.load(path, map_location=self._device, weights_only=False)
            self._model.load_state_dict(ckpt["model_state_dict"])
            self._optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self._scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self._scaler.load_state_dict(ckpt["scaler_state_dict"])
            self._best_val_sharpe = ckpt.get("best_val_sharpe", -float("inf"))
            self._best_val_loss = ckpt.get("best_val_loss", float("inf"))
            self._epoch = ckpt.get("epoch", 0)
            self._train_history = ckpt.get("train_history", [])
            self._val_history = ckpt.get("val_history", [])
            logger.info("Checkpoint loaded: %s (epoch %d)", path, self._epoch)
