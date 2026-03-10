"""
APHELION HYDRA Trainer (SUPER INSANE Edition)
Trains the Full 6-model Ensemble end-to-end with:
- Label Smoothing Focal Loss
- Mixup Data Augmentation
- Gradient Accumulation (effective 4x batch size)
- All 6 sub-model auxiliary losses (TFT, LSTM, CNN, MoE, TCN, Transformer)
- MoE Load Balancing Loss
- Diversity Loss (encourage sub-model disagreement)
- Linear Warmup + Cosine Annealing with Warm Restarts
- Stochastic Weight Averaging (SWA) in final phase
- SUPER LOGGING: tqdm progress bars, GPU stats, per-batch metrics
"""

from __future__ import annotations

import logging
import time
import math
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

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from aphelion.intelligence.hydra.ensemble import EnsembleConfig, HydraGate

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Training configuration — SUPER INSANE defaults."""
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 200
    gradient_clip_norm: float = 1.0

    warmup_epochs: int = 10
    cosine_t_max: int = 40

    # Loss weights
    classification_loss_weight: float = 1.0
    quantile_loss_weight: float = 0.3
    aux_loss_weight: float = 0.15       # Auxiliary sub-model losses
    moe_balance_weight: float = 0.01    # MoE load balance loss
    diversity_loss_weight: float = 0.05  # Encourages sub-model disagreement

    # Focal loss configs
    focal_gamma: float = 2.0
    focal_alpha: list[float] = field(default_factory=lambda: [1.5, 0.5, 1.5])
    label_smoothing: float = 0.1  # Label smoothing for noisy market data

    # Mixup augmentation
    mixup_alpha: float = 0.2  # Beta distribution parameter (0 = disabled)

    # Gradient accumulation
    gradient_accumulation_steps: int = 4

    patience: int = 25
    min_delta: float = 0.005

    use_amp: bool = True
    checkpoint_dir: str = "models/hydra"
    save_every_n_epochs: int = 5

    # SWA: Stochastic Weight Averaging
    swa_start_epoch: int = 100  # Start SWA after this epoch
    swa_lr: float = 1e-5

    ensemble_config: EnsembleConfig = field(default_factory=EnsembleConfig)


if HAS_TORCH:
    class LabelSmoothingFocalLoss(nn.Module):
        """Focal loss with label smoothing — ideal for noisy market data."""

        def __init__(self, gamma: float = 2.0, alpha: Optional[list[float]] = None,
                     smoothing: float = 0.1, n_classes: int = 3):
            super().__init__()
            self.gamma = gamma
            self.smoothing = smoothing
            self.n_classes = n_classes
            if alpha is not None:
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = None

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            # Apply label smoothing
            with torch.no_grad():
                smooth_targets = torch.full_like(logits, self.smoothing / (self.n_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

            log_probs = F.log_softmax(logits, dim=-1)
            ce_loss = -(smooth_targets * log_probs).sum(dim=-1)

            # Focal modulation
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
        """SUPER INSANE Trainer for the full HYDRA Ensemble."""

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
                "HYDRA SUPER INSANE Ensemble: %s parameters on %s",
                f"{self._model.count_parameters():,}", self._device,
            )

            # OPTIMIZED: Enable cudnn benchmark for consistent input sizes
            if self._device.type == "cuda":
                torch.backends.cudnn.benchmark = True

            # OPTIMIZED: Detect bfloat16 support (A100/H100)
            self._use_bf16 = False
            if cfg.use_amp and self._device.type == "cuda":
                try:
                    if torch.cuda.is_bf16_supported():
                        self._use_bf16 = True
                        logger.info("BFloat16 supported — using BF16 (faster than FP16, no scaler needed)")
                except AttributeError:
                    pass

            # OPTIMIZED: torch.compile() for 20-50% speedup on PyTorch 2.x
            # NOTE: mode="max-autotune" avoids CUDAGraphs which crash with
            # dynamic MoE routing (.item() calls) and multi-model ensembles.
            self._compiled = False
            if hasattr(torch, 'compile') and self._device.type == "cuda":
                try:
                    self._model = torch.compile(self._model, mode="max-autotune")
                    self._compiled = True
                    logger.info("torch.compile() enabled (max-autotune mode)")
                except Exception as e:
                    logger.warning("torch.compile() failed: %s — falling back to eager", e)

            # Separate LR groups: sub-models vs gate layers
            sub_model_params = []
            gate_params = []
            for name, param in self._model.named_parameters():
                if any(m in name for m in ('tft.', 'lstm.', 'cnn.', 'moe.', 'tcn.',
                                            'transformer.')):
                    if not any(p in name for p in ('_proj', '_adapter')):
                        sub_model_params.append(param)
                        continue
                gate_params.append(param)

            self._optimizer = torch.optim.AdamW([
                {"params": sub_model_params, "lr": cfg.learning_rate * 0.5},  # Sub-models: lower LR
                {"params": gate_params, "lr": cfg.learning_rate},             # Gate: full LR
            ], weight_decay=cfg.weight_decay)

            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self._optimizer, T_0=cfg.cosine_t_max, T_mult=2,
            )

            self._focal = LabelSmoothingFocalLoss(
                cfg.focal_gamma, cfg.focal_alpha, cfg.label_smoothing,
            )
            self._quantile = QuantileLoss(cfg.ensemble_config.tft_config.quantile_targets)
            amp_enabled = cfg.use_amp and self._device.type == "cuda"
            # BF16 doesn't need GradScaler
            scaler_enabled = amp_enabled and not self._use_bf16
            self._scaler = self._create_grad_scaler(enabled=scaler_enabled)

            # SWA model
            # Store initial LRs for warmup scheduling
            for pg in self._optimizer.param_groups:
                pg['initial_lr'] = pg['lr']

            self._swa_model = None
            self._swa_scheduler = None
            if cfg.swa_start_epoch < cfg.max_epochs:
                try:
                    from torch.optim.swa_utils import AveragedModel, SWALR
                    self._swa_model = AveragedModel(self._model)
                    self._swa_scheduler = SWALR(self._optimizer, swa_lr=cfg.swa_lr)
                    logger.info("SWA enabled (starts epoch %d)", cfg.swa_start_epoch)
                except ImportError:
                    logger.warning("SWA unavailable — torch.optim.swa_utils not found")

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
            # OPTIMIZED: Use bfloat16 on A100/H100 for faster math
            dtype = torch.bfloat16 if self._use_bf16 else torch.float16
            try:
                return torch.amp.autocast("cuda", enabled=True, dtype=dtype)
            except (AttributeError, TypeError):
                return torch.cuda.amp.autocast(enabled=True)

        def _get_warmup_factor(self) -> float:
            """Linear warmup from 0.1 to 1.0 over warmup_epochs."""
            if self._epoch >= self._config.warmup_epochs:
                return 1.0
            return 0.1 + 0.9 * (self._epoch / max(self._config.warmup_epochs, 1))

        @staticmethod
        def _mixup_data(x_cont, x_cat, y5m, y15m, y1h, raw_ret, alpha=0.2):
            """Mixup augmentation — interpolates continuous features and soft labels."""
            if alpha <= 0:
                return x_cont, x_cat, y5m, y15m, y1h, raw_ret, None
            lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
            lam = max(lam, 1 - lam)  # Ensure lam >= 0.5
            batch_size = x_cont.size(0)
            index = torch.randperm(batch_size, device=x_cont.device)

            mixed_cont = lam * x_cont + (1 - lam) * x_cont[index]
            # Don't mix categorical — keep original
            mixed_ret = lam * raw_ret + (1 - lam) * raw_ret[index]

            return mixed_cont, x_cat, y5m, y15m, y1h, mixed_ret, (index, lam)

        def _gpu_stats(self) -> str:
            """Get GPU memory usage string."""
            if self._device.type != "cuda":
                return ""
            try:
                allocated = torch.cuda.memory_allocated(self._device) / 1024**3
                reserved = torch.cuda.memory_reserved(self._device) / 1024**3
                return f"GPU: {allocated:.1f}/{reserved:.1f} GB"
            except Exception:
                return ""

        def train(self, train_loader, val_loader) -> dict:
            cfg = self._config
            total_batches = len(train_loader)
            total_params = self._model.count_parameters()

            # Print training banner
            print(f"\n{'━' * 70}")
            print(f"  🔥 HYDRA SUPER INSANE TRAINING — {total_params:,} parameters")
            print(f"  Epochs: {cfg.max_epochs} | Batch size: {train_loader.batch_size}")
            print(f"  Batches/epoch: {total_batches} | Grad accum: {cfg.gradient_accumulation_steps}x")
            print(f"  Effective batch: {train_loader.batch_size * cfg.gradient_accumulation_steps}")
            print(f"  LR: {cfg.learning_rate} | Warmup: {cfg.warmup_epochs} epochs")
            print(f"  Losses: Focal+Quantile+6xAux+MoE_LB+Diversity")
            print(f"  AMP: {cfg.use_amp} ({'BF16' if self._use_bf16 else 'FP16'}) | "
                  f"Device: {self._device} | Compiled: {self._compiled}")
            print(f"  {self._gpu_stats()}")
            print(f"{'━' * 70}\n")

            training_start = time.time()

            for epoch in range(cfg.max_epochs):
                self._epoch = epoch
                epoch_start = time.time()

                # Apply warmup LR scaling
                if epoch < cfg.warmup_epochs:
                    warmup_factor = self._get_warmup_factor()
                    for pg in self._optimizer.param_groups:
                        pg['lr'] = pg.get('initial_lr', cfg.learning_rate) * warmup_factor

                train_metrics = self._train_epoch(train_loader)
                self._train_history.append(train_metrics)

                val_metrics = self._validate(val_loader)
                self._val_history.append(val_metrics)

                # Scheduler step
                use_swa = (self._swa_model is not None and epoch >= cfg.swa_start_epoch)
                if use_swa:
                    self._swa_model.update_parameters(self._model)
                    self._swa_scheduler.step()
                else:
                    self._scheduler.step()

                lr = self._optimizer.param_groups[0]["lr"]
                epoch_time = time.time() - epoch_start
                total_time = time.time() - training_start
                remaining_epochs = cfg.max_epochs - (epoch + 1)
                eta = remaining_epochs * epoch_time

                improved = self._check_improvement(val_metrics)

                # Build status indicators
                sharpe = val_metrics.get("sharpe_proxy", 0.0)
                conf_str = ""
                if "mean_confidence" in val_metrics:
                    conf_str = f" conf={val_metrics['mean_confidence']:.3f}"

                improve_marker = " ★ NEW BEST" if improved else ""
                swa_marker = " [SWA]" if use_swa else ""
                patience_bar = f"[{'█' * self._epochs_no_improve}{'░' * (cfg.patience - self._epochs_no_improve)}]" if cfg.patience <= 30 else f"[{self._epochs_no_improve}/{cfg.patience}]"

                # Progress bar for epoch
                pct = (epoch + 1) / cfg.max_epochs
                bar_len = 20
                filled = int(bar_len * pct)
                bar = f"[{'█' * filled}{'░' * (bar_len - filled)}]"

                print(
                    f"  Epoch {epoch + 1:>3}/{cfg.max_epochs} {bar} "
                    f"loss={train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} "
                    f"acc={val_metrics['accuracy'] * 100:.1f}% "
                    f"sharpe={sharpe:+.3f}{conf_str} "
                    f"lr={lr:.2e}{swa_marker} "
                    f"({epoch_time:.0f}s) ETA={self._format_eta(eta)} "
                    f"{self._gpu_stats()} "
                    f"patience={patience_bar}{improve_marker}"
                )

                if (epoch + 1) % cfg.save_every_n_epochs == 0:
                    self._save_checkpoint("latest")
                    print(f"    💾 Checkpoint saved (epoch {epoch + 1})")

                if self._epochs_no_improve >= cfg.patience:
                    print(f"\n  ⛔ Early stopping at epoch {epoch + 1} "
                          f"(no improvement for {cfg.patience} epochs)")
                    print(f"     Best Sharpe: {self._best_val_sharpe:.4f} | "
                          f"Best Loss: {self._best_val_loss:.4f}")
                    break

            # Finalize SWA
            if self._swa_model is not None and self._epoch >= cfg.swa_start_epoch:
                try:
                    from torch.optim.swa_utils import update_bn
                    print(f"\n  🔄 Running SWA batch-norm update...")
                    update_bn(train_loader, self._swa_model, device=self._device)
                    self._save_checkpoint("swa_final")
                    print(f"  ✓ SWA finalized and saved")
                except Exception as e:
                    print(f"  ⚠ SWA BN update failed: {e}")

            total_time = time.time() - training_start
            print(f"\n{'━' * 70}")
            print(f"  ✅ TRAINING COMPLETE — {self._format_eta(total_time)}")
            print(f"     Epochs:      {self._epoch + 1}")
            print(f"     Best Sharpe: {self._best_val_sharpe:.4f}")
            print(f"     Best Loss:   {self._best_val_loss:.4f}")
            print(f"     Final Loss:  {self._val_history[-1]['loss']:.4f}")
            print(f"{'━' * 70}\n")

            return {
                "total_epochs": self._epoch + 1,
                "best_val_sharpe": self._best_val_sharpe,
                "best_val_loss": self._best_val_loss,
                "final_train_loss": self._train_history[-1]["loss"],
                "final_val_loss": self._val_history[-1]["loss"],
                "model_params": self._model.count_parameters(),
            }

        @staticmethod
        def _format_eta(seconds: float) -> str:
            if seconds < 60:
                return f"{seconds:.0f}s"
            elif seconds < 3600:
                return f"{seconds / 60:.0f}m {seconds % 60:.0f}s"
            else:
                h = int(seconds // 3600)
                m = int((seconds % 3600) // 60)
                return f"{h}h {m}m"

        def _compute_aux_losses(self, aux_logits_list: list[torch.Tensor],
                                targets: list[torch.Tensor]) -> torch.Tensor:
            """Compute loss for auxiliary heads [y5m, y15m, y1h]."""
            loss = 0.0
            for logits, target in zip(aux_logits_list, targets):
                loss += self._focal(logits, target)
            return loss / 3.0

        @staticmethod
        def _diversity_loss(logits_list: list[list[torch.Tensor]]) -> torch.Tensor:
            """
            OPTIMIZED diversity loss: encourage sub-models to produce different predictions.
            Uses batched cosine similarity instead of Python loops.
            """
            # Collect the primary logits from each sub-model
            flat = []
            for logits in logits_list:
                if isinstance(logits, list):
                    flat.append(torch.cat(logits, dim=-1))
                else:
                    flat.append(logits)

            if len(flat) < 2:
                return torch.tensor(0.0)

            # Truncate all to same dim (minimum) and stack
            min_dim = min(f.size(-1) for f in flat)
            stacked = torch.stack([f[:, :min_dim] for f in flat], dim=0)  # (n_models, batch, dim)

            # Normalize once
            stacked_norm = F.normalize(stacked, p=2, dim=-1)

            # Batch pairwise cosine similarity: (n_models, n_models, batch)
            # Use einsum for efficiency
            sim_matrix = torch.einsum('ibd,jbd->ijb', stacked_norm, stacked_norm)

            # Extract upper triangle (exclude diagonal) and average
            n = stacked.size(0)
            mask = torch.triu(torch.ones(n, n, device=stacked.device), diagonal=1).bool()
            pairwise_sims = sim_matrix[mask]  # all upper-triangle similarities
            return pairwise_sims.abs().mean()

        def _train_epoch(self, loader) -> dict:
            self._model.train()
            cfg = self._config

            total_loss = 0.0
            total_cls_loss = 0.0
            total_aux_loss = 0.0
            total_correct = 0
            total_samples = 0
            n_batches = 0
            batch_times = []

            self._optimizer.zero_grad(set_to_none=True)

            # Create progress bar
            n_total = len(loader)
            if HAS_TQDM:
                pbar = tqdm(
                    loader, desc=f"  Train E{self._epoch + 1:>3}",
                    ncols=120, leave=False,
                    bar_format="  {desc} |{bar:25}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                )
            else:
                pbar = loader

            for batch_idx, batch in enumerate(pbar):
                batch_t0 = time.time()
                cont, cat, y5m, y15m, y1h, raw_ret = [b.to(self._device) for b in batch]
                targets = [y5m, y15m, y1h]

                # ── NaN input guard ──
                if torch.isnan(cont).any() or torch.isinf(cont).any():
                    if batch_idx == 0:
                        print(f"    ⚠ NaN/Inf in input features at batch {batch_idx} — skipping")
                    continue
                if torch.isnan(raw_ret).any() or torch.isinf(raw_ret).any():
                    raw_ret = torch.nan_to_num(raw_ret, nan=0.0, posinf=0.0, neginf=0.0)

                # Mixup augmentation
                if cfg.mixup_alpha > 0 and self.training_uses_mixup:
                    cont, cat, y5m, y15m, y1h, raw_ret, mixup_info = self._mixup_data(
                        cont, cat, y5m, y15m, y1h, raw_ret, cfg.mixup_alpha,
                    )
                    targets = [y5m, y15m, y1h]

                with self._autocast_context():
                    outputs = self._model(cont, cat)

                    # 1. Main Classification Loss (Label Smoothing Focal)
                    loss_cls = (
                        self._focal(outputs["logits_5m"], y5m) +
                        self._focal(outputs["logits_15m"], y15m) +
                        self._focal(outputs["logits_1h"], y1h)
                    ) / 3.0

                    # 2. Quantile Loss
                    loss_q = (
                        self._quantile(outputs["quantiles_5m"], raw_ret[:, 0:1].expand(-1, 3)) +
                        self._quantile(outputs["quantiles_15m"], raw_ret[:, 1:2].expand(-1, 3)) +
                        self._quantile(outputs["quantiles_1h"], raw_ret[:, 2:3].expand(-1, 3))
                    ) / 3.0

                    # 3. ALL 6 Auxiliary Sub-Model Losses
                    loss_tft_aux = self._compute_aux_losses(outputs["tft_logits"], targets)
                    loss_lstm_aux = self._compute_aux_losses(outputs["lstm_logits"], targets)
                    loss_cnn_aux = self._compute_aux_losses(outputs["cnn_logits"], targets)
                    loss_moe_aux = self._compute_aux_losses(outputs["moe_logits"], targets)
                    loss_tcn_aux = self._compute_aux_losses(outputs["tcn_logits"], targets)
                    loss_trans_aux = self._compute_aux_losses(outputs["transformer_logits"], targets)

                    total_aux = (
                        loss_tft_aux + loss_lstm_aux + loss_cnn_aux +
                        loss_moe_aux + loss_tcn_aux + loss_trans_aux
                    ) / 6.0

                    # 4. MoE Load Balance Loss
                    moe_lb_loss = outputs.get("moe_load_balance_loss", torch.tensor(0.0))

                    # 5. Diversity Loss
                    diversity = self._diversity_loss([
                        outputs["tft_logits"],
                        outputs["lstm_logits"][0] if outputs["lstm_logits"] else outputs["logits_5m"],
                        outputs["cnn_logits"][0] if outputs["cnn_logits"] else outputs["logits_5m"],
                        outputs["moe_logits"][0] if outputs["moe_logits"] else outputs["logits_5m"],
                    ])

                    # Combined SUPER INSANE Objective Function
                    loss = (
                        cfg.classification_loss_weight * loss_cls +
                        cfg.quantile_loss_weight * loss_q +
                        cfg.aux_loss_weight * total_aux +
                        cfg.moe_balance_weight * moe_lb_loss +
                        cfg.diversity_loss_weight * diversity
                    )

                    # Scale for gradient accumulation
                    loss = loss / cfg.gradient_accumulation_steps

                # ── NaN loss guard ──
                if torch.isnan(loss) or torch.isinf(loss):
                    if batch_idx < 3:
                        print(f"    ⚠ NaN/Inf loss at batch {batch_idx} — "
                              f"cls={loss_cls.item():.4f} q={loss_q.item():.4f} "
                              f"aux={total_aux.item():.4f} moe={moe_lb_loss.item():.4f}")
                    self._optimizer.zero_grad(set_to_none=True)
                    n_batches += 1
                    batch_times.append(time.time() - batch_t0)
                    continue

                self._scaler.scale(loss).backward()

                # Gradient accumulation step
                if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                    self._scaler.unscale_(self._optimizer)
                    nn.utils.clip_grad_norm_(self._model.parameters(), cfg.gradient_clip_norm)
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                    self._optimizer.zero_grad(set_to_none=True)

                batch_loss = loss.item() * cfg.gradient_accumulation_steps
                total_loss += batch_loss
                total_cls_loss += loss_cls.item()
                total_aux_loss += total_aux.item()
                preds = outputs["logits_1h"].argmax(dim=-1)
                total_correct += (preds == y1h).sum().item()
                total_samples += y1h.shape[0]
                n_batches += 1
                batch_times.append(time.time() - batch_t0)

                # Update progress bar every batch
                if HAS_TQDM and batch_idx % 2 == 0:
                    avg_loss = total_loss / n_batches
                    acc = total_correct / max(total_samples, 1) * 100
                    samples_sec = total_samples / sum(batch_times) if batch_times else 0
                    pbar.set_postfix_str(
                        f"loss={avg_loss:.4f} cls={total_cls_loss / n_batches:.3f} "
                        f"aux={total_aux_loss / n_batches:.3f} "
                        f"acc={acc:.1f}% {samples_sec:.0f} samp/s"
                    )

            if HAS_TQDM:
                pbar.close()

            # Flush remaining gradients
            if n_batches % cfg.gradient_accumulation_steps != 0:
                self._scaler.unscale_(self._optimizer)
                nn.utils.clip_grad_norm_(self._model.parameters(), cfg.gradient_clip_norm)
                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._optimizer.zero_grad(set_to_none=True)

            return {
                "loss": total_loss / max(n_batches, 1),
                "accuracy": total_correct / max(total_samples, 1),
                "cls_loss": total_cls_loss / max(n_batches, 1),
                "aux_loss": total_aux_loss / max(n_batches, 1),
                "samples_per_sec": total_samples / sum(batch_times) if batch_times else 0,
            }

        @property
        def training_uses_mixup(self) -> bool:
            """Only use mixup after warmup phase."""
            return self._epoch >= self._config.warmup_epochs

        @torch.inference_mode()
        def _validate(self, loader) -> dict:
            """OPTIMIZED: Uses torch.inference_mode (faster than no_grad)."""
            self._model.eval()
            cfg = self._config

            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            n_batches = 0
            all_preds = []
            all_confidence = []

            if HAS_TQDM:
                pbar = tqdm(
                    loader, desc=f"  Valid E{self._epoch + 1:>3}",
                    ncols=120, leave=False,
                    bar_format="  {desc} |{bar:25}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
                )
            else:
                pbar = loader

            for batch in pbar:
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

                # Track confidence
                if "confidence" in outputs:
                    all_confidence.extend(outputs["confidence"].squeeze(-1).cpu().numpy().tolist())

                pred_dir = preds.float() - 1.0  # SHORT=-1, FLAT=0, LONG=1
                actual_ret = raw_ret[:, 2]      # 1h returns
                strategy_ret = pred_dir * actual_ret
                all_preds.extend(strategy_ret.cpu().numpy().tolist())

                n_batches += 1

                # Update validation progress bar
                if HAS_TQDM and n_batches % 5 == 0:
                    avg_loss = total_loss / n_batches
                    acc = total_correct / max(total_samples, 1) * 100
                    pbar.set_postfix_str(f"loss={avg_loss:.4f} acc={acc:.1f}%")

            if HAS_TQDM:
                pbar.close()

            all_preds_arr = np.array(all_preds)
            sharpe_proxy = 0.0
            if len(all_preds_arr) > 1 and np.std(all_preds_arr) > 0:
                sharpe_proxy = float(np.mean(all_preds_arr) / np.std(all_preds_arr) * np.sqrt(252 * 24))

            result = {
                "loss": total_loss / max(n_batches, 1),
                "accuracy": total_correct / max(total_samples, 1),
                "sharpe_proxy": sharpe_proxy,
            }

            if all_confidence:
                result["mean_confidence"] = float(np.mean(all_confidence))
                result["confidence_std"] = float(np.std(all_confidence))

            return result

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

            save_dict = {
                "epoch": self._epoch,
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "scheduler_state_dict": self._scheduler.state_dict(),
                "scaler_state_dict": self._scaler.state_dict(),
                "best_val_sharpe": self._best_val_sharpe,
                "best_val_loss": self._best_val_loss,
                "ensemble_config": self._config.ensemble_config,
                "trainer_config": self._config,
                "train_history": self._train_history,
                "val_history": self._val_history,
            }
            if self._swa_model is not None:
                save_dict["swa_model_state_dict"] = self._swa_model.state_dict()

            torch.save(save_dict, path)
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
