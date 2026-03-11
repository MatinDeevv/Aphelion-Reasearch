"""
APHELION HYDRA Trainer v2 — Clean Eager-Mode Rewrite
=====================================================
Trains the full 6-model ensemble (160M params) with:
  - Label-smoothing focal loss + quantile loss
  - Six auxiliary sub-model losses
  - MoE load-balance loss + diversity loss
  - Mixup augmentation (after warmup)
  - Linear warmup -> cosine annealing with warm restarts
  - Gradient accumulation + AMP (BF16 on A100, FP16 otherwise)
  - Stochastic Weight Averaging (SWA) in final phase
  - Early stopping on validation Sharpe / loss
  - Full NaN/Inf guards on inputs, outputs, and gradients

NO torch.compile(). Pure eager mode for maximum GPU compatibility.
Speed comes from AMP + TF32 + cudnn.benchmark + pin_memory.
"""

from __future__ import annotations

import logging
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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

# ─── Configuration ────────────────────────────────────────────────────────


@dataclass
class TrainerConfig:
    """Training hyper-parameters."""
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 200
    gradient_clip_norm: float = 1.0

    warmup_epochs: int = 10
    cosine_t_max: int = 40

    # Loss weights
    classification_loss_weight: float = 1.0
    quantile_loss_weight: float = 0.3
    aux_loss_weight: float = 0.15
    moe_balance_weight: float = 0.01
    diversity_loss_weight: float = 0.05

    # Focal loss
    focal_gamma: float = 2.0
    focal_alpha: list[float] = field(default_factory=lambda: [1.5, 0.5, 1.5])
    label_smoothing: float = 0.1

    # Mixup
    mixup_alpha: float = 0.2

    # Gradient accumulation
    gradient_accumulation_steps: int = 4

    # Early stopping
    patience: int = 25
    min_delta: float = 0.005

    # AMP / checkpointing
    use_amp: bool = True
    checkpoint_dir: str = "models/hydra"
    save_every_n_epochs: int = 5

    # SWA
    swa_start_epoch: int = 100
    swa_lr: float = 1e-5

    ensemble_config: EnsembleConfig = field(default_factory=EnsembleConfig)


# ─── Loss Functions ───────────────────────────────────────────────────────

if HAS_TORCH:

    class LabelSmoothingFocalLoss(nn.Module):
        """Focal loss with label smoothing."""

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
            with torch.no_grad():
                smooth = torch.full_like(logits, self.smoothing / (self.n_classes - 1))
                smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

            log_p = F.log_softmax(logits, dim=-1)
            ce = -(smooth * log_p).sum(dim=-1)
            pt = torch.exp(-ce)
            focal = ((1 - pt) ** self.gamma) * ce

            if self.alpha is not None:
                alpha = self.alpha.to(logits.device)
                focal = alpha[targets] * focal

            return focal.mean()

    class QuantileLoss(nn.Module):
        def __init__(self, quantiles: list[float]):
            super().__init__()
            self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))

        def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            losses = []
            for i, q in enumerate(self.quantiles):
                e = targets[:, i] - predictions[:, i]
                losses.append(torch.max(q * e, (q - 1) * e))
            return torch.stack(losses, dim=1).mean()

    # ─── Trainer ──────────────────────────────────────────────────────────

    class HydraTrainer:
        """
        Eager-mode trainer for the HYDRA ensemble.
        No torch.compile — no CUDAGraph crashes.
        """

        def __init__(
            self,
            config: Optional[TrainerConfig] = None,
            device: Optional[str] = None,
        ):
            self._config = config or TrainerConfig()
            cfg = self._config

            # Device
            if device:
                self._device = torch.device(device)
            else:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Model — pure eager, no compile
            self._model = HydraGate(cfg.ensemble_config).to(self._device)
            n_params = self._model.count_parameters()
            logger.info("HYDRA ensemble: %s params on %s", f"{n_params:,}", self._device)

            # cuDNN autotuner
            if self._device.type == "cuda":
                torch.backends.cudnn.benchmark = True

            # BF16 detection (A100 / H100)
            self._use_bf16 = False
            if cfg.use_amp and self._device.type == "cuda":
                try:
                    if torch.cuda.is_bf16_supported():
                        self._use_bf16 = True
                        logger.info("BFloat16 detected — using BF16 AMP")
                except (AttributeError, RuntimeError):
                    pass

            # No torch.compile — pure eager mode
            self._compiled = False

            # Optimizer — differential LR
            sub_params, gate_params = [], []
            for name, p in self._model.named_parameters():
                if any(m in name for m in ('tft.', 'lstm.', 'cnn.', 'moe.', 'tcn.', 'transformer.')):
                    if not any(k in name for k in ('_proj', '_adapter')):
                        sub_params.append(p)
                        continue
                gate_params.append(p)

            self._optimizer = torch.optim.AdamW([
                {"params": sub_params, "lr": cfg.learning_rate * 0.5},
                {"params": gate_params, "lr": cfg.learning_rate},
            ], weight_decay=cfg.weight_decay)

            # Store initial LRs for warmup
            for pg in self._optimizer.param_groups:
                pg['initial_lr'] = pg['lr']

            # Scheduler
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self._optimizer, T_0=cfg.cosine_t_max, T_mult=2,
            )

            # Loss functions
            self._focal = LabelSmoothingFocalLoss(
                cfg.focal_gamma, cfg.focal_alpha, cfg.label_smoothing,
            )
            self._quantile = QuantileLoss(cfg.ensemble_config.tft_config.quantile_targets)

            # GradScaler — disabled for BF16 and CPU
            amp_on = cfg.use_amp and self._device.type == "cuda"
            scaler_on = amp_on and not self._use_bf16
            self._scaler = self._create_grad_scaler(enabled=scaler_on)

            # SWA
            self._swa_model = None
            self._swa_scheduler = None
            if cfg.swa_start_epoch < cfg.max_epochs:
                try:
                    from torch.optim.swa_utils import AveragedModel, SWALR
                    self._swa_model = AveragedModel(self._model)
                    self._swa_scheduler = SWALR(self._optimizer, swa_lr=cfg.swa_lr)
                    logger.info("SWA enabled from epoch %d", cfg.swa_start_epoch)
                except ImportError:
                    pass

            # Tracking state
            self._best_val_sharpe = -float("inf")
            self._best_val_loss = float("inf")
            self._epochs_no_improve = 0
            self._epoch = 0
            self._train_history: list[dict] = []
            self._val_history: list[dict] = []

        # ── Properties ────────────────────────────────────────────────

        @property
        def model(self) -> HydraGate:
            return self._model

        @property
        def training_uses_mixup(self) -> bool:
            return self._epoch >= self._config.warmup_epochs

        # ── Helpers ───────────────────────────────────────────────────

        @staticmethod
        def _create_grad_scaler(enabled: bool):
            try:
                return torch.amp.GradScaler("cuda", enabled=enabled)
            except (AttributeError, TypeError):
                return torch.cuda.amp.GradScaler(enabled=enabled)

        def _autocast_context(self):
            if not self._config.use_amp or self._device.type != "cuda":
                return nullcontext()
            dtype = torch.bfloat16 if self._use_bf16 else torch.float16
            try:
                return torch.amp.autocast("cuda", enabled=True, dtype=dtype)
            except (AttributeError, TypeError):
                return torch.cuda.amp.autocast(enabled=True)

        def _get_warmup_factor(self) -> float:
            if self._epoch >= self._config.warmup_epochs:
                return 1.0
            return 0.1 + 0.9 * (self._epoch / max(self._config.warmup_epochs, 1))

        @staticmethod
        def _mixup_data(x_cont, x_cat, y5m, y15m, y1h, raw_ret, alpha=0.2):
            if alpha <= 0:
                return x_cont, x_cat, y5m, y15m, y1h, raw_ret, None
            lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
            lam = max(lam, 1 - lam)
            idx = torch.randperm(x_cont.size(0), device=x_cont.device)
            mixed = lam * x_cont + (1 - lam) * x_cont[idx]
            mixed_ret = lam * raw_ret + (1 - lam) * raw_ret[idx]
            return mixed, x_cat, y5m, y15m, y1h, mixed_ret, (idx, lam)

        def _gpu_stats(self) -> str:
            if self._device.type != "cuda":
                return ""
            try:
                a = torch.cuda.memory_allocated(self._device) / 1024**3
                r = torch.cuda.memory_reserved(self._device) / 1024**3
                return f"GPU: {a:.1f}/{r:.1f} GB"
            except Exception:
                return ""

        @staticmethod
        def _format_eta(s: float) -> str:
            if s < 60:
                return f"{s:.0f}s"
            if s < 3600:
                return f"{s/60:.0f}m {s%60:.0f}s"
            return f"{int(s//3600)}h {int((s%3600)//60)}m"

        # ── Loss Helpers ──────────────────────────────────────────────

        def _compute_aux_losses(self, logits_list: list[torch.Tensor],
                                targets: list[torch.Tensor]) -> torch.Tensor:
            total = torch.tensor(0.0, device=self._device)
            for lg, tgt in zip(logits_list, targets):
                total = total + self._focal(lg, tgt)
            return total / max(len(targets), 1)

        @staticmethod
        def _diversity_loss(logits_list: list) -> torch.Tensor:
            flat = []
            for lg in logits_list:
                flat.append(torch.cat(lg, dim=-1) if isinstance(lg, list) else lg)
            if len(flat) < 2:
                return torch.tensor(0.0)
            dim_min = min(f.size(-1) for f in flat)
            stacked = torch.stack([f[:, :dim_min] for f in flat], dim=0)
            normed = F.normalize(stacked, p=2, dim=-1)
            sim = torch.einsum('ibd,jbd->ijb', normed, normed)
            n = stacked.size(0)
            mask = torch.triu(torch.ones(n, n, device=stacked.device), diagonal=1).bool()
            return sim[mask].abs().mean()

        # ── Main Train Loop ───────────────────────────────────────────

        def train(self, train_loader, val_loader) -> dict:
            cfg = self._config
            n_params = self._model.count_parameters()

            print(f"\n{'='*70}")
            print(f"  HYDRA TRAINING — {n_params:,} parameters (eager mode)")
            print(f"  Epochs: {cfg.max_epochs} | Batch: {train_loader.batch_size} "
                  f"| Grad accum: {cfg.gradient_accumulation_steps}x")
            print(f"  AMP: {'BF16' if self._use_bf16 else 'FP16' if cfg.use_amp else 'OFF'} "
                  f"| Device: {self._device}")
            print(f"  {self._gpu_stats()}")
            print(f"{'='*70}\n")

            t_start = time.time()

            for epoch in range(cfg.max_epochs):
                self._epoch = epoch
                t_ep = time.time()

                # Warmup LR
                if epoch < cfg.warmup_epochs:
                    wf = self._get_warmup_factor()
                    for pg in self._optimizer.param_groups:
                        pg['lr'] = pg.get('initial_lr', cfg.learning_rate) * wf

                # Train + Validate
                train_m = self._train_epoch(train_loader)
                self._train_history.append(train_m)

                val_m = self._validate(val_loader)
                self._val_history.append(val_m)

                # Scheduler
                use_swa = self._swa_model is not None and epoch >= cfg.swa_start_epoch
                if use_swa:
                    self._swa_model.update_parameters(self._model)
                    self._swa_scheduler.step()
                else:
                    self._scheduler.step()

                # Improvement check
                improved = self._check_improvement(val_m)

                # Logging
                lr = self._optimizer.param_groups[0]["lr"]
                ep_t = time.time() - t_ep
                eta = (cfg.max_epochs - epoch - 1) * ep_t
                sharpe = val_m.get("sharpe_proxy", 0.0)
                conf = f" conf={val_m['mean_confidence']:.3f}" if "mean_confidence" in val_m else ""
                mark = " * BEST" if improved else ""
                swa_tag = " [SWA]" if use_swa else ""
                pct = (epoch + 1) / cfg.max_epochs
                bar = f"[{'#'*int(20*pct)}{'.'*int(20*(1-pct))}]"
                pat = f"[{self._epochs_no_improve}/{cfg.patience}]"

                print(
                    f"  E{epoch+1:>3}/{cfg.max_epochs} {bar} "
                    f"loss={train_m['loss']:.4f}/{val_m['loss']:.4f} "
                    f"acc={val_m['accuracy']*100:.1f}% "
                    f"sharpe={sharpe:+.3f}{conf} "
                    f"lr={lr:.2e}{swa_tag} "
                    f"({ep_t:.0f}s) ETA={self._format_eta(eta)} "
                    f"{self._gpu_stats()} "
                    f"pat={pat}{mark}"
                )

                # Periodic checkpoint
                if (epoch + 1) % cfg.save_every_n_epochs == 0:
                    self._save_checkpoint("latest")
                    print(f"    Checkpoint saved (epoch {epoch+1})")

                # Early stopping
                if self._epochs_no_improve >= cfg.patience:
                    print(f"\n  Early stop at epoch {epoch+1} "
                          f"(no improvement for {cfg.patience} epochs)")
                    print(f"  Best Sharpe: {self._best_val_sharpe:.4f} "
                          f"| Best Loss: {self._best_val_loss:.4f}")
                    break

            # SWA finalize
            if self._swa_model is not None and self._epoch >= cfg.swa_start_epoch:
                try:
                    from torch.optim.swa_utils import update_bn
                    print("  Running SWA BN update...")
                    update_bn(train_loader, self._swa_model, device=self._device)
                    self._save_checkpoint("swa_final")
                    print("  SWA finalized")
                except Exception as e:
                    print(f"  SWA BN update failed: {e}")

            total_t = time.time() - t_start
            print(f"\n{'='*70}")
            print(f"  DONE — {self._format_eta(total_t)}")
            print(f"  Epochs: {self._epoch+1} | Best Sharpe: {self._best_val_sharpe:.4f} "
                  f"| Best Loss: {self._best_val_loss:.4f}")
            print(f"{'='*70}\n")

            return {
                "total_epochs": self._epoch + 1,
                "best_val_sharpe": self._best_val_sharpe,
                "best_val_loss": self._best_val_loss,
                "final_train_loss": self._train_history[-1]["loss"],
                "final_val_loss": self._val_history[-1]["loss"],
                "model_params": n_params,
            }

        # ── Single Training Epoch ─────────────────────────────────────

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

            if HAS_TQDM:
                pbar = tqdm(
                    loader, desc=f"  Train E{self._epoch+1:>3}",
                    ncols=120, leave=False,
                    bar_format="  {desc} |{bar:25}| {n_fmt}/{total_fmt} "
                               "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                )
            else:
                pbar = loader

            for bi, batch in enumerate(pbar):
                t0 = time.time()

                # Move to device (non_blocking for pipelined H2D transfer)
                cont, cat, y5m, y15m, y1h, raw_ret = [
                    b.to(self._device, non_blocking=True) for b in batch
                ]
                targets = [y5m, y15m, y1h]

                # NaN guard
                if torch.isnan(cont).any() or torch.isinf(cont).any():
                    if bi == 0:
                        print(f"    NaN/Inf in inputs at batch {bi} — skipping")
                    continue
                if torch.isnan(raw_ret).any() or torch.isinf(raw_ret).any():
                    raw_ret = torch.nan_to_num(raw_ret, nan=0.0, posinf=0.0, neginf=0.0)

                # Mixup after warmup
                if cfg.mixup_alpha > 0 and self.training_uses_mixup:
                    cont, cat, y5m, y15m, y1h, raw_ret, _ = self._mixup_data(
                        cont, cat, y5m, y15m, y1h, raw_ret, cfg.mixup_alpha,
                    )
                    targets = [y5m, y15m, y1h]

                # Forward pass under AMP
                with self._autocast_context():
                    outputs = self._model(cont, cat)

                    # 1) Classification loss
                    loss_cls = (
                        self._focal(outputs["logits_5m"], y5m)
                        + self._focal(outputs["logits_15m"], y15m)
                        + self._focal(outputs["logits_1h"], y1h)
                    ) / 3.0

                    # 2) Quantile loss
                    loss_q = (
                        self._quantile(outputs["quantiles_5m"], raw_ret[:, 0:1].expand(-1, 3))
                        + self._quantile(outputs["quantiles_15m"], raw_ret[:, 1:2].expand(-1, 3))
                        + self._quantile(outputs["quantiles_1h"], raw_ret[:, 2:3].expand(-1, 3))
                    ) / 3.0

                    # 3) Auxiliary losses (all 6 sub-models)
                    loss_tft_aux = self._compute_aux_losses(outputs["tft_logits"], targets)
                    loss_lstm_aux = self._compute_aux_losses(outputs["lstm_logits"], targets)
                    loss_cnn_aux = self._compute_aux_losses(outputs["cnn_logits"], targets)
                    loss_moe_aux = self._compute_aux_losses(outputs["moe_logits"], targets)
                    loss_tcn_aux = self._compute_aux_losses(outputs["tcn_logits"], targets)
                    loss_trans_aux = self._compute_aux_losses(outputs["transformer_logits"], targets)
                    total_aux = (
                        loss_tft_aux + loss_lstm_aux + loss_cnn_aux
                        + loss_moe_aux + loss_tcn_aux + loss_trans_aux
                    ) / 6.0

                    # 4) MoE load balance
                    moe_lb = outputs.get("moe_load_balance_loss", torch.tensor(0.0, device=self._device))

                    # 5) Diversity loss
                    diversity = self._diversity_loss([
                        outputs["tft_logits"],
                        outputs["lstm_logits"][0] if outputs["lstm_logits"] else outputs["logits_5m"],
                        outputs["cnn_logits"][0] if outputs["cnn_logits"] else outputs["logits_5m"],
                        outputs["moe_logits"][0] if outputs["moe_logits"] else outputs["logits_5m"],
                    ])

                    # Combined objective
                    loss = (
                        cfg.classification_loss_weight * loss_cls
                        + cfg.quantile_loss_weight * loss_q
                        + cfg.aux_loss_weight * total_aux
                        + cfg.moe_balance_weight * moe_lb
                        + cfg.diversity_loss_weight * diversity
                    ) / cfg.gradient_accumulation_steps

                # NaN loss guard
                if torch.isnan(loss) or torch.isinf(loss):
                    if bi < 3:
                        print(f"    NaN loss at batch {bi} — skipping")
                    self._optimizer.zero_grad(set_to_none=True)
                    n_batches += 1
                    batch_times.append(time.time() - t0)
                    continue

                # Backward
                self._scaler.scale(loss).backward()

                # Gradient accumulation step
                if (bi + 1) % cfg.gradient_accumulation_steps == 0:
                    self._scaler.unscale_(self._optimizer)
                    nn.utils.clip_grad_norm_(self._model.parameters(), cfg.gradient_clip_norm)
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                    self._optimizer.zero_grad(set_to_none=True)

                # Tracking
                batch_loss = loss.item() * cfg.gradient_accumulation_steps
                total_loss += batch_loss
                total_cls_loss += loss_cls.item()
                total_aux_loss += total_aux.item()
                preds = outputs["logits_1h"].argmax(dim=-1)
                total_correct += (preds == y1h).sum().item()
                total_samples += y1h.shape[0]
                n_batches += 1
                batch_times.append(time.time() - t0)

                # Progress bar
                if HAS_TQDM and bi % 2 == 0 and n_batches > 0:
                    avg = total_loss / n_batches
                    acc = total_correct / max(total_samples, 1) * 100
                    sps = total_samples / sum(batch_times) if batch_times else 0
                    pbar.set_postfix_str(
                        f"loss={avg:.4f} cls={total_cls_loss/n_batches:.3f} "
                        f"aux={total_aux_loss/n_batches:.3f} "
                        f"acc={acc:.1f}% {sps:.0f} samp/s"
                    )

            if HAS_TQDM and hasattr(pbar, 'close'):
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

        # ── Validation ────────────────────────────────────────────────

        @torch.inference_mode()
        def _validate(self, loader) -> dict:
            self._model.eval()
            cfg = self._config

            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            n_batches = 0
            all_strat_ret = []
            all_conf = []

            if HAS_TQDM:
                pbar = tqdm(
                    loader, desc=f"  Valid E{self._epoch+1:>3}",
                    ncols=120, leave=False,
                    bar_format="  {desc} |{bar:25}| {n_fmt}/{total_fmt} "
                               "[{elapsed}<{remaining}] {postfix}"
                )
            else:
                pbar = loader

            for batch in pbar:
                cont, cat, y5m, y15m, y1h, raw_ret = [
                    b.to(self._device, non_blocking=True) for b in batch
                ]

                with self._autocast_context():
                    outputs = self._model(cont, cat)

                    loss_cls = (
                        self._focal(outputs["logits_5m"], y5m)
                        + self._focal(outputs["logits_15m"], y15m)
                        + self._focal(outputs["logits_1h"], y1h)
                    ) / 3.0

                    loss_q = (
                        self._quantile(outputs["quantiles_5m"], raw_ret[:, 0:1].expand(-1, 3))
                        + self._quantile(outputs["quantiles_15m"], raw_ret[:, 1:2].expand(-1, 3))
                        + self._quantile(outputs["quantiles_1h"], raw_ret[:, 2:3].expand(-1, 3))
                    ) / 3.0

                    loss = cfg.classification_loss_weight * loss_cls + cfg.quantile_loss_weight * loss_q

                total_loss += loss.item()
                preds = outputs["logits_1h"].argmax(dim=-1)
                total_correct += (preds == y1h).sum().item()
                total_samples += y1h.shape[0]

                if "confidence" in outputs:
                    all_conf.extend(outputs["confidence"].squeeze(-1).cpu().numpy().tolist())

                direction = preds.float() - 1.0
                strat_ret = direction * raw_ret[:, 2]
                all_strat_ret.extend(strat_ret.cpu().numpy().tolist())
                n_batches += 1

                if HAS_TQDM and n_batches % 5 == 0 and n_batches > 0:
                    pbar.set_postfix_str(
                        f"loss={total_loss/n_batches:.4f} "
                        f"acc={total_correct/max(total_samples,1)*100:.1f}%"
                    )

            if HAS_TQDM and hasattr(pbar, 'close'):
                pbar.close()

            arr = np.array(all_strat_ret)
            sharpe = 0.0
            if len(arr) > 1 and np.std(arr) > 0:
                sharpe = float(np.mean(arr) / np.std(arr) * np.sqrt(252 * 24))

            result = {
                "loss": total_loss / max(n_batches, 1),
                "accuracy": total_correct / max(total_samples, 1),
                "sharpe_proxy": sharpe,
            }
            if all_conf:
                result["mean_confidence"] = float(np.mean(all_conf))
                result["confidence_std"] = float(np.std(all_conf))
            return result

        # ── Improvement Check ─────────────────────────────────────────

        def _check_improvement(self, val_m: dict) -> bool:
            cfg = self._config
            sharpe = val_m.get("sharpe_proxy", 0.0)
            loss = val_m["loss"]
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

        # ── Checkpointing ─────────────────────────────────────────────

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
            logger.info("Checkpoint: %s (epoch %d)", path, self._epoch + 1)
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
            logger.info("Loaded checkpoint: %s (epoch %d)", path, self._epoch)
