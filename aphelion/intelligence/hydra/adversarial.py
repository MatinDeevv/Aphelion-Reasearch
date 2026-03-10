"""
HYDRA adversarial robustness guard.

Generates targeted perturbations on continuous input features to estimate how
close the model is to an adversarial decision boundary. Used online to
downweight confidence before live trade decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class AdversarialConfig:
    enabled: bool = True
    epsilon: float = 0.20
    step_size: float = 0.05
    steps: int = 4
    risk_penalty: float = 0.65


@dataclass
class AdversarialAssessment:
    raw_confidence: float = 0.0
    robust_confidence: float = 0.0
    risk_score: float = 0.0
    boundary_distance: float = 0.0
    confidence_drop: float = 0.0
    base_class: int = 1
    adversarial_class: int = 1
    attack_success: bool = False


class AdversarialFeaturePerturbationDetector:
    def __init__(self, config: AdversarialConfig | None = None):
        self._cfg = config or AdversarialConfig()

    def assess(self, model, cont_t, cat_t, base_outputs=None) -> AdversarialAssessment:
        if not HAS_TORCH or not self._cfg.enabled:
            return AdversarialAssessment()

        was_training = bool(model.training)
        model.eval()

        try:
            with torch.no_grad():
                outputs = base_outputs if base_outputs is not None else model(cont_t, cat_t)
                probs = outputs["probs_1h"][0]
                base_class = int(torch.argmax(probs).item())
                sorted_probs, _ = torch.sort(probs, descending=True)
                raw_conf = float(sorted_probs[0].item())
                margin = float((sorted_probs[0] - sorted_probs[1]).item()) if len(sorted_probs) >= 2 else raw_conf

            if base_class == 2:
                target_class = 0
            elif base_class == 0:
                target_class = 2
            else:
                target_class = 0 if float(probs[0]) >= float(probs[2]) else 2

            x0 = cont_t.detach()
            adv = x0.clone().detach()

            eps = float(max(self._cfg.epsilon, 1e-6))
            step = float(max(self._cfg.step_size, 1e-6))
            n_steps = int(max(self._cfg.steps, 1))

            for _ in range(n_steps):
                adv.requires_grad_(True)
                out_adv = model(adv, cat_t)
                logits = out_adv["logits_1h"]
                target = torch.full(
                    (logits.shape[0],),
                    int(target_class),
                    device=logits.device,
                    dtype=torch.long,
                )
                # Targeted attack: reduce CE toward target class.
                loss = F.cross_entropy(logits, target)
                grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]

                adv = adv - step * torch.sign(grad)
                perturb = torch.clamp(adv - x0, min=-eps, max=eps)
                adv = (x0 + perturb).detach()

            with torch.no_grad():
                out_adv = model(adv, cat_t)
                probs_adv = out_adv["probs_1h"][0]
                adv_class = int(torch.argmax(probs_adv).item())
                base_prob_after = float(probs_adv[base_class].item())
                confidence_drop = max(raw_conf - base_prob_after, 0.0)

            perturb = adv - x0
            l2 = float(torch.norm(perturb).item()) / float((perturb.numel() ** 0.5) + 1e-12)
            attack_success = adv_class != base_class
            boundary = l2 if attack_success else eps + margin

            closeness = max(0.0, min(1.0, 1.0 - boundary / eps))
            flip_term = 1.0 if attack_success else 0.0
            risk = 0.45 * closeness + 0.40 * min(confidence_drop, 1.0) + 0.15 * flip_term
            risk = max(0.0, min(1.0, risk))

            penalty = min(risk * self._cfg.risk_penalty, 0.95)
            robust_conf = max(0.0, raw_conf * (1.0 - penalty))

            return AdversarialAssessment(
                raw_confidence=raw_conf,
                robust_confidence=float(robust_conf),
                risk_score=float(risk),
                boundary_distance=float(boundary),
                confidence_drop=float(confidence_drop),
                base_class=int(base_class),
                adversarial_class=int(adv_class),
                attack_success=bool(attack_success),
            )

        finally:
            if was_training:
                model.train()
