"""
APHELION HYDRA — XGBoost & RandomForest wrappers (Phase 7 v2)
Fast, interpretable models complementing deep-learning ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List

import numpy as np

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class TreeModelConfig:
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    min_samples_leaf: int = 20
    n_classes: int = 3  # DOWN, FLAT, UP
    lookback: int = 50
    random_state: int = 42


@dataclass
class TreePrediction:
    direction: int  # -1, 0, 1
    confidence: float
    class_probabilities: Dict[int, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)


class HydraXGBoost:
    """Gradient Boosting classifier for HYDRA ensemble — fast and interpretable."""

    def __init__(self, config: Optional[TreeModelConfig] = None):
        self.config = config or TreeModelConfig()
        self._model = None
        self._feature_names: List[str] = []
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        if not HAS_SKLEARN:
            raise RuntimeError("scikit-learn required for XGBoost model")
        cfg = self.config
        self._model = GradientBoostingClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.random_state,
        )
        self._model.fit(X, y)
        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self._fitted = True
        # Return training accuracy
        train_acc = self._model.score(X, y)
        return {"train_accuracy": train_acc}

    def predict(self, X: np.ndarray) -> TreePrediction:
        if not self._fitted or self._model is None:
            return TreePrediction(direction=0, confidence=0.0)
        proba = self._model.predict_proba(X[-1:])
        classes = self._model.classes_
        class_probs = {int(c): float(p) for c, p in zip(classes, proba[0])}

        best_class = int(classes[np.argmax(proba[0])])
        direction = -1 if best_class == 0 else (1 if best_class == 2 else 0)
        confidence = float(np.max(proba[0]))

        importance = {}
        if hasattr(self._model, "feature_importances_"):
            for name, imp in zip(self._feature_names, self._model.feature_importances_):
                importance[name] = float(imp)

        return TreePrediction(
            direction=direction,
            confidence=confidence,
            class_probabilities=class_probs,
            feature_importance=importance,
        )

    @property
    def is_fitted(self) -> bool:
        return self._fitted


class HydraRandomForest:
    """Random Forest classifier for HYDRA ensemble — decorrelates from DL models."""

    def __init__(self, config: Optional[TreeModelConfig] = None):
        self.config = config or TreeModelConfig()
        self._model = None
        self._feature_names: List[str] = []
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        if not HAS_SKLEARN:
            raise RuntimeError("scikit-learn required for RandomForest model")
        cfg = self.config
        self._model = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.random_state,
        )
        self._model.fit(X, y)
        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self._fitted = True
        return {"train_accuracy": self._model.score(X, y)}

    def predict(self, X: np.ndarray) -> TreePrediction:
        if not self._fitted or self._model is None:
            return TreePrediction(direction=0, confidence=0.0)
        proba = self._model.predict_proba(X[-1:])
        classes = self._model.classes_
        class_probs = {int(c): float(p) for c, p in zip(classes, proba[0])}

        best_class = int(classes[np.argmax(proba[0])])
        direction = -1 if best_class == 0 else (1 if best_class == 2 else 0)
        confidence = float(np.max(proba[0]))

        importance = {}
        if hasattr(self._model, "feature_importances_"):
            for name, imp in zip(self._feature_names, self._model.feature_importances_):
                importance[name] = float(imp)

        return TreePrediction(
            direction=direction,
            confidence=confidence,
            class_probabilities=class_probs,
            feature_importance=importance,
        )

    @property
    def is_fitted(self) -> bool:
        return self._fitted


class HydraTreeEnsemble:
    """Combines XGBoost + RandomForest votes — tree-based ensemble layer."""

    def __init__(self, config: Optional[TreeModelConfig] = None):
        self.xgb = HydraXGBoost(config)
        self.rf = HydraRandomForest(config)

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        xgb_metrics = self.xgb.fit(X, y, feature_names)
        rf_metrics = self.rf.fit(X, y, feature_names)
        return {"xgb_accuracy": xgb_metrics["train_accuracy"],
                "rf_accuracy": rf_metrics["train_accuracy"]}

    def predict(self, X: np.ndarray) -> TreePrediction:
        xgb_pred = self.xgb.predict(X)
        rf_pred = self.rf.predict(X)

        # Average confidence, majority-vote direction
        if xgb_pred.direction == rf_pred.direction:
            direction = xgb_pred.direction
            confidence = (xgb_pred.confidence + rf_pred.confidence) / 2
        elif xgb_pred.confidence > rf_pred.confidence:
            direction = xgb_pred.direction
            confidence = xgb_pred.confidence * 0.6
        else:
            direction = rf_pred.direction
            confidence = rf_pred.confidence * 0.6

        merged_importance = {}
        for k in set(list(xgb_pred.feature_importance.keys()) + list(rf_pred.feature_importance.keys())):
            v1 = xgb_pred.feature_importance.get(k, 0.0)
            v2 = rf_pred.feature_importance.get(k, 0.0)
            merged_importance[k] = (v1 + v2) / 2

        return TreePrediction(
            direction=direction,
            confidence=confidence,
            feature_importance=merged_importance,
        )

    @property
    def is_fitted(self) -> bool:
        return self.xgb.is_fitted and self.rf.is_fitted
