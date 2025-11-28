"""Torch-based MLP probe for probing classification targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from .base import ArrayLike, BaseProbe, LabelLike, WeightsLike
from .helpers import ensure_1d_array, ensure_2d_array, ensure_sample_weights


@dataclass
class MLPProbeConfig:
    hidden_dim: int = 32
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.01


class ShallowMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.net(input)


class MLPProbe(BaseProbe):
    """Simple MLP for probing classification targets."""

    def __init__(self, config: Optional[MLPProbeConfig] = None) -> None:
        self.config = config or MLPProbeConfig()
        self.model: Optional[ShallowMLP] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self,
        features: ArrayLike,
        labels: LabelLike,
        sample_weights: WeightsLike = None,
    ) -> "MLPProbe":
        X = ensure_2d_array(features)
        y = ensure_1d_array(labels)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Feature rows ({X.shape[0]}) and label count ({y.shape[0]}) must match")

        classes, inverse = np.unique(y, return_inverse=True)
        if classes.size < 2:
            raise ValueError("MLPProbe requires at least two unique classes.")
        self.classes_ = classes

        # Setting up the dataloader
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(inverse, dtype=torch.long)

        weights = ensure_sample_weights(sample_weights, X.shape[0])
        if weights is not None:
            weight_tensor = torch.tensor(weights, dtype=torch.float32)
            dataset = TensorDataset(X_tensor, y_tensor, weight_tensor)
        else:
            dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        # Setting up the model
        _, input_dim = X_tensor.shape
        output_dim = classes.shape[0]
        self.model = ShallowMLP(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=output_dim,
        )
        loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss(reduction="none")
        optimizer: torch.optim.AdamW = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

        for epoch in range(self.config.epochs):
            self.model.train()
            for batch in data_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    X_batch, y_batch, w_batch = batch
                else:
                    X_batch, y_batch = batch
                    w_batch = None

                preds = self.model(X_batch)
                per_example_loss = loss_fn(preds, y_batch)
                if w_batch is not None:
                    total_weight = torch.sum(w_batch).item()
                    if total_weight == 0:
                        raise ValueError("Sample weights must sum to a positive value.")
                    loss = torch.sum(per_example_loss * w_batch) / total_weight
                else:
                    loss = per_example_loss.mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        return self

    def predict(self, features: ArrayLike) -> np.ndarray:
        model = self._require_model()
        X = ensure_2d_array(features)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            logits: torch.Tensor = model(X_tensor)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()

        classes = self._require_classes()
        return classes[predictions]

    def predict_proba(self, features: ArrayLike) -> np.ndarray:
        model = self._require_model()
        X = ensure_2d_array(features)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            logits = model(X_tensor)
            probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    def _require_model(self) -> ShallowMLP:
        if self.model is None:
            raise RuntimeError("MLPProbe has not been fitted yet.")
        return self.model

    def _require_classes(self) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("MLPProbe label encoder has not been initialized; call fit() first.")
        return self.classes_


__all__ = ["MLPProbe", "MLPProbeConfig"]
