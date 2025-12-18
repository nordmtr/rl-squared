from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch


@dataclass
class PCATransform:
    mean: np.ndarray
    components: np.ndarray

    def project(self, data: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        centered = data - self.mean
        return centered @ self.components.T

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "components": self.components}

    @classmethod
    def from_file(cls, path: Path) -> "PCATransform":
        data = np.load(path)
        return cls(mean=data["mean"], components=data["components"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, components=self.components)


def compute_pca(hidden_states: torch.Tensor, n_components: int = 2) -> PCATransform:
    """Compute a compact PCA transform without sklearn."""
    if hidden_states.dim() != 2:
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    hidden_states = hidden_states.detach().cpu()
    mean = hidden_states.mean(dim=0, keepdim=True)
    centered = hidden_states - mean
    u, s, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:n_components]
    return PCATransform(mean=mean.squeeze(0).numpy(), components=components.numpy())
