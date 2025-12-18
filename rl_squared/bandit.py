from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class BanditConfig:
    num_arms: int = 5
    alpha: float = 3.0
    beta: float = 3.0
    distribution: str = "beta"
    best_min: float = 0.9
    best_max: float = 1.0
    bad_min: float = 0.0
    bad_max: float = 0.1


def sample_bandit_probs(batch_size: int, config: BanditConfig, device: torch.device) -> torch.Tensor:
    """Sample Bernoulli arm probabilities for a batch of bandits."""
    if config.distribution == "one_good":
        probs = torch.empty(batch_size, config.num_arms, device=device)
        best_arm = torch.randint(0, config.num_arms, (batch_size,), device=device)
        p_best = torch.empty(batch_size, device=device).uniform_(config.best_min, config.best_max)
        p_bad = torch.empty(batch_size, config.num_arms, device=device).uniform_(config.bad_min, config.bad_max)
        probs.copy_(p_bad)
        probs[torch.arange(batch_size, device=device), best_arm] = p_best
        return probs
    if config.distribution != "beta":
        raise ValueError(f"Unknown bandit distribution: {config.distribution}")
    beta = torch.distributions.Beta(config.alpha, config.beta)
    return beta.sample((batch_size, config.num_arms)).to(device)


def pull_arm(probs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Sample Bernoulli reward for chosen actions."""
    chosen = probs.gather(1, action.unsqueeze(1)).squeeze(1)
    reward = torch.bernoulli(chosen)
    return reward


def expected_reward(probs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Return expected reward for chosen actions (no sampling)."""
    return probs.gather(1, action.unsqueeze(1)).squeeze(1)


def best_expected_reward(probs: torch.Tensor) -> torch.Tensor:
    """Return expected reward of the optimal arm for each bandit."""
    return probs.max(dim=1).values


def sample_single_bandit(config: BanditConfig, device: torch.device | None = None) -> torch.Tensor:
    device = device or torch.device("cpu")
    return sample_bandit_probs(1, config, device=device).squeeze(0)


def describe_bandit(probs: torch.Tensor) -> Tuple[int, float]:
    """Return (best_arm_index, best_expected_reward) for display."""
    best_arm = int(torch.argmax(probs).item())
    best_reward = float(torch.max(probs).item())
    return best_arm, best_reward
