from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from .bandit import pull_arm
from .model import build_observation, one_hot, PolicyValueRNN


@dataclass
class RolloutResult:
    actions: List[int]
    rewards: List[float]
    probs: List[np.ndarray]
    hidden: List[np.ndarray]
    values: List[float]
    bandit_probs: np.ndarray
    regret: List[float]


def run_policy_rollout(
    model: PolicyValueRNN,
    bandit_probs: torch.Tensor,
    steps: int,
    deterministic: bool = False,
    explore_steps: int | None = None,
    include_episode_start: bool | None = None,
    force_explore: bool = False,
    explore_strategy: str = "random",
    mark_boundary: bool = False,
    reset_prev_at_boundary: bool = False,
    use_time_fraction: bool | None = None,
    reset_hidden_each_step: bool = False,
    device: Optional[torch.device] = None,
) -> RolloutResult:
    device = device or next(model.parameters()).device
    model.eval()
    bandit_probs = bandit_probs.to(device)
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    prev_action = torch.zeros(batch_size, model.num_arms, device=device)
    prev_reward = torch.zeros(batch_size, 1, device=device)
    boundary_step = None
    if explore_steps is not None and 0 < explore_steps < steps:
        boundary_step = explore_steps
    if include_episode_start is None:
        include_episode_start = model.include_episode_start
    if use_time_fraction is None:
        use_time_fraction = True
    explore_offsets = None
    if boundary_step is not None and force_explore and explore_strategy == "round_robin":
        explore_offsets = torch.randint(0, model.num_arms, (batch_size,), device=device)

    actions: List[int] = []
    rewards: List[float] = []
    probs_list: List[np.ndarray] = []
    hidden_list: List[np.ndarray] = []
    values: List[float] = []
    regret: List[float] = []

    best = bandit_probs.max().item()
    cumulative_reward = 0.0

    with torch.no_grad():
        for t in range(steps):
            if reset_hidden_each_step:
                hidden = model.init_hidden(batch_size, device)
            episode_start = torch.zeros(batch_size, 1, device=device)
            if t == 0:
                episode_start = torch.ones(batch_size, 1, device=device)
            if boundary_step is not None and t == boundary_step:
                if reset_prev_at_boundary:
                    prev_action = torch.zeros_like(prev_action)
                    prev_reward = torch.zeros_like(prev_reward)
                if mark_boundary:
                    episode_start = torch.ones(batch_size, 1, device=device)

            if use_time_fraction:
                if boundary_step is None:
                    time_fraction_value = float(t) / float(max(1, steps - 1))
                elif t < boundary_step:
                    time_fraction_value = float(t) / float(max(1, boundary_step - 1))
                else:
                    time_fraction_value = float(t - boundary_step) / float(max(1, steps - boundary_step - 1))
            else:
                time_fraction_value = 0.0
            time_fraction = torch.full((batch_size, 1), time_fraction_value, device=device)
            obs = build_observation(
                prev_action,
                prev_reward,
                time_fraction,
                episode_start=episode_start,
                include_episode_start=include_episode_start,
            )
            if boundary_step is not None and force_explore and t < boundary_step:
                logits, value, hidden = model(obs, hidden)
                dist = torch.distributions.Categorical(logits=logits)
                if explore_strategy == "round_robin":
                    offsets = explore_offsets if explore_offsets is not None else 0
                    action = (offsets + t) % model.num_arms
                else:
                    action = torch.randint(0, model.num_arms, (batch_size,), device=device)
                probs = dist.probs
            else:
                action, log_prob, value, hidden, probs, entropy = model.act(obs, hidden, greedy=deterministic)
            reward = pull_arm(bandit_probs.unsqueeze(0), action).float()

            cumulative_reward += reward.item()
            regret.append(best * (t + 1) - cumulative_reward)

            actions.append(int(action.item()))
            rewards.append(float(reward.item()))
            probs_list.append(probs.squeeze(0).cpu().numpy())
            hidden_list.append(hidden.squeeze(0).squeeze(0).cpu().numpy())
            values.append(float(value.item()))

            prev_action = one_hot(action, model.num_arms).float()
            prev_reward = reward.unsqueeze(1)

    return RolloutResult(
        actions=actions,
        rewards=rewards,
        probs=probs_list,
        hidden=hidden_list,
        values=values,
        bandit_probs=bandit_probs.cpu().numpy(),
        regret=regret,
    )


def epsilon_greedy_baseline(bandit_probs: np.ndarray, steps: int, epsilon: float = 0.1, seed: int | None = None):
    rng = np.random.default_rng(seed)
    num_arms = bandit_probs.shape[0]
    counts = np.zeros(num_arms, dtype=int)
    rewards = np.zeros(num_arms, dtype=float)

    actions: List[int] = []
    reward_list: List[float] = []
    cumulative_reward = 0.0
    regret: List[float] = []
    best = float(np.max(bandit_probs))

    for t in range(steps):
        if rng.random() < epsilon or counts.sum() == 0:
            action = rng.integers(0, num_arms)
        else:
            avg_rewards = np.divide(
                rewards,
                np.maximum(counts, 1),
                out=np.zeros_like(rewards),
                where=counts > 0,
            )
            action = int(np.argmax(avg_rewards))
        reward = float(rng.random() < bandit_probs[action])
        counts[action] += 1
        rewards[action] += reward

        actions.append(action)
        reward_list.append(reward)
        cumulative_reward += reward
        regret.append(best * (t + 1) - cumulative_reward)

    return {
        "actions": actions,
        "rewards": reward_list,
        "regret": regret,
    }
