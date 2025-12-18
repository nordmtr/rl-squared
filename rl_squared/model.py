from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PolicyValueRNN(nn.Module):
    def __init__(self, num_arms: int, hidden_size: int = 64, include_episode_start: bool = False):
        super().__init__()
        self.num_arms = num_arms
        self.hidden_size = hidden_size
        self.include_episode_start = include_episode_start
        # Input: one-hot previous action + previous reward + time fraction.
        input_dim = num_arms + 2 + (1 if include_episode_start else 0)
        self.rnn = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.policy_head = nn.Linear(hidden_size, num_arms)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # obs: [batch, input_dim]; hidden: [1, batch, hidden]
        out, new_hidden = self.rnn(obs.unsqueeze(1), hidden)
        out = out.squeeze(1)
        logits = self.policy_head(out)
        value = self.value_head(out).squeeze(-1)
        return logits, value, new_hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def act(self, obs: torch.Tensor, hidden: torch.Tensor, greedy: bool = False):
        logits, value, new_hidden = self(obs, hidden)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if greedy else dist.sample()
        log_prob = dist.log_prob(action)
        probs = dist.probs
        entropy = dist.entropy()
        return action, log_prob, value, new_hidden, probs, entropy


def build_observation(
    prev_action: torch.Tensor,
    prev_reward: torch.Tensor,
    time_fraction: torch.Tensor,
    episode_start: torch.Tensor | None = None,
    include_episode_start: bool = False,
) -> torch.Tensor:
    if include_episode_start:
        if episode_start is None:
            episode_start = torch.zeros_like(prev_reward)
        return torch.cat([prev_action, prev_reward, time_fraction, episode_start], dim=1)
    return torch.cat([prev_action, prev_reward, time_fraction], dim=1)


def one_hot(actions: torch.Tensor, num_arms: int) -> torch.Tensor:
    return F.one_hot(actions, num_classes=num_arms).float()
