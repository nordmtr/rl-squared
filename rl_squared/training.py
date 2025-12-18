from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn, optim
from tqdm import trange

from .bandit import BanditConfig, best_expected_reward, expected_reward, pull_arm, sample_bandit_probs
from .model import PolicyValueRNN, build_observation, one_hot
from .pca_utils import compute_pca
from .rollout import run_policy_rollout


@dataclass
class TrainingConfig:
    num_arms: int = 5
    alpha: float = 3.0
    beta: float = 3.0
    distribution: str = "one_good"
    best_min: float = 0.9
    best_max: float = 1.0
    bad_min: float = 0.0
    bad_max: float = 0.1
    hidden_size: int = 64
    steps_per_trial: int = 150
    explore_steps: int = 20
    include_episode_start: bool = True
    force_explore: bool = True
    explore_strategy: str = "round_robin"
    use_time_fraction: bool = False
    normalize_advantages: bool = False
    use_expected_rewards: bool = True
    batch_size: int = 256
    num_iterations: int = 500
    gamma: float = 0.0
    entropy_coef: float = 0.0
    value_coef: float = 0.03
    learning_rate: float = 1e-3
    grad_clip: float = 5.0
    device: str = "cpu"
    seed: int = 0
    save_dir: str = "artifacts"
    pca_rollouts: int = 50


def set_seed(seed: int):
    torch.manual_seed(seed)


def train(config: TrainingConfig) -> Tuple[PolicyValueRNN, List[Dict[str, float]]]:
    device = torch.device(config.device)
    set_seed(config.seed)

    bandit_cfg = BanditConfig(
        num_arms=config.num_arms,
        alpha=config.alpha,
        beta=config.beta,
        distribution=config.distribution,
        best_min=config.best_min,
        best_max=config.best_max,
        bad_min=config.bad_min,
        bad_max=config.bad_max,
    )
    model = PolicyValueRNN(
        num_arms=config.num_arms,
        hidden_size=config.hidden_size,
        include_episode_start=config.include_episode_start,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    metrics: List[Dict[str, float]] = []

    for it in trange(config.num_iterations, desc="training"):
        optimizer.zero_grad()
        bandit_probs = sample_bandit_probs(config.batch_size, bandit_cfg, device=device)
        hidden = model.init_hidden(config.batch_size, device)
        prev_action = torch.zeros(config.batch_size, config.num_arms, device=device)
        prev_reward = torch.zeros(config.batch_size, 1, device=device)
        boundary_step = None
        if 0 < config.explore_steps < config.steps_per_trial:
            boundary_step = config.explore_steps
        if boundary_step is not None and config.force_explore and config.explore_strategy == "round_robin":
            explore_offsets = torch.randint(0, config.num_arms, (config.batch_size,), device=device)
        else:
            explore_offsets = None

        log_probs = []
        values = []
        entropies = []
        rewards_per_step = []
        best_arm = bandit_probs.argmax(dim=1)
        last_half_start = config.steps_per_trial // 2
        best_arm_hits = 0
        best_arm_total = 0
        exploit_hits = 0
        exploit_total = 0

        for t in range(config.steps_per_trial):
            episode_start = torch.zeros(config.batch_size, 1, device=device)
            if t == 0:
                episode_start = torch.ones(config.batch_size, 1, device=device)
            if boundary_step is not None and t == boundary_step:
                prev_action = torch.zeros_like(prev_action)
                prev_reward = torch.zeros_like(prev_reward)
                episode_start = torch.ones(config.batch_size, 1, device=device)

            if config.use_time_fraction:
                if boundary_step is None:
                    time_fraction_value = float(t) / float(max(1, config.steps_per_trial - 1))
                elif t < boundary_step:
                    time_fraction_value = float(t) / float(max(1, boundary_step - 1))
                else:
                    time_fraction_value = float(t - boundary_step) / float(
                        max(1, config.steps_per_trial - boundary_step - 1)
                    )
            else:
                time_fraction_value = 0.0

            time_fraction = torch.full((config.batch_size, 1), time_fraction_value, device=device)
            obs = build_observation(
                prev_action,
                prev_reward,
                time_fraction,
                episode_start=episode_start,
                include_episode_start=config.include_episode_start,
            )
            logits, value_t, hidden = model(obs, hidden)
            dist = torch.distributions.Categorical(logits=logits)
            if boundary_step is not None and config.force_explore and t < boundary_step:
                if config.explore_strategy == "round_robin":
                    offsets = explore_offsets if explore_offsets is not None else 0
                    actions = (offsets + t) % config.num_arms
                else:
                    actions = torch.randint(0, config.num_arms, (config.batch_size,), device=device)
            else:
                actions = dist.sample()
            if config.use_expected_rewards:
                rewards = expected_reward(bandit_probs, actions)
            else:
                rewards = pull_arm(bandit_probs, actions).float()

            log_probs.append(dist.log_prob(actions))
            values.append(value_t)
            entropies.append(dist.entropy())
            rewards_per_step.append(rewards)

            if t >= last_half_start:
                best_arm_hits += (actions == best_arm).sum().item()
                best_arm_total += actions.numel()
            if boundary_step is not None and t >= boundary_step:
                exploit_hits += (actions == best_arm).sum().item()
                exploit_total += actions.numel()

            prev_action = one_hot(actions, config.num_arms)
            prev_reward = rewards.unsqueeze(1)

        log_probs = torch.stack(log_probs)  # [T, B]
        values = torch.stack(values)  # [T, B]
        entropies = torch.stack(entropies)  # [T, B]
        rewards_tensor = torch.stack(rewards_per_step)  # [T, B]

        rewards_for_returns = rewards_tensor
        if boundary_step is not None and config.force_explore:
            rewards_for_returns = rewards_tensor.clone()
            rewards_for_returns[:boundary_step] = 0.0

        returns = torch.zeros_like(rewards_for_returns)
        running = torch.zeros(config.batch_size, device=device)
        for t in reversed(range(config.steps_per_trial)):
            running = rewards_for_returns[t] + config.gamma * running
            returns[t] = running

        advantages = returns - values.detach()
        if config.normalize_advantages:
            # Normalize per timestep across the batch to keep exploration signal intact.
            adv_mean = advantages.mean(dim=1, keepdim=True)
            adv_std = advantages.std(dim=1, keepdim=True) + 1e-6
            advantages = (advantages - adv_mean) / adv_std

        if boundary_step is not None and config.force_explore:
            mask = torch.zeros_like(advantages)
            mask[boundary_step:] = 1.0
        else:
            mask = torch.ones_like(advantages)
        mask_sum = mask.sum().clamp(min=1.0)

        policy_loss = -((log_probs * advantages) * mask).sum() / mask_sum
        value_loss = 0.5 * (((returns - values).pow(2)) * mask).sum() / mask_sum
        entropy_term = (entropies * mask).sum() / mask_sum

        loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_term
        loss.backward()
        policy_grad_norm = 0.0
        rnn_grad_norm = 0.0
        if model.policy_head.weight.grad is not None:
            policy_grad_norm = model.policy_head.weight.grad.norm().item()
        if model.rnn.weight_ih_l0.grad is not None:
            rnn_grad_norm = model.rnn.weight_ih_l0.grad.norm().item()
        if config.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        with torch.no_grad():
            total_reward = rewards_tensor.sum(dim=0)
            avg_reward = total_reward.mean().item() / config.steps_per_trial
            if boundary_step is not None:
                exploit_rewards = rewards_tensor[boundary_step:].mean().item()
            else:
                exploit_rewards = avg_reward
            expected_best = (best_expected_reward(bandit_probs) * config.steps_per_trial).mean().item()
            regret = expected_best - total_reward.mean().item()
            best_arm_pick_rate = best_arm_hits / max(1, best_arm_total)
            if exploit_total > 0:
                exploit_pick_rate = exploit_hits / exploit_total
            else:
                exploit_pick_rate = best_arm_pick_rate
            hidden_var_last = hidden.squeeze(0).var(dim=0).mean().item()
            metrics.append(
                {
                    "iteration": it,
                    "avg_reward": avg_reward,
                    "exploit_avg_reward": exploit_rewards,
                    "entropy": entropy_term.item(),
                    "regret": regret,
                    "loss": loss.item(),
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "best_arm_pick_rate": best_arm_pick_rate,
                    "exploit_best_arm_pick_rate": exploit_pick_rate,
                    "policy_grad_norm": policy_grad_norm,
                    "rnn_grad_norm": rnn_grad_norm,
                    "hidden_var_last": hidden_var_last,
                }
            )

    return model, metrics


def collect_hidden_states(model: PolicyValueRNN, bandit_cfg: BanditConfig, config: TrainingConfig) -> torch.Tensor:
    hidden_states = []
    device = torch.device(config.device)
    for _ in range(config.pca_rollouts):
        bandit = sample_bandit_probs(1, bandit_cfg, device=device).squeeze(0)
        rollout = run_policy_rollout(
            model,
            bandit,
            steps=config.steps_per_trial,
            deterministic=False,
            explore_steps=config.explore_steps,
            include_episode_start=config.include_episode_start,
            force_explore=config.force_explore,
            explore_strategy=config.explore_strategy,
            use_time_fraction=config.use_time_fraction,
            device=device,
        )
        hidden_states.append(torch.tensor(rollout.hidden, device=device))
    return torch.cat(hidden_states, dim=0)


def save_artifacts(
    model: PolicyValueRNN,
    metrics: List[Dict[str, float]],
    config: TrainingConfig,
    bandit_cfg: BanditConfig,
) -> Dict[str, Path]:
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    latest_path = save_dir / "latest.pt"
    stamped_path = save_dir / f"checkpoint-{timestamp}.pt"
    info = {"config": asdict(config), "bandit_config": asdict(bandit_cfg)}

    torch.save({"model_state_dict": model.state_dict(), **info}, latest_path)
    torch.save({"model_state_dict": model.state_dict(), **info}, stamped_path)

    metrics_path = save_dir / "training_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    config_path = save_dir / "config.json"
    with config_path.open("w") as f:
        json.dump(info, f, indent=2)

    return {"latest": latest_path, "timestamped": stamped_path, "metrics": metrics_path, "config": config_path}


def save_pca_transform(model: PolicyValueRNN, config: TrainingConfig, bandit_cfg: BanditConfig) -> Path:
    hidden_states = collect_hidden_states(model, bandit_cfg, config)
    pca = compute_pca(hidden_states, n_components=2)
    pca_path = Path(config.save_dir) / "pca.npz"
    pca.save(pca_path)
    return pca_path


def load_checkpoint(checkpoint_path: Path) -> Tuple[PolicyValueRNN, Dict]:
    data = torch.load(checkpoint_path, map_location="cpu")
    config = data.get("config", {})
    num_arms = config.get("num_arms", data.get("bandit_config", {}).get("num_arms", 5))
    hidden_size = config.get("hidden_size", 64)
    include_episode_start = config.get("include_episode_start", False)
    model = PolicyValueRNN(
        num_arms=num_arms,
        hidden_size=hidden_size,
        include_episode_start=include_episode_start,
    )
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    return model, data


def train_and_save(config: TrainingConfig) -> Dict[str, Path]:
    bandit_cfg = BanditConfig(
        num_arms=config.num_arms,
        alpha=config.alpha,
        beta=config.beta,
        distribution=config.distribution,
        best_min=config.best_min,
        best_max=config.best_max,
        bad_min=config.bad_min,
        bad_max=config.bad_max,
    )
    model, metrics = train(config)
    artifact_paths = save_artifacts(model, metrics, config, bandit_cfg)
    pca_path = save_pca_transform(model, config, bandit_cfg)
    artifact_paths["pca"] = pca_path
    return artifact_paths


def default_config() -> TrainingConfig:
    return TrainingConfig()


def main():
    cfg = default_config()
    artifacts = train_and_save(cfg)
    print(f"Saved artifacts to {cfg.save_dir}:")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
