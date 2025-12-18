from __future__ import annotations

import argparse

from rl_squared.training import TrainingConfig, train_and_save


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train an RL^2 recurrent bandit agent.")
    parser.add_argument("--num-arms", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--distribution", type=str, default="one_good", choices=["beta", "one_good"])
    parser.add_argument("--best-min", type=float, default=0.7)
    parser.add_argument("--best-max", type=float, default=1.0)
    parser.add_argument("--bad-min", type=float, default=0.0)
    parser.add_argument("--bad-max", type=float, default=0.3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--steps-per-trial", type=int, default=150)
    parser.add_argument("--explore-steps", type=int, default=30)
    parser.add_argument("--no-episode-start", action="store_true", help="Disable episode start signal bit.")
    parser.add_argument("--no-force-explore", action="store_true", help="Disable forced random exploration phase.")
    parser.add_argument("--explore-strategy", type=str, default="round_robin", choices=["random", "round_robin"])
    parser.add_argument("--mark-boundary", action="store_true", help="Set the episode-start bit at the boundary.")
    parser.add_argument("--reset-prev-at-boundary", action="store_true", help="Zero prev action/reward at the boundary.")
    parser.add_argument("--use-time-fraction", action="store_true", help="Include time fraction input feature.")
    parser.add_argument("--normalize-advantages", action="store_true", help="Normalize advantages per timestep.")
    parser.add_argument("--no-expected-rewards", action="store_true", help="Use Bernoulli rewards during training.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-iterations", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--value-coef", type=float, default=0.03)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="artifacts")
    parser.add_argument("--pca-rollouts", type=int, default=50)
    args = parser.parse_args()
    return TrainingConfig(
        num_arms=args.num_arms,
        alpha=args.alpha,
        beta=args.beta,
        distribution=args.distribution,
        best_min=args.best_min,
        best_max=args.best_max,
        bad_min=args.bad_min,
        bad_max=args.bad_max,
        hidden_size=args.hidden_size,
        steps_per_trial=args.steps_per_trial,
        explore_steps=args.explore_steps,
        include_episode_start=not args.no_episode_start,
        force_explore=not args.no_force_explore,
        explore_strategy=args.explore_strategy,
        mark_boundary=args.mark_boundary,
        reset_prev_at_boundary=args.reset_prev_at_boundary,
        use_time_fraction=args.use_time_fraction,
        normalize_advantages=args.normalize_advantages,
        use_expected_rewards=not args.no_expected_rewards,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        device=args.device,
        seed=args.seed,
        save_dir=args.save_dir,
        pca_rollouts=args.pca_rollouts,
    )


def main():
    cfg = parse_args()
    paths = train_and_save(cfg)
    print("Training complete. Artifacts:")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
