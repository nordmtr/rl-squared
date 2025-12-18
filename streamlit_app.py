from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import altair as alt

from rl_squared.bandit import BanditConfig, describe_bandit, sample_single_bandit
from rl_squared.pca_utils import PCATransform, compute_pca
from rl_squared.rollout import epsilon_greedy_baseline, run_policy_rollout
from rl_squared.training import TrainingConfig, load_checkpoint, train_and_save

ARTIFACT_DIR = Path("artifacts")
CHECKPOINT_PATH = ARTIFACT_DIR / "latest.pt"
PCA_PATH = ARTIFACT_DIR / "pca.npz"
METRICS_PATH = ARTIFACT_DIR / "training_metrics.json"
CONFIG_PATH = ARTIFACT_DIR / "config.json"


def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return []


@st.cache_resource(show_spinner=False)
def cached_model():
    if not CHECKPOINT_PATH.exists():
        return None, None, None
    model, data = load_checkpoint(CHECKPOINT_PATH)
    bandit_cfg = BanditConfig(**data.get("bandit_config", {}))
    train_cfg = data.get("config", {})
    return model, bandit_cfg, train_cfg


@st.cache_resource(show_spinner=False)
def cached_pca():
    if PCA_PATH.exists():
        return PCATransform.from_file(PCA_PATH)
    return None


def ensure_bandit(bandit_cfg: BanditConfig):
    if "bandit_probs" not in st.session_state:
        st.session_state.bandit_probs = sample_single_bandit(bandit_cfg).cpu().numpy()
    return st.session_state.bandit_probs


def reset_rollout_state():
    st.session_state.pop("rollout", None)
    st.session_state.pop("ablation", None)


def run_training_from_ui(default_cfg: TrainingConfig):
    st.sidebar.header("Train model")
    with st.sidebar.expander("Training hyperparameters", expanded=False):
        num_iterations = st.slider("Iterations", 100, 1200, default_cfg.num_iterations, step=50)
        steps_per_trial = st.slider("Steps per bandit episode (T)", 50, 300, default_cfg.steps_per_trial, step=10)
        explore_default = min(default_cfg.explore_steps, steps_per_trial - 1)
        explore_steps = st.slider(
            "Exploration steps (episode 1)",
            0,
            steps_per_trial - 1,
            explore_default,
            step=5,
        )
        batch_size = st.slider("Batch size (bandits per iter)", 32, 512, default_cfg.batch_size, step=32)
        hidden_size = st.slider("Hidden size", 16, 256, default_cfg.hidden_size, step=16)
        include_episode_start = st.checkbox(
            "Include episode-start bit",
            value=default_cfg.include_episode_start,
            help="Adds a bit that is 1 only at t=0 (unless boundary marking is enabled).",
        )
        force_explore = st.checkbox(
            "Force explore phase",
            value=default_cfg.force_explore,
            help="Use fixed actions in episode 1 and train policy only on episode 2.",
        )
        explore_strategy = st.selectbox(
            "Explore strategy",
            ["round_robin", "random"],
            index=0 if default_cfg.explore_strategy == "round_robin" else 1,
            disabled=not force_explore,
        )
        mark_boundary = st.checkbox(
            "Mark boundary with signal",
            value=default_cfg.mark_boundary,
            help="Sets the episode-start bit at the explore/exploit boundary.",
        )
        reset_prev_at_boundary = st.checkbox(
            "Reset prev action/reward at boundary",
            value=default_cfg.reset_prev_at_boundary,
            help="Zeroes previous action/reward at the boundary (can hurt memory).",
        )
        use_expected_rewards = st.checkbox(
            "Use expected rewards in training",
            value=default_cfg.use_expected_rewards,
            help="Uses arm probabilities instead of Bernoulli sampling (lower variance).",
        )
        use_time_fraction = st.checkbox(
            "Use time fraction input",
            value=default_cfg.use_time_fraction,
            help="Provide a normalized timestep input to the model.",
        )
        normalize_advantages = st.checkbox(
            "Normalize advantages (per timestep)",
            value=default_cfg.normalize_advantages,
            help="Reduces variance but can wash out small signals.",
        )
        entropy_coef = st.number_input("Entropy coefficient", value=default_cfg.entropy_coef, min_value=0.0, step=0.005)
        value_coef = st.number_input("Value coefficient", value=default_cfg.value_coef, min_value=0.0, step=0.01)
        gamma = st.slider("Discount (gamma)", 0.0, 1.0, float(default_cfg.gamma), step=0.05)
        distribution = st.selectbox(
            "Bandit distribution",
            ["one_good", "beta"],
            index=0 if default_cfg.distribution == "one_good" else 1,
        )
        if distribution == "beta":
            alpha = st.number_input("Bandit alpha", value=default_cfg.alpha, min_value=0.5, step=0.5)
            beta = st.number_input("Bandit beta", value=default_cfg.beta, min_value=0.5, step=0.5)
            best_min = default_cfg.best_min
            best_max = default_cfg.best_max
            bad_min = default_cfg.bad_min
            bad_max = default_cfg.bad_max
        else:
            alpha = default_cfg.alpha
            beta = default_cfg.beta
            best_min = st.number_input("Best arm p min", value=default_cfg.best_min, min_value=0.5, max_value=1.0, step=0.05)
            best_max = st.number_input("Best arm p max", value=default_cfg.best_max, min_value=0.5, max_value=1.0, step=0.05)
            bad_min = st.number_input("Other arms p min", value=default_cfg.bad_min, min_value=0.0, max_value=0.5, step=0.05)
            bad_max = st.number_input("Other arms p max", value=default_cfg.bad_max, min_value=0.05, max_value=0.6, step=0.05)
        num_arms = st.slider("Number of arms", 2, 10, default_cfg.num_arms, step=1)
        learning_rate = st.number_input("Learning rate", value=default_cfg.learning_rate, step=1e-4, format="%.5f")

    if st.sidebar.button("Train RL² agent", type="primary"):
        cfg = TrainingConfig(
            num_arms=num_arms,
            alpha=alpha,
            beta=beta,
            distribution=distribution,
            best_min=best_min,
            best_max=best_max,
            bad_min=bad_min,
            bad_max=bad_max,
            hidden_size=hidden_size,
            steps_per_trial=steps_per_trial,
            explore_steps=explore_steps,
            include_episode_start=include_episode_start,
            force_explore=force_explore,
            explore_strategy=explore_strategy,
            mark_boundary=mark_boundary,
            reset_prev_at_boundary=reset_prev_at_boundary,
            use_time_fraction=use_time_fraction,
            normalize_advantages=normalize_advantages,
            use_expected_rewards=use_expected_rewards,
            batch_size=batch_size,
            num_iterations=num_iterations,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            gamma=gamma,
            learning_rate=learning_rate,
        )
        with st.spinner("Training on sampled bandits..."):
            train_and_save(cfg)
        st.sidebar.success("Training complete. Reloading artifacts.")
        st.cache_resource.clear()
        reset_rollout_state()
        st.rerun()


def plot_rollout(result, pca: PCATransform | None):
    st.subheader("Rollout diagnostics")
    steps = len(result.actions)
    df_actions = pd.DataFrame(
        {"t": range(steps), "action": result.actions, "reward": result.rewards, "regret": result.regret}
    )
    df_actions["cumulative_reward"] = df_actions["reward"].cumsum()

    df_probs = pd.DataFrame(result.probs)
    prob_long = df_probs.reset_index().melt(id_vars="index", var_name="arm", value_name="prob")
    prob_long = prob_long.rename(columns={"index": "t"})

    bandit_df = pd.DataFrame({"arm": range(len(result.bandit_probs)), "p": result.bandit_probs})
    best_arm, best_reward = describe_bandit(torch.tensor(result.bandit_probs))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**True bandit** — best arm {best_arm} (p={best_reward:.2f})")
        st.bar_chart(bandit_df.set_index("arm"))
    with col2:
        st.markdown("**Action timeline** (color = reward)")
        chart = (
            alt.Chart(df_actions)
            .mark_circle(size=70)
            .encode(
                x="t:Q",
                y=alt.Y("action:O", title="arm"),
                color=alt.Color("reward:Q", scale=alt.Scale(domain=[0, 1])),
                tooltip=["t", "action", "reward", "cumulative_reward"],
            )
            .interactive()
        )
        st.altair_chart(chart, width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Policy probabilities over time**")
        chart = (
            alt.Chart(prob_long)
            .mark_line()
            .encode(
                x="t:Q",
                y=alt.Y("prob:Q", scale=alt.Scale(domain=[0, 1])),
                color="arm:N",
                tooltip=["t", "arm", "prob"],
            )
            .interactive()
        )
        st.altair_chart(chart, width="stretch")
    with col4:
        st.markdown("**Cumulative reward & regret**")
        cum_df = df_actions[["t", "cumulative_reward", "regret"]].melt(
            id_vars="t", var_name="metric", value_name="value"
        )
        chart = (
            alt.Chart(cum_df)
            .mark_line()
            .encode(x="t:Q", y="value:Q", color="metric:N", tooltip=["t", "metric", "value"])
            .interactive()
        )
        st.altair_chart(chart, width="stretch")

    st.markdown("**Hidden state trajectory (PCA)**")
    hidden = np.stack(result.hidden)
    if pca is None:
        pca = compute_pca(torch.tensor(hidden), n_components=2)
    proj = pca.project(hidden)
    hidden_df = pd.DataFrame({"x": proj[:, 0], "y": proj[:, 1], "t": range(len(proj))})
    line = alt.Chart(hidden_df).mark_line(point=True).encode(
        x="x:Q", y="y:Q", color="t:Q", tooltip=["t", "x", "y"]
    )
    st.altair_chart(line.interactive(), width="stretch")
    st.caption("PCA uses saved transform when available; otherwise it is fit on this rollout.")


def plot_baseline_compare(
    bandit_probs: np.ndarray,
    policy_rewards: list[float],
    policy_regret: list[float],
    ablation_rewards: list[float] | None = None,
    ablation_regret: list[float] | None = None,
):
    steps = len(policy_rewards)
    baseline = epsilon_greedy_baseline(bandit_probs, steps=steps, epsilon=0.1)
    df = {
        "t": range(steps),
        "policy_reward": np.cumsum(policy_rewards),
        "baseline_reward": np.cumsum(baseline["rewards"]),
        "policy_regret": np.array(policy_regret),
        "baseline_regret": np.array(baseline["regret"]),
    }
    if ablation_rewards is not None and ablation_regret is not None:
        df["policy_no_memory_reward"] = np.cumsum(ablation_rewards)
        df["policy_no_memory_regret"] = np.array(ablation_regret)
    df = pd.DataFrame(df)
    reward_cols = ["policy_reward", "baseline_reward"]
    if "policy_no_memory_reward" in df:
        reward_cols.append("policy_no_memory_reward")
    reward_long = df[["t"] + reward_cols].melt(id_vars="t", var_name="agent", value_name="cum_reward")
    chart = (
        alt.Chart(reward_long)
        .mark_line()
        .encode(x="t:Q", y="cum_reward:Q", color="agent:N", tooltip=["t", "agent", "cum_reward"])
        .interactive()
    )
    st.altair_chart(chart, width="stretch")
    st.caption("ε-greedy baseline uses ε=0.1.")

    regret_cols = ["policy_regret", "baseline_regret"]
    if "policy_no_memory_regret" in df:
        regret_cols.append("policy_no_memory_regret")
    regret_long = df[["t"] + regret_cols].melt(id_vars="t", var_name="agent", value_name="regret")
    regret_chart = (
        alt.Chart(regret_long)
        .mark_line()
        .encode(x="t:Q", y="regret:Q", color="agent:N", tooltip=["t", "agent", "regret"])
        .interactive()
    )
    st.altair_chart(regret_chart, width="stretch")


def artifacts_viewer():
    st.header("Artifacts viewer")
    metrics = load_metrics()
    if not metrics:
        st.info("Train the agent to populate metrics and checkpoint metadata.")
        return
    df = pd.DataFrame(metrics)
    st.subheader("Training curves")
    metric_choice = st.multiselect(
        "Metrics to plot",
        [
            "avg_reward",
            "exploit_avg_reward",
            "regret",
            "entropy",
            "loss",
            "policy_loss",
            "value_loss",
            "best_arm_pick_rate",
            "exploit_best_arm_pick_rate",
            "policy_grad_norm",
            "rnn_grad_norm",
            "hidden_var_last",
        ],
        default=["avg_reward"],
    )
    if metric_choice:
        chart_df = df[["iteration"] + metric_choice].melt(id_vars="iteration", var_name="metric", value_name="value")
        chart = (
            alt.Chart(chart_df)
            .mark_line()
            .encode(
                x="iteration:Q",
                y="value:Q",
                color="metric:N",
                tooltip=["iteration", "metric", "value"],
            )
            .interactive()
        )
        st.altair_chart(chart, width="stretch")

    if CONFIG_PATH.exists():
        st.subheader("Checkpoint config")
        st.json(json.loads(CONFIG_PATH.read_text()))


def main():
    st.set_page_config(page_title="RL² Bandit Playground", layout="wide")
    st.title("RL²-style recurrent bandit agent")
    st.write(
        "Train once on a distribution of bandits, then sample fresh bandits to watch the recurrent policy adapt "
        "within a single episode."
    )

    model, bandit_cfg, saved_cfg = cached_model()
    default_cfg = TrainingConfig()

    run_training_from_ui(default_cfg)

    playground_tab, artifacts_tab = st.tabs(["Inference playground", "Artifacts"])

    with playground_tab:
        if model is None:
            st.warning("No checkpoint found. Train the model to enable the playground.")
            if st.button("Train with defaults", type="primary"):
                with st.spinner("Training with default hyperparameters..."):
                    train_and_save(default_cfg)
                st.cache_resource.clear()
                st.rerun()
            return

        bandit_probs = ensure_bandit(bandit_cfg)
        steps_default = saved_cfg.get("steps_per_trial", default_cfg.steps_per_trial) if saved_cfg else default_cfg.steps_per_trial
        if saved_cfg and not saved_cfg.get("force_explore", False):
            explore_steps = 0
        else:
            explore_steps = saved_cfg.get("explore_steps", 0)
        include_episode_start = (
            saved_cfg.get("include_episode_start", model.include_episode_start) if saved_cfg else model.include_episode_start
        )
        force_explore_default = saved_cfg.get("force_explore", False) if saved_cfg else False
        use_time_fraction = saved_cfg.get("use_time_fraction", True) if saved_cfg else True
        explore_strategy_default = saved_cfg.get("explore_strategy", "random") if saved_cfg else "random"
        mark_boundary = saved_cfg.get("mark_boundary", False) if saved_cfg else False
        reset_prev_at_boundary = saved_cfg.get("reset_prev_at_boundary", False) if saved_cfg else False
        if bandit_cfg.distribution == "one_good":
            dist_desc = (
                f"one_good (best U[{bandit_cfg.best_min:.2f},{bandit_cfg.best_max:.2f}], "
                f"others U[{bandit_cfg.bad_min:.2f},{bandit_cfg.bad_max:.2f}])"
            )
        else:
            dist_desc = f"beta(α={bandit_cfg.alpha:.1f}, β={bandit_cfg.beta:.1f})"
        st.subheader("Checkpoint summary")
        info_cols = st.columns(4)
        info_cols[0].metric("K", bandit_cfg.num_arms)
        info_cols[1].metric("T", steps_default)
        info_cols[2].metric("Explore steps", explore_steps)
        info_cols[3].metric("Gamma", saved_cfg.get("gamma", default_cfg.gamma) if saved_cfg else default_cfg.gamma)
        st.caption(f"Distribution: {dist_desc}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Sample new bandit"):
                st.session_state.bandit_probs = sample_single_bandit(bandit_cfg).cpu().numpy()
                reset_rollout_state()
        with col2:
            rollout_steps = st.number_input(
                "Rollout length (steps)",
                value=int(steps_default),
                min_value=10,
                max_value=400,
                step=10,
            )
        with col3:
            deterministic = st.checkbox("Greedy actions", value=False, help="Use argmax instead of sampling.")
        with col4:
            force_explore_rollout = st.checkbox(
                "Force explore phase",
                value=force_explore_default,
                help="Use random actions in the exploration segment.",
            )
            explore_strategy_rollout = st.selectbox(
                "Explore strategy",
                ["round_robin", "random"],
                index=0 if explore_strategy_default == "round_robin" else 1,
                disabled=not force_explore_rollout,
            )
            ablate_memory = st.checkbox(
                "Ablate memory",
                value=False,
                help="Reset the hidden state each step to test reliance on memory.",
            )

        if st.button("Run rollout", type="primary"):
            with st.spinner("Running policy on a fresh bandit..."):
                result = run_policy_rollout(
                    model,
                    torch.tensor(st.session_state.bandit_probs, dtype=torch.float32),
                    steps=int(rollout_steps),
                    deterministic=deterministic,
                    explore_steps=int(explore_steps) if explore_steps else None,
                    include_episode_start=include_episode_start,
                    force_explore=force_explore_rollout,
                    explore_strategy=explore_strategy_rollout,
                    mark_boundary=mark_boundary,
                    reset_prev_at_boundary=reset_prev_at_boundary,
                    use_time_fraction=use_time_fraction,
                    device=torch.device("cpu"),
                )
            st.session_state.rollout = result
            if ablate_memory:
                ablation = run_policy_rollout(
                    model,
                    torch.tensor(st.session_state.bandit_probs, dtype=torch.float32),
                    steps=int(rollout_steps),
                    deterministic=deterministic,
                    explore_steps=int(explore_steps) if explore_steps else None,
                    include_episode_start=include_episode_start,
                    force_explore=force_explore_rollout,
                    explore_strategy=explore_strategy_rollout,
                    mark_boundary=mark_boundary,
                    reset_prev_at_boundary=reset_prev_at_boundary,
                    use_time_fraction=use_time_fraction,
                    reset_hidden_each_step=True,
                    device=torch.device("cpu"),
                )
                st.session_state.ablation = ablation
            else:
                st.session_state.pop("ablation", None)

        if "rollout" in st.session_state:
            plot_rollout(st.session_state.rollout, cached_pca())
            ablation = st.session_state.get("ablation")
            plot_baseline_compare(
                st.session_state.bandit_probs,
                st.session_state.rollout.rewards,
                st.session_state.rollout.regret,
                ablation_rewards=ablation.rewards if ablation else None,
                ablation_regret=ablation.regret if ablation else None,
            )
        else:
            st.info("Sample a bandit and run a rollout to visualize adaptation.")

    with artifacts_tab:
        artifacts_viewer()


if __name__ == "__main__":
    main()
