## RL² recurrent bandit playground

Train a small recurrent policy on a distribution of Bernoulli K-armed bandits, then load the checkpoint into a Streamlit playground to watch the policy adapt to fresh bandits within a single episode.

### Setup

```bash
# install deps (uv or pip)
uv pip install -e .
# or: pip install -e .
```

### Quick training run

```bash
python train.py --num-iterations 300 --steps-per-trial 50 --batch-size 32
```

Artifacts go to `artifacts/`:
- `latest.pt` + timestamped checkpoints
- `training_metrics.json` and `config.json`
- `pca.npz` (hidden state PCA used for visualization)

### Streamlit inference playground

```bash
streamlit run streamlit_app.py
```

Behavior:
- Loads `artifacts/latest.pt` if present; otherwise shows a prominent **Train** button.
- Controls to sample a new bandit, run a rollout, and compare to an ε-greedy baseline.
- Plots: action timeline, policy probabilities, cumulative reward/regret, PCA trajectory of the hidden state.
- Side tab: training curves and checkpoint config viewer.

### Minimal API surface

- Training: `rl_squared.training.train_and_save(TrainingConfig)` or CLI `python train.py`.
- Rollouts: `rl_squared.rollout.run_policy_rollout(model, bandit_probs, steps)`.
- PCA: `rl_squared.pca_utils.PCATransform.from_file("artifacts/pca.npz")`.

### Expected sanity checks

- Train → load checkpoint → sample 10 fresh bandits; policy should explore early, exploit late.
- During inference, gradients and optimizer are not touched.
- Hidden state resets when sampling a new bandit in the UI.
