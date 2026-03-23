# Continuous A2C Homework (MuJoCo)

This folder contains a continuous-action A2C implementation for Gymnasium MuJoCo environments.

## What is included

- `a2c_continuous.py`
  - Continuous policy (Gaussian + tanh squashing) for Box action spaces.
  - Three estimators:
    - `td1`: 1-step TD (equivalent to `lambda=0`).
    - `mc`: Monte Carlo style (equivalent to `lambda=1` with rollout bootstrap).
    - `gae`: Generalized Advantage Estimation (`lambda` configurable).
  - Three-environment experiment runner.
  - Grid search over:
    - number of parallel environments
    - policy learning rate
    - value learning rate
    - `gamma`
    - `lambda`
  - Learning-curve plot generation.

## Suggested environments

Use three MuJoCo locomotion tasks:

- `Hopper-v4`
- `Walker2d-v4`
- `HalfCheetah-v4`

## Install dependencies

```bash
pip install "gymnasium[mujoco]" torch matplotlib numpy
```

## Run experiments

Run full pipeline (grid search + compare curves):

```bash
python actor_acritic/a2c_continuous.py \
  --mode full \
  --envs Hopper-v4,Walker2d-v4,HalfCheetah-v4 \
  --updates 200 \
  --search-updates 80
```

Run one environment per script (full grid, non-quick):

```bash
python actor_acritic/run_hopper.py
python actor_acritic/run_walker2d.py
python actor_acritic/run_halfcheetah.py
```

Run only curve comparison (no grid search):

```bash
python actor_acritic/a2c_continuous.py --mode compare
```

Run only grid search:

```bash
python actor_acritic/a2c_continuous.py --mode grid
```

## Latest tuned hyperparameters

The following best combinations are from a rerun on:

- command mode: `--mode grid --grid-preset full`
- tuning budget: `--search-updates 20 --rollout-len 128`
- environments: `Hopper-v4,Walker2d-v4,HalfCheetah-v4`
- output: `actor_acritic/results_full_rerun`

### Hopper-v4

| Algorithm | num_envs | policy_lr | value_lr | gamma | lambda |
|:--|--:|--:|--:|--:|--:|
| td1 | 4 | 3e-4 | 1e-3 | 0.97 | 0.0 |
| mc  | 8 | 3e-4 | 5e-4 | 0.97 | 1.0 |
| gae | 8 | 3e-4 | 1e-3 | 0.97 | 0.99 |

### Walker2d-v4

| Algorithm | num_envs | policy_lr | value_lr | gamma | lambda |
|:--|--:|--:|--:|--:|--:|
| td1 | 4 | 3e-4 | 5e-4 | 0.97 | 0.0 |
| mc  | 8 | 3e-4 | 1e-3 | 0.99 | 1.0 |
| gae | 8 | 3e-4 | 5e-4 | 0.99 | 0.95 |

### HalfCheetah-v4

| Algorithm | num_envs | policy_lr | value_lr | gamma | lambda |
|:--|--:|--:|--:|--:|--:|
| td1 | 4 | 3e-4 | 1e-3 | 0.97 | 0.0 |
| mc  | 8 | 3e-4 | 5e-4 | 0.99 | 1.0 |
| gae | 4 | 3e-4 | 5e-4 | 0.99 | 0.95 |

## Output files

Default output directory: `actor_acritic/results`

- `*_learning_curves.png`: one chart per environment with all 3 algorithms.
- `*_{algo}_grid_search.json`: grid-search trials and best configuration.
- `summary.json`: overall experiment summary.
