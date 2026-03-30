# Poly-PPO Homework (Four Rooms)

This folder contains the submission version for one Table-1 environment:

- `MiniGrid-FourRooms-v0` (Four Rooms)
- algorithms: REINFORCE, PPO, Poly-PPO (no UCB variants)

## Install

```bash
pip install torch gymnasium minigrid numpy
```

## Run (Homework)

```bash
./ppo/run_fourrooms_homework.sh 250 8 42
```

This runs all three algorithms with the current best homework config and writes:

- `ppo/results_fourrooms_enhanced.json`

It also prints a compact table:

- `(avg_reward, success_rate%)` for REINFORCE / PPO / Poly-PPO.

## Summarize Existing Result

```bash
python ppo/summarize_results.py --input ppo/results_fourrooms_enhanced.json
```

| Algorithm | (avg_reward, success_rate%) |
| --- | --- |
| REINFORCE | (0.245, 29.0) |
| PPO | (0.418, 53.0) |
| Poly-PPO | (0.417, 54.0) |