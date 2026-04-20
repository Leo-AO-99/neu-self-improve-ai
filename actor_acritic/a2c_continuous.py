import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Literal

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam

Algorithm = Literal["td1", "mc", "gae"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class A2CConfig:
    env_id: str
    algo: Algorithm
    total_updates: int = 300
    rollout_len: int = 256
    num_envs: int = 8
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    gamma: float = 0.99
    lam: float = 0.95
    hidden_size: int = 128
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    # Time-limit truncation should bootstrap value to reduce boundary bias.
    bootstrap_time_limit: bool = True
    seed: int = 42
    device: str = "cpu"


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, action_low: np.ndarray, action_high: np.ndarray, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(action_bias, dtype=torch.float32))

    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -0.999999, 0.999999)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def _base_dist(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(torch.clamp(self.log_std, -20.0, 2.0))
        return Normal(mean, std)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._base_dist(obs)
        u = dist.rsample()
        squashed = torch.tanh(u)
        actions = self.action_bias + self.action_scale * squashed
        correction = torch.log(1.0 - squashed.pow(2) + 1e-6)
        log_prob = (dist.log_prob(u) - correction).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return actions, log_prob, entropy

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        dist = self._base_dist(obs)
        squashed = (actions - self.action_bias) / (self.action_scale + 1e-8)
        u = self._atanh(squashed)
        correction = torch.log(1.0 - squashed.pow(2) + 1e-6)
        return (dist.log_prob(u) - correction).sum(dim=-1)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self._base_dist(obs)
        return dist.entropy().sum(dim=-1)


class ValueEstimator(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_adv = np.zeros(rewards.shape[1], dtype=np.float32)

    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * mask * next_value - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv

    returns = advantages + values
    return advantages, returns


def resolve_lambda(algo: Algorithm, configured_lam: float) -> float:
    if algo == "td1":
        return 0.0
    if algo == "mc":
        return 1.0
    return configured_lam


def make_vector_env(env_id: str, num_envs: int, seed: int) -> gym.vector.VectorEnv:
    def make_fn(rank: int):
        def thunk():
            env = gym.make(env_id)
            env.reset(seed=seed + rank)
            return env

        return thunk

    return gym.vector.SyncVectorEnv([make_fn(i) for i in range(num_envs)])


def train_a2c(cfg: A2CConfig) -> dict:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    envs = make_vector_env(cfg.env_id, cfg.num_envs, cfg.seed)
    assert isinstance(envs.single_observation_space, gym.spaces.Box), "Only Box observations are supported."
    assert isinstance(envs.single_action_space, gym.spaces.Box), "Continuous action space required."

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    action_low = envs.single_action_space.low
    action_high = envs.single_action_space.high

    policy = GaussianPolicy(obs_dim, act_dim, action_low, action_high, cfg.hidden_size).to(device)
    value_fn = ValueEstimator(obs_dim, cfg.hidden_size).to(device)
    actor_opt = Adam(policy.parameters(), lr=cfg.policy_lr)
    critic_opt = Adam(value_fn.parameters(), lr=cfg.value_lr)

    obs, _ = envs.reset(seed=cfg.seed)
    episode_return_tracker = np.zeros(cfg.num_envs, dtype=np.float32)
    completed_returns: list[float] = []
    update_returns: list[float] = []
    timesteps: list[int] = []

    lam = resolve_lambda(cfg.algo, cfg.lam)

    for update in range(1, cfg.total_updates + 1):
        obs_buf = np.zeros((cfg.rollout_len, cfg.num_envs, obs_dim), dtype=np.float32)
        act_buf = np.zeros((cfg.rollout_len, cfg.num_envs, act_dim), dtype=np.float32)
        rew_buf = np.zeros((cfg.rollout_len, cfg.num_envs), dtype=np.float32)
        done_buf = np.zeros((cfg.rollout_len, cfg.num_envs), dtype=np.float32)
        value_buf = np.zeros((cfg.rollout_len, cfg.num_envs), dtype=np.float32)

        for t in range(cfg.rollout_len):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_tensor, _, _ = policy.sample(obs_tensor)
                value_tensor = value_fn(obs_tensor)

            actions_np = action_tensor.cpu().numpy()
            next_obs, rewards, terminated, truncated, infos = envs.step(actions_np)
            rewards = rewards.astype(np.float32, copy=False)

            if cfg.bootstrap_time_limit and np.any(truncated):
                final_obs = infos.get("final_observation")
                if final_obs is not None:
                    trunc_indices = np.where(truncated)[0]
                    valid_indices: list[int] = []
                    final_obs_batch: list[np.ndarray] = []
                    for idx in trunc_indices:
                        obs_i = final_obs[idx]
                        if obs_i is not None:
                            valid_indices.append(int(idx))
                            final_obs_batch.append(np.asarray(obs_i, dtype=np.float32).reshape(-1))

                    if final_obs_batch:
                        with torch.no_grad():
                            final_obs_tensor = torch.as_tensor(np.asarray(final_obs_batch), dtype=torch.float32, device=device)
                            final_v = value_fn(final_obs_tensor).cpu().numpy().astype(np.float32)
                        rewards[valid_indices] += cfg.gamma * final_v

            dones = np.logical_or(terminated, truncated).astype(np.float32)

            obs_buf[t] = obs
            act_buf[t] = actions_np
            rew_buf[t] = rewards
            done_buf[t] = dones
            value_buf[t] = value_tensor.cpu().numpy()

            episode_return_tracker += rewards
            for i, done_i in enumerate(dones):
                if done_i > 0.5:
                    completed_returns.append(float(episode_return_tracker[i]))
                    episode_return_tracker[i] = 0.0

            obs = next_obs

        with torch.no_grad():
            last_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            last_value = value_fn(last_obs_tensor).cpu().numpy()

        advantages, returns = compute_gae(
            rewards=rew_buf,
            values=value_buf,
            dones=done_buf,
            last_value=last_value,
            gamma=cfg.gamma,
            lam=lam,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.as_tensor(obs_buf.reshape(-1, obs_dim), dtype=torch.float32, device=device)
        act_t = torch.as_tensor(act_buf.reshape(-1, act_dim), dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=device)

        log_prob = policy.log_prob(obs_t, act_t)
        entropy = policy.entropy(obs_t)
        policy_loss = -(log_prob * adv_t).mean() - cfg.entropy_coef * entropy.mean()

        value_pred = value_fn(obs_t)
        value_loss = nn.functional.mse_loss(value_pred, ret_t)

        actor_opt.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
        actor_opt.step()

        critic_opt.zero_grad()
        (cfg.value_coef * value_loss).backward()
        nn.utils.clip_grad_norm_(value_fn.parameters(), cfg.max_grad_norm)
        critic_opt.step()

        recent_returns = completed_returns[-20:] if completed_returns else [0.0]
        avg_recent = float(np.mean(recent_returns))
        update_returns.append(avg_recent)
        timesteps.append(update * cfg.rollout_len * cfg.num_envs)

        print(
            f"[{cfg.env_id}][{cfg.algo}] "
            f"update {update:04d}/{cfg.total_updates}, "
            f"avg_return(last20)={avg_recent:.2f}, "
            f"policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}"
        )

    envs.close()
    score = float(np.mean(update_returns[-10:])) if update_returns else -math.inf
    return {
        "config": asdict(cfg),
        "timesteps": timesteps,
        "learning_curve": update_returns,
        "episode_returns": completed_returns,
        "score": score,
    }


def plot_env_curves(env_id: str, all_results: dict[Algorithm, dict], output_dir: Path) -> None:
    plt.figure(figsize=(9, 5))
    for algo, result in all_results.items():
        plt.plot(result["timesteps"], result["learning_curve"], label=algo)

    plt.title(f"A2C Variants on {env_id}")
    plt.xlabel("Environment steps")
    plt.ylabel("Average episodic return (last 20 episodes)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / f"{env_id}_learning_curves.png"
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_three_algorithms(env_id: str, base_cfg: A2CConfig, output_dir: Path) -> dict[Algorithm, dict]:
    results: dict[Algorithm, dict] = {}
    for algo in ("td1", "mc", "gae"):
        cfg = A2CConfig(**asdict(base_cfg))
        cfg.env_id = env_id
        cfg.algo = algo
        results[algo] = train_a2c(cfg)

    plot_env_curves(env_id, results, output_dir)
    return results


def run_grid_search(
    env_id: str,
    algo: Algorithm,
    base_cfg: A2CConfig,
    output_dir: Path,
    grid: dict[str, list[float]],
    search_updates: int,
) -> dict:
    best_result: dict | None = None
    trials: list[dict] = []

    lam_candidates = grid["lam"]
    if algo == "td1":
        lam_candidates = [0.0]
    elif algo == "mc":
        lam_candidates = [1.0]

    keys = ["num_envs", "policy_lr", "value_lr", "gamma", "lam"]
    values = [grid["num_envs"], grid["policy_lr"], grid["value_lr"], grid["gamma"], lam_candidates]
    total_trials = int(np.prod([len(v) for v in values]))
    print(f"[Grid][{env_id}][{algo}] total unique trials: {total_trials}")

    for trial_id, (num_envs, policy_lr, value_lr, gamma, lam) in enumerate(product(*values), start=1):
        cfg = A2CConfig(**asdict(base_cfg))
        cfg.env_id = env_id
        cfg.algo = algo
        cfg.total_updates = search_updates
        cfg.num_envs = int(num_envs)
        cfg.policy_lr = float(policy_lr)
        cfg.value_lr = float(value_lr)
        cfg.gamma = float(gamma)
        cfg.lam = float(lam)

        print(
            f"[Grid][{env_id}][{algo}] trial {trial_id}/{total_trials}: "
            f"num_envs={cfg.num_envs}, policy_lr={cfg.policy_lr}, value_lr={cfg.value_lr}, "
            f"gamma={cfg.gamma}, lambda={cfg.lam}"
        )
        result = train_a2c(cfg)
        trial_record = {
            "score": result["score"],
            "config": result["config"],
        }
        trials.append(trial_record)
        if best_result is None or result["score"] > best_result["score"]:
            best_result = result

    assert best_result is not None, "Grid search must execute at least one trial."
    payload = {
        "env_id": env_id,
        "algo": algo,
        "best_score": best_result["score"],
        "best_config": best_result["config"],
        "all_trials": trials,
    }
    out_file = output_dir / f"{env_id}_{algo}_grid_search.json"
    out_file.write_text(json.dumps(payload, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A2C continuous-action homework runner.")
    parser.add_argument(
        "--envs",
        type=str,
        default="Hopper-v5,Walker2d-v4,HalfCheetah-v4",
        help="Comma-separated MuJoCo env ids.",
    )
    parser.add_argument("--mode", type=str, choices=["compare", "grid", "full"], default="full")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--search-updates", type=int, default=80)
    parser.add_argument("--rollout-len", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--grid-preset",
        type=str,
        choices=["quick", "full"],
        default="full",
        help="quick: smaller search space for fast tuning; full: larger search space.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_ids = [x.strip() for x in args.envs.split(",") if x.strip()]
    base_cfg = A2CConfig(
        env_id=env_ids[0],
        algo="gae",
        total_updates=args.updates,
        rollout_len=args.rollout_len,
        num_envs=args.num_envs,
        policy_lr=args.policy_lr,
        value_lr=args.value_lr,
        gamma=args.gamma,
        lam=args.lam,
        seed=args.seed,
    )

    if args.grid_preset == "quick":
        grid = {
            "num_envs": [4, 8],
            "policy_lr": [1e-4, 3e-4],
            "value_lr": [5e-4, 1e-3],
            "gamma": [0.99],
            "lam": [0.95],
        }
    else:
        grid = {
            "num_envs": [4, 8],
            "policy_lr": [1e-4, 3e-4],
            "value_lr": [5e-4, 1e-3],
            "gamma": [0.97, 0.99],
            "lam": [0.9, 0.95, 0.99],
        }

    summary: dict[str, dict] = {}
    for env_id in env_ids:
        print(f"\n========== Running environment: {env_id} ==========")
        summary[env_id] = {}

        if args.mode in ("grid", "full"):
            summary[env_id]["grid_search"] = {}
            for algo in ("td1", "mc", "gae"):
                grid_result = run_grid_search(
                    env_id=env_id,
                    algo=algo,
                    base_cfg=base_cfg,
                    output_dir=output_dir,
                    grid=grid,
                    search_updates=args.search_updates,
                )
                summary[env_id]["grid_search"][algo] = grid_result["best_config"]

        if args.mode in ("compare", "full"):
            if args.mode == "full":
                per_algo_best: dict[Algorithm, dict] = {}
                for algo in ("td1", "mc", "gae"):
                    best_cfg_dict = summary[env_id]["grid_search"][algo]
                    cfg = A2CConfig(**best_cfg_dict)
                    cfg.total_updates = args.updates
                    cfg.seed = args.seed
                    per_algo_best[algo] = train_a2c(cfg)
                plot_env_curves(env_id, per_algo_best, output_dir)
                summary[env_id]["compare"] = {k: v["score"] for k, v in per_algo_best.items()}
            else:
                compare_results = run_three_algorithms(env_id, base_cfg, output_dir)
                summary[env_id]["compare"] = {k: v["score"] for k, v in compare_results.items()}

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nAll done. Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
