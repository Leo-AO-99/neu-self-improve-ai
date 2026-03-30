import argparse
import copy
import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ObsAdapter(gym.ObservationWrapper):
    """Convert dict observations to flattened float vectors."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        sample_obs, _ = self.env.reset(seed=0)
        vec = self._to_vector(sample_obs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=vec.shape,
            dtype=np.float32,
        )

    def _to_vector(self, obs) -> np.ndarray:
        if isinstance(obs, dict):
            if "image" in obs:
                arr = obs["image"].astype(np.float32).reshape(-1)
            else:
                chunks = []
                for _, value in sorted(obs.items(), key=lambda x: x[0]):
                    if np.isscalar(value):
                        chunks.append(np.array([value], dtype=np.float32))
                    else:
                        chunks.append(np.asarray(value, dtype=np.float32).reshape(-1))
                arr = np.concatenate(chunks, axis=0)
        else:
            arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        return arr

    def observation(self, observation):
        return self._to_vector(observation)


def make_env(env_id: str, seed: int) -> gym.Env:
    import minigrid  # noqa: F401

    env = gym.make(env_id)
    env = ObsAdapter(env)
    env.reset(seed=seed)
    if isinstance(env.action_space, gym.spaces.Discrete):
        return env
    raise ValueError(f"Only discrete action spaces are supported. Got {env.action_space}.")


def _dir_to_vec(dir_idx: int) -> tuple[int, int]:
    # MiniGrid convention: 0:right, 1:down, 2:left, 3:up
    lookup = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    return lookup[dir_idx % 4]


def _is_walkable_cell(cell) -> bool:
    if cell is None:
        return True
    cell_type = getattr(cell, "type", "")
    return cell_type not in {"wall", "lava"}


def _find_goal_pos(inner) -> Optional[tuple[int, int]]:
    if hasattr(inner, "goal_pos") and inner.goal_pos is not None:
        return tuple(int(v) for v in inner.goal_pos)
    width = int(inner.width)
    height = int(inner.height)
    for gx in range(width):
        for gy in range(height):
            cell = inner.grid.get(gx, gy)
            if getattr(cell, "type", "") == "goal":
                return (gx, gy)
    return None


def _grid_shortest_distance_to_goal(env: gym.Env) -> float:
    inner = env.unwrapped
    if not hasattr(inner, "grid") or not hasattr(inner, "agent_pos"):
        return float("inf")
    goal_pos = _find_goal_pos(inner)
    if goal_pos is None:
        return float("inf")
    sx, sy = tuple(int(v) for v in inner.agent_pos)
    if (sx, sy) == goal_pos:
        return 0.0

    width = int(inner.width)
    height = int(inner.height)
    q = deque([(sx, sy, 0)])
    visited = {(sx, sy)}
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while q:
        x, y, d = q.popleft()
        for dx, dy in steps:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if (nx, ny) in visited:
                continue
            cell = inner.grid.get(nx, ny)
            if not _is_walkable_cell(cell):
                continue
            if (nx, ny) == goal_pos:
                return float(d + 1)
            visited.add((nx, ny))
            q.append((nx, ny, d + 1))
    return float("inf")


def _shortest_action_plan_to_goal(env: gym.Env) -> list[int]:
    inner = env.unwrapped
    if not hasattr(inner, "grid"):
        return []
    goal_pos = _find_goal_pos(inner)
    if goal_pos is None:
        return []
    start_pos = tuple(int(v) for v in inner.agent_pos)
    start_dir = int(inner.agent_dir)
    width = int(inner.width)
    height = int(inner.height)

    start = (start_pos[0], start_pos[1], start_dir)
    q = deque([start])
    prev: dict[tuple[int, int, int], tuple[tuple[int, int, int], int]] = {}
    visited = {start}
    goal_state: Optional[tuple[int, int, int]] = None

    while q:
        x, y, d = q.popleft()
        if (x, y) == goal_pos:
            goal_state = (x, y, d)
            break

        left_state = (x, y, (d - 1) % 4)
        if left_state not in visited:
            visited.add(left_state)
            prev[left_state] = ((x, y, d), 0)
            q.append(left_state)

        right_state = (x, y, (d + 1) % 4)
        if right_state not in visited:
            visited.add(right_state)
            prev[right_state] = ((x, y, d), 1)
            q.append(right_state)

        dx, dy = _dir_to_vec(d)
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            cell = inner.grid.get(nx, ny)
            if _is_walkable_cell(cell):
                fwd_state = (nx, ny, d)
                if fwd_state not in visited:
                    visited.add(fwd_state)
                    prev[fwd_state] = ((x, y, d), 2)
                    q.append(fwd_state)

    if goal_state is None:
        return []

    actions_rev = []
    cur = goal_state
    while cur != start:
        parent, act = prev[cur]
        actions_rev.append(act)
        cur = parent
    actions_rev.reverse()
    return actions_rev


def discounted_cumsum(values: list[float], gamma: float) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(values))):
        running = values[i] + gamma * running
        out[i] = running
    return out


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float,
    lam: float,
    last_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(rewards), dtype=np.float32)
    returns = np.zeros(len(rewards), dtype=np.float32)
    next_adv = 0.0
    next_value = last_value
    for t in reversed(range(len(rewards))):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * mask - values[t]
        next_adv = delta + gamma * lam * mask * next_adv
        advantages[t] = next_adv
        returns[t] = advantages[t] + values[t]
        next_value = values[t]
    return advantages, returns


def room_id_from_env(env: gym.Env) -> int:
    """
    Approximate room ID for FourRooms-like maps by splitting around grid center.
    """
    inner = env.unwrapped
    if not hasattr(inner, "agent_pos"):
        return 0
    x, y = inner.agent_pos
    width = int(getattr(inner, "width", 19))
    height = int(getattr(inner, "height", 19))
    mid_x = width // 2
    mid_y = height // 2
    return int(x > mid_x) + 2 * int(y > mid_y)


@dataclass
class Trajectory:
    seed: int
    obs: list[np.ndarray]
    actions: list[int]
    log_probs: list[float]
    rewards: list[float]
    dones: list[bool]
    values: list[float]
    rooms: list[int]

    def episodic_return(self, gamma: float) -> float:
        return float(discounted_cumsum(self.rewards, gamma=gamma)[0]) if self.rewards else 0.0


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


def behavior_clone_pretrain(
    env_id: str,
    model: ActorCritic,
    device: torch.device,
    seed: int,
    num_rollouts: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict:
    """
    Build a supervised dataset from shortest-path expert trajectories and pretrain actor logits.
    This approximates the "pretrained policy + RLFT" setup used in the paper.
    """
    if num_rollouts <= 0 or epochs <= 0:
        return {"dataset_size": 0, "loss": 0.0}

    env = make_env(env_id, seed=seed)
    obs_ds: list[np.ndarray] = []
    act_ds: list[int] = []

    for i in range(num_rollouts):
        obs, _ = env.reset(seed=seed + i)
        plan = _shortest_action_plan_to_goal(env)
        if not plan:
            continue
        for action in plan:
            obs_ds.append(np.asarray(obs, dtype=np.float32))
            act_ds.append(int(action))
            obs, _reward, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break

    env.close()
    if not obs_ds:
        return {"dataset_size": 0, "loss": 0.0}

    obs_t = torch.as_tensor(np.asarray(obs_ds), dtype=torch.float32, device=device)
    act_t = torch.as_tensor(np.asarray(act_ds), dtype=torch.int64, device=device)
    idx = np.arange(obs_t.shape[0])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    last_loss = 0.0

    model.train()
    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            mb = idx[start : start + batch_size]
            logits, _ = model(obs_t[mb])
            loss = nn.functional.cross_entropy(logits, act_t[mb])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            last_loss = float(loss.item())

    return {"dataset_size": int(obs_t.shape[0]), "loss": last_loss}


def rollout_episode(
    env: gym.Env,
    model: ActorCritic,
    device: torch.device,
    seed: int,
    gamma: float,
    shaping_coef: float = 0.0,
    replay_actions: Optional[list[int]] = None,
    max_steps: Optional[int] = None,
) -> Trajectory:
    obs, _ = env.reset(seed=seed)
    if replay_actions:
        for a in replay_actions:
            obs, _r, terminated, truncated, _ = env.step(a)
            if terminated or truncated:
                # Replay failed to reach the requested branch state; start over.
                obs, _ = env.reset(seed=seed)
                break

    traj = Trajectory(seed=seed, obs=[], actions=[], log_probs=[], rewards=[], dones=[], values=[], rooms=[])
    steps = 0
    prev_dist = _grid_shortest_distance_to_goal(env) if shaping_coef > 0 else 0.0
    while True:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_t, logp_t, value_t = model.act(obs_t)

        action = int(action_t.item())
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        if shaping_coef > 0:
            next_dist = _grid_shortest_distance_to_goal(env)
            if np.isfinite(prev_dist) and np.isfinite(next_dist):
                reward = float(reward) + shaping_coef * float(prev_dist - next_dist)
            prev_dist = next_dist

        traj.obs.append(np.asarray(obs, dtype=np.float32))
        traj.actions.append(action)
        traj.log_probs.append(float(logp_t.item()))
        traj.rewards.append(float(reward))
        traj.dones.append(done)
        traj.values.append(float(value_t.item()))
        traj.rooms.append(room_id_from_env(env))

        obs = next_obs
        steps += 1
        if done:
            break
        if max_steps is not None and steps >= max_steps:
            break
    return traj


def trajectory_signature_rooms(traj: Trajectory) -> tuple[int, ...]:
    return tuple(sorted(set(traj.rooms)))


def poly_set_score(group: list[Trajectory], gamma: float) -> float:
    returns = np.array([t.episodic_return(gamma=gamma) for t in group], dtype=np.float32)
    mean_return = float(np.mean(returns))
    signatures = [trajectory_signature_rooms(t) for t in group]
    diversity = float(len(set(signatures)) / max(1, len(signatures)))
    return mean_return * diversity


def select_rollout_steps(length: int, p: int) -> list[int]:
    if length <= 2 or p <= 0:
        return []
    raw = np.linspace(1, length - 1, p + 2)[1:-1]
    steps = sorted({int(x) for x in raw if 0 <= int(x) < length})
    return steps


@dataclass
class TrainConfig:
    env_id: str = "MiniGrid-FourRooms-v0"
    total_updates: int = 250
    episodes_per_update: int = 8
    ppo_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    hidden_size: int = 128
    seed: int = 42
    # Poly-PPO params
    vine_n: int = 8
    set_size: int = 4
    num_sets: int = 4
    rollout_states_per_traj: int = 2
    poly_window: int = 3
    # Shared pretraining params
    pretrain_rollouts: int = 0
    pretrain_epochs: int = 0
    pretrain_batch_size: int = 512
    pretrain_lr: float = 1e-3
    shaping_coef: float = 0.0


def evaluate_policy(
    env_id: str,
    model: ActorCritic,
    device: torch.device,
    episodes: int,
    seed: int,
) -> dict:
    env = make_env(env_id, seed=seed)
    returns = []
    success = 0
    model.eval()
    with torch.no_grad():
        for i in range(episodes):
            obs, _ = env.reset(seed=seed + 10_000 + i)
            done = False
            total = 0.0
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = model(obs_t)
                action = int(torch.argmax(logits, dim=-1).item())
                obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                total += float(reward)
            returns.append(total)
            if total > 0.0:
                success += 1
    env.close()
    return {
        "avg_reward": float(np.mean(returns)),
        "success_rate": float(success / episodes),
    }


def train_reinforce(
    cfg: TrainConfig,
    device: torch.device,
    init_state_dict: Optional[dict[str, torch.Tensor]] = None,
) -> tuple[ActorCritic, dict]:
    set_seed(cfg.seed)
    env = make_env(cfg.env_id, seed=cfg.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(env.action_space.n)
    model = ActorCritic(obs_dim, act_dim, hidden=cfg.hidden_size).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = []
    next_seed = cfg.seed
    model.train()
    for update in range(1, cfg.total_updates + 1):
        policy_losses = []
        returns_hist = []
        for _ in range(cfg.episodes_per_update):
            traj = rollout_episode(
                env,
                model,
                device,
                seed=next_seed,
                gamma=cfg.gamma,
                shaping_coef=cfg.shaping_coef,
            )
            next_seed += 1
            returns = discounted_cumsum(traj.rewards, gamma=cfg.gamma)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            obs_t = torch.as_tensor(np.asarray(traj.obs), dtype=torch.float32, device=device)
            actions_t = torch.as_tensor(np.asarray(traj.actions), dtype=torch.int64, device=device)
            ret_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

            logits, _ = model(obs_t)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(actions_t)
            loss = -(logp * ret_t).mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()

            policy_losses.append(float(loss.item()))
            returns_hist.append(float(np.sum(traj.rewards)))

        avg_return = float(np.mean(returns_hist))
        history.append(avg_return)
        if update % 10 == 0 or update == 1:
            print(f"[REINFORCE] update {update:04d}/{cfg.total_updates} avg_return={avg_return:.3f}")

    env.close()
    metrics = evaluate_policy(cfg.env_id, model, device, episodes=100, seed=cfg.seed)
    return model, {"train_curve": history, "eval": metrics}


def flatten_batch(
    trajectories: list[Trajectory],
    advantages_by_traj: list[np.ndarray],
    returns_by_traj: list[np.ndarray],
) -> dict[str, np.ndarray]:
    obs = np.concatenate([np.asarray(t.obs, dtype=np.float32) for t in trajectories], axis=0)
    actions = np.concatenate([np.asarray(t.actions, dtype=np.int64) for t in trajectories], axis=0)
    old_logp = np.concatenate([np.asarray(t.log_probs, dtype=np.float32) for t in trajectories], axis=0)
    adv = np.concatenate([a.astype(np.float32) for a in advantages_by_traj], axis=0)
    ret = np.concatenate([r.astype(np.float32) for r in returns_by_traj], axis=0)
    return {"obs": obs, "actions": actions, "old_logp": old_logp, "adv": adv, "ret": ret}


def inject_poly_advantages(
    cfg: TrainConfig,
    env_id: str,
    model: ActorCritic,
    device: torch.device,
    trajectories: list[Trajectory],
    advantages_by_traj: list[np.ndarray],
) -> None:
    env = make_env(env_id, seed=cfg.seed + 12345)
    rng = np.random.default_rng(cfg.seed)

    for traj_idx, traj in enumerate(trajectories):
        rollout_steps = select_rollout_steps(len(traj.actions), cfg.rollout_states_per_traj)
        for step in rollout_steps:
            prefix_actions = traj.actions[:step]
            vines = []
            for k in range(cfg.vine_n):
                vine_seed = traj.seed + 100_000 + step * 101 + k
                vine = rollout_episode(
                    env=env,
                    model=model,
                    device=device,
                    seed=traj.seed,
                    gamma=cfg.gamma,
                    shaping_coef=cfg.shaping_coef,
                    replay_actions=prefix_actions,
                )
                vines.append(vine)

            if len(vines) < cfg.set_size:
                continue

            scores = []
            for _ in range(cfg.num_sets):
                idx = rng.choice(len(vines), size=cfg.set_size, replace=False)
                group = [vines[int(i)] for i in idx]
                scores.append(poly_set_score(group, gamma=cfg.gamma))

            baseline = float(np.mean(scores))
            ref_adv = float(scores[0] - baseline)

            start = step
            end = min(len(advantages_by_traj[traj_idx]), step + cfg.poly_window + 1)
            advantages_by_traj[traj_idx][start:end] = ref_adv
    env.close()


def train_ppo_like(
    cfg: TrainConfig,
    device: torch.device,
    poly: bool,
    init_state_dict: Optional[dict[str, torch.Tensor]] = None,
) -> tuple[ActorCritic, dict]:
    set_seed(cfg.seed)
    env = make_env(cfg.env_id, seed=cfg.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(env.action_space.n)
    model = ActorCritic(obs_dim, act_dim, hidden=cfg.hidden_size).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    next_seed = cfg.seed

    history = []
    model.train()
    for update in range(1, cfg.total_updates + 1):
        trajectories = []
        for _ in range(cfg.episodes_per_update):
            traj = rollout_episode(
                env,
                model,
                device,
                seed=next_seed,
                gamma=cfg.gamma,
                shaping_coef=cfg.shaping_coef,
            )
            trajectories.append(traj)
            next_seed += 1

        advantages_by_traj = []
        returns_by_traj = []
        for traj in trajectories:
            adv, ret = compute_gae(
                rewards=traj.rewards,
                values=traj.values,
                dones=traj.dones,
                gamma=cfg.gamma,
                lam=cfg.lam,
                last_value=0.0,
            )
            advantages_by_traj.append(adv)
            returns_by_traj.append(ret)

        if poly:
            inject_poly_advantages(
                cfg=cfg,
                env_id=cfg.env_id,
                model=model,
                device=device,
                trajectories=trajectories,
                advantages_by_traj=advantages_by_traj,
            )

        batch = flatten_batch(trajectories, advantages_by_traj, returns_by_traj)
        batch["adv"] = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8)

        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
        act_t = torch.as_tensor(batch["actions"], dtype=torch.int64, device=device)
        old_logp_t = torch.as_tensor(batch["old_logp"], dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(batch["adv"], dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(batch["ret"], dtype=torch.float32, device=device)

        n = obs_t.shape[0]
        idx_all = np.arange(n)
        for _ in range(cfg.ppo_epochs):
            np.random.shuffle(idx_all)
            for start in range(0, n, cfg.minibatch_size):
                mb_idx = idx_all[start : start + cfg.minibatch_size]
                mb_obs = obs_t[mb_idx]
                mb_act = act_t[mb_idx]
                mb_old_logp = old_logp_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]

                logits, value = model(mb_obs)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - mb_old_logp)
                clip_ratio = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
                policy_loss = -torch.min(ratio * mb_adv, clip_ratio * mb_adv).mean()
                value_loss = torch.mean((value - mb_ret) ** 2)
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()

        avg_return = float(np.mean([np.sum(t.rewards) for t in trajectories]))
        history.append(avg_return)
        algo_name = "Poly-PPO" if poly else "PPO"
        if update % 10 == 0 or update == 1:
            print(f"[{algo_name}] update {update:04d}/{cfg.total_updates} avg_return={avg_return:.3f}")

    env.close()
    metrics = evaluate_policy(cfg.env_id, model, device, episodes=100, seed=cfg.seed)
    return model, {"train_curve": history, "eval": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REINFORCE, PPO and Poly-PPO on a Table-1 style environment.")
    parser.add_argument("--env-id", type=str, default="MiniGrid-FourRooms-v0")
    parser.add_argument("--algo", type=str, choices=["reinforce", "ppo", "poly-ppo", "all"], default="all")
    parser.add_argument("--updates", type=int, default=250)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--vine-n", type=int, default=8)
    parser.add_argument("--set-size", type=int, default=4)
    parser.add_argument("--num-sets", type=int, default=4)
    parser.add_argument("--rollout-states-per-traj", type=int, default=2)
    parser.add_argument("--poly-window", type=int, default=3)
    parser.add_argument("--pretrain-rollouts", type=int, default=0)
    parser.add_argument("--pretrain-epochs", type=int, default=0)
    parser.add_argument("--pretrain-batch-size", type=int, default=512)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--shaping-coef", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="ppo/results_fourrooms.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        env_id=args.env_id,
        total_updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        vine_n=args.vine_n,
        set_size=args.set_size,
        num_sets=args.num_sets,
        rollout_states_per_traj=args.rollout_states_per_traj,
        poly_window=args.poly_window,
        pretrain_rollouts=args.pretrain_rollouts,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_batch_size=args.pretrain_batch_size,
        pretrain_lr=args.pretrain_lr,
        shaping_coef=args.shaping_coef,
        seed=args.seed,
    )
    device = torch.device(args.device)
    results = {"config": asdict(cfg), "env_id": cfg.env_id, "runs": {}}

    # Shared pretrained initialization for fair comparison across methods.
    probe_env = make_env(cfg.env_id, seed=cfg.seed)
    obs_dim = int(np.prod(probe_env.observation_space.shape))
    act_dim = int(probe_env.action_space.n)
    probe_env.close()
    base_model = ActorCritic(obs_dim, act_dim, hidden=cfg.hidden_size).to(device)
    pretrain_info = behavior_clone_pretrain(
        env_id=cfg.env_id,
        model=base_model,
        device=device,
        seed=cfg.seed,
        num_rollouts=cfg.pretrain_rollouts,
        epochs=cfg.pretrain_epochs,
        batch_size=cfg.pretrain_batch_size,
        lr=cfg.pretrain_lr,
    )
    print(
        "[Pretrain] "
        f"dataset={pretrain_info['dataset_size']}, final_ce_loss={pretrain_info['loss']:.4f}"
    )
    init_state_dict = copy.deepcopy(base_model.state_dict())
    results["pretrain"] = pretrain_info

    if args.algo in ("reinforce", "all"):
        _, res = train_reinforce(cfg, device=device, init_state_dict=init_state_dict)
        results["runs"]["reinforce"] = res

    if args.algo in ("ppo", "all"):
        _, res = train_ppo_like(cfg, device=device, poly=False, init_state_dict=init_state_dict)
        results["runs"]["ppo"] = res

    if args.algo in ("poly-ppo", "all"):
        _, res = train_ppo_like(cfg, device=device, poly=True, init_state_dict=init_state_dict)
        results["runs"]["poly_ppo"] = res

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))

    print("\n=== Final eval (avg_reward, success_rate) ===")
    for k, v in results["runs"].items():
        m = v["eval"]
        print(f"{k:>10s}: ({m['avg_reward']:.3f}, {100.0 * m['success_rate']:.1f}%)")
    print(f"Saved results to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
