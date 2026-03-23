import copy
from typing import SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam
import matplotlib.pyplot as plt
import colorama

NDArrayFloat = npt.NDArray[np.floating]
NDArrayBool = npt.NDArray[np.bool_]


class Policy:
    """Base policy class implementing REINFORCE-style gradient: ∇J(θ) ≈ E[∇log π(a|s) · A(s,a)]."""
    num_states: int
    opt: Adam

    def pi(self, s_t: torch.Tensor | NDArrayFloat) -> torch.distributions.Distribution:
        """Return the action distribution π(·|s_t). Subclasses define the parameterization."""
        raise NotImplementedError

    def act(self, s_t: NDArrayFloat) -> torch.Tensor:
        """Sample a_t ~ π(·|s_t)."""
        a_t: torch.Tensor = self.pi(s_t).sample()
        return a_t

    def learn(self, states: NDArrayFloat, actions: NDArrayFloat, advantages: NDArrayFloat) -> torch.Tensor:
        """Policy gradient step: minimize -E[log π(a|s) · Â(s,a)] (negative because we ascend J)."""
        actions_t: torch.Tensor = torch.tensor(actions)
        advantages_t: torch.Tensor = torch.tensor(advantages)
        log_prob: torch.Tensor = self.pi(states).log_prob(actions_t)
        loss: torch.Tensor = torch.mean(-log_prob * advantages_t)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss


class ValueEstimator:
    """Critic network: learns V(s) by minimizing MSE against Monte Carlo return targets."""

    def __init__(self, env: gym.Env, lr: float = 1e-2) -> None:
        self.num_states = env.observation_space.shape[0]
        self.V: nn.Sequential = nn.Sequential(
            nn.Linear(self.num_states, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).double()
        self.opt: Adam = Adam(self.V.parameters(), lr=lr)

    def predict(self, s_t: NDArrayFloat | torch.Tensor) -> torch.Tensor:
        """V(s_t) — scalar value estimate for each state in the batch."""
        s_t_tensor: torch.Tensor = torch.as_tensor(s_t).double()
        return self.V(s_t_tensor).squeeze(-1)

    def learn(self, v_pred: torch.Tensor, returns: NDArrayFloat) -> torch.Tensor:
        """Minimize L = E[(V(s) - G_t)^2] where G_t is the (bootstrapped) return."""
        returns_t: torch.Tensor = torch.tensor(returns)
        loss: torch.Tensor = torch.mean((v_pred - returns_t) ** 2)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss


class CategoricalPolicy(Policy):
    """Actor for discrete action spaces: linear logits → softmax → Categorical(a|s)."""
    num_actions: int

    def __init__(self, env: gym.Env, lr: float = 1e-2) -> None:
        self.num_states = env.observation_space.shape[0]
        self.num_actions: int = env.action_space.n
        # Single linear layer: s → logits (no hidden layers — keeps CartPole simple)
        self.p: nn.Sequential = nn.Sequential(
            nn.Linear(self.num_states, self.num_actions),
        ).double()
        self.opt: Adam = Adam(self.p.parameters(), lr=lr)

    def pi(self, s_t: torch.Tensor | NDArrayFloat) -> Categorical:
        s_t_tensor: torch.Tensor = torch.as_tensor(s_t).double()
        logits: torch.Tensor = self.p(s_t_tensor)
        return Categorical(logits=logits)  # softmax applied internally


class VectorizedEnvWrapper(gym.Wrapper):
    """Runs N independent env copies in lockstep for parallel trajectory collection."""

    def __init__(self, env: gym.Env, num_envs: int = 1) -> None:
        super().__init__(env)
        self.num_envs: int = num_envs
        self.envs: list[gym.Env] = [copy.deepcopy(env) for _ in range(num_envs)]

    def reset_all(self) -> NDArrayFloat:
        return np.asarray([env.reset()[0] for env in self.envs])

    def step(self, actions: torch.Tensor) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayBool]:
        next_states: list[npt.NDArray] = []
        rewards: list[SupportsFloat] = []
        dones: list[bool] = []
        for env, action in zip(self.envs, actions):
            next_state, reward, terminated, truncated, _info = env.step(action.item())
            done: bool = terminated or truncated
            if done:
                next_states.append(env.reset()[0])  # auto-reset on episode end
            else:
                next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
        return (
            np.asarray(next_states),
            np.asarray(rewards),
            np.asarray(dones),
        )


def calculate_returns(rewards: NDArrayFloat, dones: NDArrayFloat, gamma: float) -> NDArrayFloat:
    """Compute discounted returns G_t = r_t + γ·(1-d_t)·G_{t+1} via backward pass.
    The (1-d_t) term zeros out the future when an episode terminates."""
    result: NDArrayFloat = np.empty_like(rewards)
    result[-1] = rewards[-1]
    for t in range(len(rewards) - 2, -1, -1):
        result[t] = rewards[t] + gamma * (1 - dones[t]) * result[t + 1]
    return result


def calculate_advantages(td_errors: NDArrayFloat, dones: NDArrayFloat, lam: float, gamma: float) -> NDArrayFloat:
    """GAE(γ,λ): Â_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}, computed as a backward recursion.
    λ=0 gives pure TD (low variance, high bias); λ=1 gives Monte Carlo (high variance, low bias)."""
    result: NDArrayFloat = np.empty_like(td_errors)
    result[-1] = td_errors[-1]
    for t in range(len(td_errors) - 2, -1, -1):
        result[t] = td_errors[t] + gamma * lam * (1 - dones[t]) * result[t + 1]
    return result


def a2c(env: VectorizedEnvWrapper, agent: Policy, value_estimator: ValueEstimator,
        gamma: float, lam: float, epochs: int, train_v_iters: int, rollout_traj_len: int) -> None:
    """Advantage Actor-Critic (A2C): synchronous variant of A3C.
    Each epoch: (1) collect rollout, (2) fit critic, (3) compute GAE, (4) update actor."""

    # Pre-allocate trajectory buffers: [time, num_envs, ...]
    states: NDArrayFloat = np.empty((rollout_traj_len + 1, env.num_envs, agent.num_states))  # +1 for bootstrap state
    actions: NDArrayFloat = np.empty((rollout_traj_len, env.num_envs))
    rewards: NDArrayFloat = np.empty((rollout_traj_len, env.num_envs))
    dones: NDArrayFloat = np.empty((rollout_traj_len, env.num_envs))

    avg_returns: list[float] = []
    s_t: NDArrayFloat = env.reset_all()

    for epoch in range(epochs):
        # === Phase 1: Collect rollout using current policy ===
        for t in range(rollout_traj_len):
            a_t: torch.Tensor = agent.act(s_t)
            s_t_next, r_t, d_t = env.step(a_t)
            states[t] = s_t
            actions[t] = a_t.numpy()
            rewards[t] = r_t
            dones[t] = d_t
            s_t = s_t_next

        states[rollout_traj_len] = s_t  # final state needed for V(s_{T}) bootstrap

        # === Phase 2: Fit the critic V(s) ===
        # Snapshot V for all states (including terminal bootstrap) before training
        V_pred_pre: NDArrayFloat = value_estimator.predict(states).detach().numpy()

        # Bootstrap: at the rollout boundary, add γ·V(s_T) so we don't treat truncation as terminal
        bootstrap_rewards: NDArrayFloat = rewards.copy()
        bootstrap_rewards[-1] += gamma * (1 - dones[-1]) * V_pred_pre[-1]
        returns: NDArrayFloat = calculate_returns(bootstrap_rewards, dones, gamma)

        # Multiple gradient steps on the critic to fit V(s) ≈ G_t
        for _i in range(train_v_iters):
            V_pred_train: torch.Tensor = value_estimator.predict(states[:-1])
            value_estimator.learn(V_pred_train, returns)

        # === Phase 3: Compute advantages via GAE ===
        # TD error: δ_t = r_t + γ·V(s_{t+1}) - V(s_t)  (using pre-update V for consistency)
        td_errors: NDArrayFloat = rewards + gamma * (1 - dones) * V_pred_pre[1:] - V_pred_pre[:-1]
        advantages: NDArrayFloat = calculate_advantages(td_errors, dones, lam, gamma)

        # Normalize advantages: reduces variance and stabilizes training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # === Phase 4: Policy gradient update on the actor ===
        _pi_loss: torch.Tensor = agent.learn(states[:-1], actions, advantages)

        # Logging: average episodic return ≈ total reward / number of completed episodes
        avg_return = float(rewards.sum() / dones.sum())
        avg_returns.append(avg_return)
        print(f"{epoch}/{epochs}\t{colorama.Fore.CYAN}{int(avg_return)}{colorama.Style.RESET_ALL}")

    env.close()

    sns.lineplot(x=range(len(avg_returns)), y=avg_returns)
    plt.show()


def main():
    env: VectorizedEnvWrapper = VectorizedEnvWrapper(
        gym.make("CartPole-v1"), num_envs=8  # 8 parallel envs for variance reduction
    )

    assert isinstance(env.observation_space, gym.spaces.Box), "This example assumes a Box observation space."
    assert isinstance(env.action_space, gym.spaces.Discrete), "This example assumes a Discrete action space."

    categorical: CategoricalPolicy = CategoricalPolicy(env, lr=1e-1)   # actor
    value_est: ValueEstimator = ValueEstimator(env, lr=1e-2)            # critic (lower lr for stability)
    a2c(env, categorical, value_est, gamma=0.99, lam=0.95, epochs=100, train_v_iters=80, rollout_traj_len=4052)


if __name__ == "__main__":
    main()