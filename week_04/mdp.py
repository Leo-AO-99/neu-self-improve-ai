from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum
from load_data import load_candles, compute_spread_zscore

DATA_DIR = Path(__file__).resolve().parent / "data"


class Action(IntEnum):
    """Action set: open/close positions or hold."""
    OPEN_LONG_SPREAD = 0   # long spread
    OPEN_SHORT_SPREAD = 1  # short spread
    CLOSE = 2              # close position
    HOLD = 3               # hold position

    def next_position(self, current_position: int) -> int:
        """Given current position (-1, 0, +1), return position after this action."""
        if self == Action.OPEN_LONG_SPREAD:
            return 1
        if self == Action.OPEN_SHORT_SPREAD:
            return -1
        if self == Action.CLOSE:
            return 0
        # HOLD
        return current_position
    


class SpreadState(IntEnum):
    """Spread z-score regime (z_bin 0 = lowest z, 6 = highest z). n_z_bins=7 only."""
    SPREAD_VERY_LOW = 0      # z ≈ [-3.5, -2.3]
    SPREAD_LOW = 1           # z ≈ [-2.3, -1.2]
    SPREAD_BELOW_MEAN = 2    # z ≈ [-1.2, -0.4]
    SPREAD_NEAR_MEAN = 3     # z ≈ [-0.4,  0.4]
    SPREAD_ABOVE_MEAN = 4    # z ≈ [ 0.4,  1.2]
    SPREAD_HIGH = 5          # z ≈ [ 1.2,  2.3]
    SPREAD_VERY_HIGH = 6     # z ≈ [ 2.3,  3.5]

    @classmethod
    def z_bin_to_name(cls, z_bin: int, n_z_bins: int) -> str:
        if n_z_bins == 7 and 0 <= z_bin < 7:
            return cls(z_bin).name
        return f"Z_BIN_{z_bin}"


# Compound state: (z_bin, position). state_index = z_bin * 3 + (position + 1).
N_POSITIONS = 3  # -1, 0, +1

def make_state_index(z_bin: int, position: int, n_z_bins: int) -> int:
    """Encode (z_bin, position) into single state index. position in {-1, 0, 1}."""
    pos_idx = position + 1  # -1->0, 0->1, 1->2
    return z_bin * N_POSITIONS + pos_idx

def state_index_to_z_bin(s: int, n_z_bins: int) -> int:
    return s // N_POSITIONS

def state_index_to_position(s: int) -> int:
    return (s % N_POSITIONS) - 1

def state_index_to_name(s: int, n_z_bins: int) -> str:
    z_bin = state_index_to_z_bin(s, n_z_bins)
    pos = state_index_to_position(s)
    pos_str = "LONG" if pos == 1 else "FLAT" if pos == 0 else "SHORT"
    return f"{SpreadState.z_bin_to_name(z_bin, n_z_bins)}|{pos_str}"



@dataclass
class MDPConfig:
    n_z_bins: int = len(SpreadState)   # z-score bins (e.g. 7)
    n_states: int = len(SpreadState) * N_POSITIONS  # (z_bin, position) -> 21
    n_actions: int = len(Action)       # OPEN_LONG, OPEN_SHORT, CLOSE, HOLD
    cost_per_trade: float = 0.0005
    gamma: float = 0.99

    def __post_init__(self):
        self.n_states = self.n_z_bins * N_POSITIONS


def zscore_to_state(z: float, n_states: int) -> int:
    """Map z-score to state index in [0, n_states-1]. Center state = near zero."""
    # clip z to roughly [-3, 3] then bin
    z_clip = np.clip(z, -3.5, 3.5)
    # linear map [-3.5, 3.5] -> [0, n_states-1]
    state = int(np.round((z_clip + 3.5) / 7.0 * (n_states - 1)))
    return np.clip(state, 0, n_states - 1)


def build_mdp_from_series(
    z: pd.Series,
    config: MDPConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build T[s,a,s'], R[s,a] with compound state s=(z_bin, position).
    For each t we average over all (position, action); reward = position*spread_ret - cost when changing position.
    """
    n_z_bins = config.n_z_bins
    n_states = config.n_states
    n_actions = config.n_actions
    cost = config.cost_per_trade

    T = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))
    count_sa = np.zeros((n_states, n_actions))

    z_arr = z.values
    valid = np.isfinite(z_arr)
    n = len(z_arr) - 1

    for t in range(n):
        if not valid[t] or not valid[t + 1]:
            continue
        z_bin = zscore_to_state(z_arr[t], n_z_bins)
        z_bin_next = zscore_to_state(z_arr[t + 1], n_z_bins)
        spread_ret = z_arr[t + 1] - z_arr[t]

        for position in (-1, 0, 1):
            s = make_state_index(z_bin, position, n_z_bins)
            for a in range(n_actions):
                act = Action(a)
                pos_next = act.next_position(position)
                s_next = make_state_index(z_bin_next, pos_next, n_z_bins)
                reward = position * spread_ret - (cost if pos_next != position else 0.0)
                T[s, a, s_next] += 1 # count the transition from s to s_next with action a
                R[s, a] += reward # count the reward for the transition from s to s_next with action a
                count_sa[s, a] += 1 # count the number of times the state s and action a have been seen

    for s in range(n_states):
        for a in range(n_actions):
            if count_sa[s, a] > 0:
                t_sum = T[s, a, :].sum()
                if t_sum > 0:
                    T[s, a, :] /= t_sum
                else:
                    T[s, a, s] = 1.0
                R[s, a] /= count_sa[s, a]
            else:
                T[s, a, s] = 1.0
                R[s, a] = 0.0
    return T, R, count_sa


def value_iteration(
    T: np.ndarray,
    R: np.ndarray,
    gamma: float = 0.99,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return V[s], policy[s] (action index 0..n_actions-1)."""
    n_states, n_actions = R.shape
    V = np.zeros(n_states)
    for _ in range(max_iter):
        V_old = V.copy()
        for s in range(n_states):
            Q = np.array([R[s, a] + gamma * T[s, a, :] @ V for a in range(n_actions)])
            V[s] = Q.max()
        if np.abs(V - V_old).max() < tol:
            break
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        Q = np.array([R[s, a] + gamma * T[s, a, :] @ V for a in range(n_actions)])
        policy[s] = int(np.argmax(Q))
    return V, policy


def backtest(
    merged: pd.DataFrame,
    z: pd.Series,
    policy: np.ndarray,
    config: MDPConfig,
) -> pd.DataFrame:
    """Backtest: state = (z_bin, position), action = policy[state], new_position = action.next_position(position)."""
    z_arr = z.values
    n = len(z_arr) - 1
    n_z_bins = config.n_z_bins
    cost = config.cost_per_trade
    position = 0
    pnls = []
    positions_list = []
    actions_list = []

    for t in range(n):
        if not np.isfinite(z_arr[t]) or not np.isfinite(z_arr[t + 1]):
            pnls.append(0.0)
            positions_list.append(position)
            actions_list.append(Action.HOLD.name)
            continue
        z_bin = zscore_to_state(z_arr[t], n_z_bins)
        s = make_state_index(z_bin, position, n_z_bins)
        act = Action(policy[s])
        new_position = act.next_position(position)
        spread_ret = z_arr[t + 1] - z_arr[t]
        pnl = position * spread_ret - (cost if new_position != position else 0.0)
        pnls.append(pnl)
        positions_list.append(new_position)
        actions_list.append(act.name)
        position = new_position

    result = merged.iloc[1 : n + 1].copy()
    result["z"] = z_arr[1 : n + 1]
    result["position"] = positions_list
    result["action"] = actions_list
    result["pnl"] = pnls
    result["cum_pnl"] = np.cumsum(pnls)
    return result


def plot_policy_heatmap(
    policy: np.ndarray,
    config: MDPConfig,
    save_path: Optional[str] = "policy_heatmap.png",
) -> None:
    """Heatmap: rows = z_bin (spread state), cols = position (SHORT/FLAT/LONG), color = action."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skip policy heatmap")
        return
    n_z_bins = config.n_z_bins
    # policy[s] -> action index; s = z_bin * 3 + (position + 1)
    mat = np.zeros((n_z_bins, N_POSITIONS), dtype=int)
    for z_bin in range(n_z_bins):
        for pos in (-1, 0, 1):
            s = make_state_index(z_bin, pos, n_z_bins)
            mat[z_bin, pos + 1] = policy[s]
    fig, ax = plt.subplots(figsize=(4, 5))
    im = ax.imshow(mat, aspect="auto", cmap="tab10", vmin=0, vmax=9)
    ax.set_yticks(range(n_z_bins))
    ax.set_yticklabels([SpreadState.z_bin_to_name(i, n_z_bins) for i in range(n_z_bins)], fontsize=8)
    ax.set_xticks(range(N_POSITIONS))
    ax.set_xticklabels(["SHORT", "FLAT", "LONG"])
    ax.set_xlabel("Position")
    ax.set_ylabel("Spread (z_bin)")
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels([Action(i).name for i in range(4)])
    plt.title("Policy heatmap (action by state)")
    for i in range(n_z_bins):
        for j in range(N_POSITIONS):
            ax.text(j, i, Action(mat[i, j]).name[:4], ha="center", va="center", fontsize=7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Policy heatmap saved to {save_path}")
    plt.show()


def plot_policy_heatmap_inv(
    policy: np.ndarray,
    config: "InventoryMDPConfig",
    save_path: Optional[str] = "policy_heatmap_inv.png",
) -> None:
    """Heatmap: rows = z_bin, cols = inventory (-Q..Q), color = delta (action)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skip policy heatmap")
        return
    Q = config.Q
    n_z_bins = config.n_z_bins
    n_pos = 2 * Q + 1
    mat = np.zeros((n_z_bins, n_pos), dtype=int)
    for z_bin in range(n_z_bins):
        for inv in range(-Q, Q + 1):
            s = make_state_index_inv(z_bin, inv, n_z_bins, Q)
            mat[z_bin, inv + Q] = action_to_delta(policy[s], Q)
    fig, ax = plt.subplots(figsize=(n_pos * 0.7 + 2, 5))
    v = max(2 * Q, 1)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_yticks(range(n_z_bins))
    ax.set_yticklabels([SpreadState.z_bin_to_name(i, n_z_bins) for i in range(n_z_bins)], fontsize=8)
    ax.set_xticks(range(n_pos))
    ax.set_xticklabels([str(inv) for inv in range(-Q, Q + 1)])
    ax.set_xlabel("Inventory")
    ax.set_ylabel("Spread (z_bin)")
    plt.colorbar(im, ax=ax, label="Δ (action)")
    plt.title("Policy heatmap (Δ by z_bin × inventory)")
    for i in range(n_z_bins):
        for j in range(n_pos):
            ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Policy heatmap saved to {save_path}")
    plt.show()


# Q-Learning
def _inv_range(Q: int):
    """Inventory range: integers from -Q to Q."""
    return range(-Q, Q + 1)


def _delta_range(Q: int):
    """Action (delta) range: integers from -2*Q to 2*Q."""
    return range(-2 * Q, 2 * Q + 1)


def n_actions_inv(Q: int) -> int:
    """Number of actions = 4*Q + 1."""
    return 4 * Q + 1


def action_to_delta(a: int, Q: int) -> int:
    """Action index 0..4*Q maps to delta = -2*Q, ..., 2*Q."""
    return a - 2 * Q


def delta_to_action(delta: int, Q: int) -> int:
    return delta + 2 * Q


def next_inventory(inventory: int, delta: int, Q: int) -> int:
    """Next inventory = clip(inventory + delta, -Q, Q)."""
    return int(np.clip(inventory + delta, -Q, Q))


def make_state_index_inv(z_bin: int, inventory: int, n_z_bins: int, Q: int) -> int:
    """State = (z_bin, inventory). inventory ∈ {-Q, ..., Q}."""
    inv_idx = inventory + Q  # 0 .. 2*Q
    return z_bin * (2 * Q + 1) + inv_idx


def state_index_to_inventory(s: int, n_z_bins: int, Q: int) -> int:
    inv_idx = s % (2 * Q + 1)
    return inv_idx - Q


def state_index_to_name_inv(s: int, n_z_bins: int, Q: int) -> str:
    z_bin = s // (2 * Q + 1)
    inv = state_index_to_inventory(s, n_z_bins, Q)
    return f"{SpreadState.z_bin_to_name(z_bin, n_z_bins)}|inv={inv}"


@dataclass
class InventoryMDPConfig:
    """MDP with state=(z_bin, inventory), action=Δ ∈ {-2Q,...,2Q}. Reward = pnl - inventory_penalty - risk_penalty."""
    Q: int = 2  # max inventory; inventory ∈ {-Q,...,Q}, delta ∈ {-2Q,...,2Q}
    n_z_bins: int = len(SpreadState)
    n_states: int = 0  # n_z_bins * (2*Q+1)
    n_actions: int = 0  # 4*Q+1
    cost_per_unit: float = 0.0005  # cost per unit of |delta|
    lambda_inventory: float = 0.01  # penalty for holding inventory, e.g. 0.01 * inventory^2
    lambda_risk: float = 0.1  # penalty for spread return variance, e.g. 0.1 * spread_ret^2
    gamma: float = 0.99

    def __post_init__(self):
        self.n_states = self.n_z_bins * (2 * self.Q + 1)
        self.n_actions = n_actions_inv(self.Q)


def build_mdp_from_series_inv(
    z: pd.Series,
    config: InventoryMDPConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build T[s,a,s'], R[s,a] for inventory MDP.
    Action a = delta ∈ {-2Q,...,2Q}. next_inv = clip(inv + delta, -Q, Q).
    Reward = pnl - inventory_penalty - risk_penalty:
      pnl = inventory * spread_ret - cost_per_unit * |delta|
      inventory_penalty = lambda_inventory * inventory^2
      risk_penalty = lambda_risk * spread_ret^2
    """
    Q = config.Q
    n_z_bins = config.n_z_bins
    n_states = config.n_states
    n_actions = config.n_actions
    cost = config.cost_per_unit
    lam_inv = config.lambda_inventory
    lam_risk = config.lambda_risk

    T = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))
    count_sa = np.zeros((n_states, n_actions))

    z_arr = z.values
    valid = np.isfinite(z_arr)
    n = len(z_arr) - 1
    inv_range = list(_inv_range(Q))

    for t in range(n):
        if not valid[t] or not valid[t + 1]:
            continue
        z_bin = zscore_to_state(z_arr[t], n_z_bins)
        z_bin_next = zscore_to_state(z_arr[t + 1], n_z_bins)
        spread_ret = z_arr[t + 1] - z_arr[t]

        pnl = 0.0  # will be set per (inventory, delta)
        inv_penalty = 0.0  # will be set per inventory
        risk_penalty = lam_risk * (spread_ret ** 2)

        for inventory in inv_range:
            inv_penalty = lam_inv * (inventory ** 2)
            s = make_state_index_inv(z_bin, inventory, n_z_bins, Q)
            for a in range(n_actions):
                delta = action_to_delta(a, Q)
                inv_next = next_inventory(inventory, delta, Q)
                s_next = make_state_index_inv(z_bin_next, inv_next, n_z_bins, Q)
                pnl = inventory * spread_ret - cost * abs(delta)
                reward = pnl - inv_penalty - risk_penalty
                T[s, a, s_next] += 1
                R[s, a] += reward
                count_sa[s, a] += 1

    for s in range(n_states):
        for a in range(n_actions):
            if count_sa[s, a] > 0:
                t_sum = T[s, a, :].sum()
                if t_sum > 0:
                    T[s, a, :] /= t_sum
                else:
                    T[s, a, s] = 1.0
                R[s, a] /= count_sa[s, a]
            else:
                T[s, a, s] = 1.0
                R[s, a] = 0.0
    return T, R, count_sa


def backtest_inv(
    merged: pd.DataFrame,
    z: pd.Series,
    policy: np.ndarray,
    config: InventoryMDPConfig,
) -> pd.DataFrame:
    """Backtest: action = delta ∈ {-2Q,...,2Q}, inventory_next = clip(inventory + delta, -Q, Q). Reported pnl is raw (no penalty terms)."""
    Q = config.Q
    n_z_bins = config.n_z_bins
    cost = config.cost_per_unit
    z_arr = z.values
    n = len(z_arr) - 1
    inventory = 0
    pnls = []
    inventories_list = []
    deltas_list = []

    for t in range(n):
        if not np.isfinite(z_arr[t]) or not np.isfinite(z_arr[t + 1]):
            pnls.append(0.0)
            inventories_list.append(inventory)
            deltas_list.append(0)
            continue
        z_bin = zscore_to_state(z_arr[t], n_z_bins)
        s = make_state_index_inv(z_bin, inventory, n_z_bins, Q)
        a = policy[s]
        delta = action_to_delta(a, Q)
        inv_next = next_inventory(inventory, delta, Q)
        spread_ret = z_arr[t + 1] - z_arr[t]
        pnl = inventory * spread_ret - cost * abs(delta)
        pnls.append(pnl)
        inventories_list.append(inv_next)
        deltas_list.append(delta)
        inventory = inv_next

    result = merged.iloc[1 : n + 1].copy()
    result["z"] = z_arr[1 : n + 1]
    result["inventory"] = inventories_list
    result["delta"] = deltas_list
    result["pnl"] = pnls
    result["cum_pnl"] = np.cumsum(pnls)
    return result

