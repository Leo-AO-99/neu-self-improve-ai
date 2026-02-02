# MDP

### Action

Four Actions

```python
class Action(IntEnum):
    """Action set: open/close positions or hold."""
    OPEN_LONG_SPREAD = 0   # long spread
    OPEN_SHORT_SPREAD = 1  # short spread
    CLOSE = 2              # close position
    HOLD = 3               # hold position
```


### State

21 states, based on the z-score of candles and position

7 (Spread State) * 3 (Position) = 21 (State)

```python
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

```


### Transition Probability

Next state $s' = (z'_{\mathrm{bin}}, \mathrm{pos}')$:

- **Position** is deterministic given current state and action: $\mathrm{pos}' = \mathrm{next\_position}(\mathrm{pos}, a)$ (see `Action.next_position`).
- **Z-bin** is stochastic (market); we estimate $P(z'_{\mathrm{bin}} \mid z_{\mathrm{bin}})$ from history.

So $T(s, a, s')$ is estimated empirically: for each $(s, a)$, count how often the next state is $s'$ (with $s'$ encoding $z'_{\mathrm{bin}}$ and $\mathrm{pos}'$), then normalize. In code:

```python
# For each t: z_bin, z_bin_next from data; for each (position, a):
s = make_state_index(z_bin, position, n_z_bins)
pos_next = act.next_position(position)
s_next = make_state_index(z_bin_next, pos_next, n_z_bins)
T[s, a, s_next] += 1
# then normalize: T[s, a, :] /= T[s, a, :].sum()
```

### Reward Function

Reward at step $t$ (when in state $s$ with position $\mathrm{pos}$, taking action $a$):

$$
r_t = \mathrm{pos} \cdot (z_{t+1} - z_t) - c \cdot \mathbf{1}[\mathrm{pos}' \neq \mathrm{pos}]
$$

- $\mathrm{pos} \cdot (z_{t+1} - z_t)$: PnL from holding position over the period (spread return in z-space).
- $c$: `cost_per_trade` (e.g. 0.0005); charged only when position changes ($\mathrm{pos}' \neq \mathrm{pos}$).

$R(s, a)$ is the empirical average of $r_t$ over all $t$ where state was $s$ and action $a$ was (implicitly) considered. In code:

```python
reward = position * spread_ret - (cost if pos_next != position else 0.0)
R[s, a] += reward

...

R[s, a] /= count_sa[s, a]
```


### Output

```shell
Optimal policy (state = z_bin | position -> action):
  SPREAD_VERY_LOW|SHORT (s=0) -> OPEN_LONG_SPREAD (position -1 -> 1, V=0.6225)
  SPREAD_VERY_LOW|FLAT (s=1) -> OPEN_LONG_SPREAD (position 0 -> 1, V=0.6691)
  SPREAD_VERY_LOW|LONG (s=2) -> OPEN_LONG_SPREAD (position 1 -> 1, V=0.7162)
  SPREAD_LOW|SHORT (s=3) -> OPEN_LONG_SPREAD (position -1 -> 1, V=0.4734)
  SPREAD_LOW|FLAT (s=4) -> OPEN_LONG_SPREAD (position 0 -> 1, V=0.4782)
  SPREAD_LOW|LONG (s=5) -> OPEN_LONG_SPREAD (position 1 -> 1, V=0.4834)
  SPREAD_BELOW_MEAN|SHORT (s=6) -> OPEN_LONG_SPREAD (position -1 -> 1, V=0.3756)
  SPREAD_BELOW_MEAN|FLAT (s=7) -> OPEN_LONG_SPREAD (position 0 -> 1, V=0.3774)
  SPREAD_BELOW_MEAN|LONG (s=8) -> OPEN_LONG_SPREAD (position 1 -> 1, V=0.3797)
  SPREAD_NEAR_MEAN|SHORT (s=9) -> OPEN_SHORT_SPREAD (position -1 -> -1, V=0.3881)
  SPREAD_NEAR_MEAN|FLAT (s=10) -> OPEN_SHORT_SPREAD (position 0 -> -1, V=0.3819)
  SPREAD_NEAR_MEAN|LONG (s=11) -> OPEN_SHORT_SPREAD (position 1 -> -1, V=0.3761)
  SPREAD_ABOVE_MEAN|SHORT (s=12) -> OPEN_LONG_SPREAD (position -1 -> 1, V=0.2948)
  SPREAD_ABOVE_MEAN|FLAT (s=13) -> OPEN_LONG_SPREAD (position 0 -> 1, V=0.2955)
  SPREAD_ABOVE_MEAN|LONG (s=14) -> OPEN_LONG_SPREAD (position 1 -> 1, V=0.2967)
  SPREAD_HIGH|SHORT (s=15) -> OPEN_SHORT_SPREAD (position -1 -> -1, V=0.3495)
  SPREAD_HIGH|FLAT (s=16) -> OPEN_SHORT_SPREAD (position 0 -> -1, V=0.3450)
  SPREAD_HIGH|LONG (s=17) -> OPEN_SHORT_SPREAD (position 1 -> -1, V=0.3410)
  SPREAD_VERY_HIGH|SHORT (s=18) -> OPEN_SHORT_SPREAD (position -1 -> -1, V=0.5086)
  SPREAD_VERY_HIGH|FLAT (s=19) -> OPEN_SHORT_SPREAD (position 0 -> -1, V=0.4958)
  SPREAD_VERY_HIGH|LONG (s=20) -> OPEN_SHORT_SPREAD (position 1 -> -1, V=0.4835)
```

---

## Q-learning (Inventory MDP)

Same MDP 5-tuple (State, Action, Transition, Reward, $\gamma$), with **inventory** as position and **action = inventory change Δ**. Solved by value iteration on empirically built $T$, $R$; run with `python main.py --inv --Q 2`.

### Action

Actions are **inventory changes** $\Delta \in \{-2Q, \ldots, 2Q\}$ (so you can flip from $-Q$ to $+Q$ or vice versa in one step). $4Q+1$ actions.

```python
# Action index a maps to delta = a - 2*Q
def action_to_delta(a: int, Q: int) -> int:
    return a - 2 * Q

def next_inventory(inventory: int, delta: int, Q: int) -> int:
    return int(np.clip(inventory + delta, -Q, Q))
```

### State

State = $(z_{\mathrm{bin}}, \mathrm{inventory})$. Inventory $\in \{-Q, \ldots, Q\}$. Total states = $n_{\mathrm{z\_bins}} \times (2Q+1)$.

```python
def make_state_index_inv(z_bin: int, inventory: int, n_z_bins: int, Q: int) -> int:
    inv_idx = inventory + Q  # 0 .. 2*Q
    return z_bin * (2 * Q + 1) + inv_idx
```

### Transition Probability

Next state $s' = (z'_{\mathrm{bin}}, \mathrm{inv}')$:

- **Inventory** is deterministic: $\mathrm{inv}' = \mathrm{clip}(\mathrm{inv} + \Delta, -Q, Q)$.
- **Z-bin** $z'_{\mathrm{bin}}$ is stochastic (market); estimated from history.

$T(s, a, s')$ is estimated by counting transitions (over all $(s, a)$ and normalizing). In code:

```python
delta = action_to_delta(a, Q)
inv_next = next_inventory(inventory, delta, Q)
s_next = make_state_index_inv(z_bin_next, inv_next, n_z_bins, Q)
T[s, a, s_next] += 1
# then normalize: T[s, a, :] /= T[s, a, :].sum()
```

### Reward Function

Reward = PnL − inventory penalty − risk penalty:

$$
r_t = \mathrm{inv} \cdot (z_{t+1} - z_t) - c \cdot |\Delta| - \lambda_{\mathrm{inv}} \cdot \mathrm{inv}^2 - \lambda_{\mathrm{risk}} \cdot (z_{t+1} - z_t)^2
$$

- $\mathrm{inv} \cdot (z_{t+1} - z_t) - c \cdot |\Delta|$: PnL (spread return in z-space minus cost per unit of $|\Delta|$).
- $\lambda_{\mathrm{inv}} \cdot \mathrm{inv}^2$: inventory penalty (`lambda_inventory`).
- $\lambda_{\mathrm{risk}} \cdot (z_{t+1} - z_t)^2$: risk penalty on squared spread return (`lambda_risk`).

$R(s, a)$ is the empirical average of $r_t$ over all $(s, a)$ transitions. In code:

```python
pnl = inventory * spread_ret - cost * abs(delta)
inv_penalty = lam_inv * (inventory ** 2)
risk_penalty = lam_risk * (spread_ret ** 2)
reward = pnl - inv_penalty - risk_penalty
R[s, a] += reward
# then R[s, a] /= count_sa[s, a]
```

### Output

```shell
Optimal policy (state = z_bin | inventory -> delta):
  SPREAD_VERY_LOW|inv=-4 (s=0) -> Δ=6 (V=-0.1717)
  SPREAD_VERY_LOW|inv=-3 (s=1) -> Δ=5 (V=-0.0546)
  SPREAD_VERY_LOW|inv=-2 (s=2) -> Δ=4 (V=0.0425)
  SPREAD_VERY_LOW|inv=-1 (s=3) -> Δ=3 (V=0.1196)
  SPREAD_VERY_LOW|inv=0 (s=4) -> Δ=2 (V=0.1767)
  SPREAD_VERY_LOW|inv=1 (s=5) -> Δ=1 (V=0.2138)
  SPREAD_VERY_LOW|inv=2 (s=6) -> Δ=0 (V=0.2310)
  SPREAD_VERY_LOW|inv=3 (s=7) -> Δ=-1 (V=0.2271)
  SPREAD_VERY_LOW|inv=4 (s=8) -> Δ=-2 (V=0.2032)
  SPREAD_LOW|inv=-4 (s=9) -> Δ=4 (V=-0.1927)
  SPREAD_LOW|inv=-3 (s=10) -> Δ=3 (V=-0.1175)
  SPREAD_LOW|inv=-2 (s=11) -> Δ=2 (V=-0.0623)
  SPREAD_LOW|inv=-1 (s=12) -> Δ=1 (V=-0.0271)
  SPREAD_LOW|inv=0 (s=13) -> Δ=0 (V=-0.0119)
  SPREAD_LOW|inv=1 (s=14) -> Δ=-1 (V=-0.0177)
  SPREAD_LOW|inv=2 (s=15) -> Δ=-2 (V=-0.0435)
  SPREAD_LOW|inv=3 (s=16) -> Δ=-3 (V=-0.0892)
  SPREAD_LOW|inv=4 (s=17) -> Δ=-4 (V=-0.1550)
  SPREAD_BELOW_MEAN|inv=-4 (s=18) -> Δ=4 (V=-0.2342)
  SPREAD_BELOW_MEAN|inv=-3 (s=19) -> Δ=3 (V=-0.1619)
  SPREAD_BELOW_MEAN|inv=-2 (s=20) -> Δ=2 (V=-0.1095)
  SPREAD_BELOW_MEAN|inv=-1 (s=21) -> Δ=1 (V=-0.0772)
  SPREAD_BELOW_MEAN|inv=0 (s=22) -> Δ=0 (V=-0.0649)
  SPREAD_BELOW_MEAN|inv=1 (s=23) -> Δ=-1 (V=-0.0736)
  SPREAD_BELOW_MEAN|inv=2 (s=24) -> Δ=-2 (V=-0.1022)
  SPREAD_BELOW_MEAN|inv=3 (s=25) -> Δ=-3 (V=-0.1509)
  SPREAD_BELOW_MEAN|inv=4 (s=26) -> Δ=-4 (V=-0.2196)
  SPREAD_NEAR_MEAN|inv=-4 (s=27) -> Δ=4 (V=-0.2182)
  SPREAD_NEAR_MEAN|inv=-3 (s=28) -> Δ=3 (V=-0.1535)
  SPREAD_NEAR_MEAN|inv=-2 (s=29) -> Δ=2 (V=-0.1087)
  SPREAD_NEAR_MEAN|inv=-1 (s=30) -> Δ=1 (V=-0.0840)
  SPREAD_NEAR_MEAN|inv=0 (s=31) -> Δ=0 (V=-0.0792)
  SPREAD_NEAR_MEAN|inv=1 (s=32) -> Δ=-1 (V=-0.0955)
  SPREAD_NEAR_MEAN|inv=2 (s=33) -> Δ=-2 (V=-0.1317)
  SPREAD_NEAR_MEAN|inv=3 (s=34) -> Δ=-3 (V=-0.1880)
  SPREAD_NEAR_MEAN|inv=4 (s=35) -> Δ=-4 (V=-0.2642)
  SPREAD_ABOVE_MEAN|inv=-4 (s=36) -> Δ=4 (V=-0.2694)
  SPREAD_ABOVE_MEAN|inv=-3 (s=37) -> Δ=3 (V=-0.1982)
  SPREAD_ABOVE_MEAN|inv=-2 (s=38) -> Δ=2 (V=-0.1470)
  SPREAD_ABOVE_MEAN|inv=-1 (s=39) -> Δ=1 (V=-0.1158)
  SPREAD_ABOVE_MEAN|inv=0 (s=40) -> Δ=0 (V=-0.1046)
  SPREAD_ABOVE_MEAN|inv=1 (s=41) -> Δ=-1 (V=-0.1144)
  SPREAD_ABOVE_MEAN|inv=2 (s=42) -> Δ=-2 (V=-0.1442)
  SPREAD_ABOVE_MEAN|inv=3 (s=43) -> Δ=-3 (V=-0.1940)
  SPREAD_ABOVE_MEAN|inv=4 (s=44) -> Δ=-4 (V=-0.2638)
  SPREAD_HIGH|inv=-4 (s=45) -> Δ=4 (V=-0.2736)
  SPREAD_HIGH|inv=-3 (s=46) -> Δ=3 (V=-0.2072)
  SPREAD_HIGH|inv=-2 (s=47) -> Δ=2 (V=-0.1607)
  SPREAD_HIGH|inv=-1 (s=48) -> Δ=1 (V=-0.1342)
  SPREAD_HIGH|inv=0 (s=49) -> Δ=0 (V=-0.1277)
  SPREAD_HIGH|inv=1 (s=50) -> Δ=-1 (V=-0.1422)
  SPREAD_HIGH|inv=2 (s=51) -> Δ=-2 (V=-0.1767)
  SPREAD_HIGH|inv=3 (s=52) -> Δ=-3 (V=-0.2313)
  SPREAD_HIGH|inv=4 (s=53) -> Δ=-4 (V=-0.3058)
  SPREAD_VERY_HIGH|inv=-4 (s=54) -> Δ=3 (V=-0.3042)
  SPREAD_VERY_HIGH|inv=-3 (s=55) -> Δ=2 (V=-0.2460)
  SPREAD_VERY_HIGH|inv=-2 (s=56) -> Δ=1 (V=-0.2078)
  SPREAD_VERY_HIGH|inv=-1 (s=57) -> Δ=0 (V=-0.1896)
  SPREAD_VERY_HIGH|inv=0 (s=58) -> Δ=-1 (V=-0.1924)
  SPREAD_VERY_HIGH|inv=1 (s=59) -> Δ=-2 (V=-0.2152)
  SPREAD_VERY_HIGH|inv=2 (s=60) -> Δ=-3 (V=-0.2580)
  SPREAD_VERY_HIGH|inv=3 (s=61) -> Δ=-4 (V=-0.3208)
  SPREAD_VERY_HIGH|inv=4 (s=62) -> Δ=-5 (V=-0.4036)
```