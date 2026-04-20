# REINFORCE Video Lecture Script

---

## Section 1: Introduction (~ 2 min)

Hey everyone! Today we're going to walk through one of the most important algorithms in Reinforcement Learning — **REINFORCE**.

This algorithm is the foundation of a whole family of methods called **Policy Gradient** methods, which power a lot of modern AI systems you might have heard of.

By the end of this video, you'll understand:
- What problem REINFORCE is trying to solve
- The full mathematical derivation — step by step
- And how to implement it in PyTorch

Don't worry if you haven't seen much RL before. We'll build everything up from scratch.

---

## Section 2: The Setup — What is RL trying to do? (~ 3 min)

Let's start with the big picture.

In Reinforcement Learning, we have an **agent** interacting with an **environment**.

At every timestep $t$:
1. The agent sees a **state** $S_t$
2. It picks an **action** $A_t$ according to its **policy** $\pi$
3. The environment gives back a **reward** $R_t$ and a new state $S_{t+1}$

The agent's goal is simple: **collect as much reward as possible over time**.

We define the **return** $U_t$ as the discounted sum of all future rewards from time $t$:

$$U_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots$$

Where $\gamma$ (gamma) is a number between 0 and 1 called the **discount factor**. It makes rewards in the near future matter more than rewards far away — kind of like how you'd rather have $100 today than $100 in 10 years.

---

## Section 3: The Policy and the Objective (~ 3 min)

Now — what is the policy $\pi$?

The policy is the agent's **decision-making function**. We parameterize it with a neural network with parameters $\theta$, so we write it as $\pi_\theta$.

For a given state $s$, the policy outputs a **probability distribution over actions**:

$$\pi_\theta(a | s) = P(\text{take action } a \mid \text{in state } s)$$

Our goal is to find the parameters $\theta$ that make the agent perform as well as possible.

We measure "performance" with the **objective function** $J(\theta)$:

$$J(\theta) = \mathbb{E}_{\pi_\theta}[U_t]$$

In plain English: **the expected total reward when the agent follows policy $\pi_\theta$**.

We want to **maximize** $J(\theta)$. So we'll use **gradient ascent**:

$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$

The only question is: how do we compute $\nabla_\theta J(\theta)$?

That's exactly what the **Policy Gradient Theorem** tells us.

---

## Section 4: The Mathematical Derivation (~ 10 min)

Okay, this is the heart of the lecture. Let's derive the policy gradient step by step.

---

### Step 1: Write out the expectation explicitly

$$J(\theta) = \mathbb{E}_{\pi_\theta}[U_t] = \sum_s \mu(s) \sum_a \pi_\theta(a|s) \cdot Q_\pi(s, a)$$

Where:
- $\mu(s)$ is how often we visit state $s$ under policy $\pi_\theta$
- $Q_\pi(s, a)$ is the **action-value function** — the expected return if we're in state $s$, take action $a$, and follow $\pi_\theta$ from then on

We want $\nabla_\theta J(\theta)$. The tricky part is that $\mu(s)$ also depends on $\theta$ — but the **Policy Gradient Theorem** tells us we can actually ignore that dependency. We'll take that as given here.

---

### Step 2: Take the gradient

$$\nabla_\theta J(\theta) = \sum_s \mu(s) \sum_a \nabla_\theta \pi_\theta(a|s) \cdot Q_\pi(s, a)$$

We only differentiate $\pi_\theta(a|s)$ because $Q_\pi$ doesn't directly depend on $\theta$ in this expression.

---

### Step 3: The log-derivative trick

Here's the key mathematical trick. For any function $f$:

$$\nabla_\theta f = f \cdot \nabla_\theta \log f$$

This comes from the chain rule: $\nabla \log f = \frac{\nabla f}{f}$, so $\nabla f = f \cdot \nabla \log f$.

Applying this to $\pi_\theta(a|s)$:

$$\nabla_\theta \pi_\theta(a|s) = \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s)$$

Why do we do this? Because now we have $\pi_\theta(a|s)$ sitting inside the sum — which means we can turn it back into an expectation!

---

### Step 4: Substitute back

$$\nabla_\theta J(\theta) = \sum_s \mu(s) \sum_a \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s) \cdot Q_\pi(s, a)$$

This double sum weighted by $\mu(s)$ and $\pi_\theta(a|s)$ is exactly the definition of an expectation over states and actions sampled from the policy:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(A_t|S_t) \cdot Q_\pi(S_t, A_t) \right]$$

---

### Step 5: Replace $Q$ with the actual return $U_t$

In practice, we don't know the true $Q_\pi(s, a)$. But remember — $Q_\pi(s,a)$ is defined as the **expected return** starting from $(s, a)$.

So we can use the actual sampled return $U_t$ as an **unbiased estimate** of $Q_\pi(S_t, A_t)$:

$$\nabla_\theta J(\theta) \approx \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(A_t|S_t) \cdot U_t \right]$$

This is the **REINFORCE gradient estimator**.

---

### Intuition check — what does this formula say?

$$\nabla_\theta \log \pi_\theta(A_t|S_t) \cdot U_t$$

- If $U_t$ is **large** (the episode went well): we push $\theta$ to make action $A_t$ **more likely** in state $S_t$
- If $U_t$ is **small or negative** (the episode went badly): we push $\theta$ to make $A_t$ **less likely**

In other words: **reinforce actions that led to good outcomes, suppress actions that led to bad ones**. That's where the name comes from!

---

## Section 5: The REINFORCE Algorithm (~ 3 min)

Now let's write out the full algorithm:

```
Initialize policy network π_θ randomly

Repeat:
  1. Run one full episode using π_θ
     → collect (S_0, A_0, R_0, S_1, A_1, R_1, ..., S_T)

  2. For each timestep t:
     → compute return U_t = R_t + γR_{t+1} + γ²R_{t+2} + ...

  3. For each timestep t:
     → compute gradient estimate: ∇θ log π_θ(A_t|S_t) · U_t

  4. Update parameters:
     θ ← θ + α · Σ_t ∇θ log π_θ(A_t|S_t) · U_t
```

Key point: we **must complete a full episode** before updating — because we need the full return $U_t$.

---

## Section 6: PyTorch Implementation (~ 8 min)

Let's now see how this translates to code. We'll use **CartPole-v1** from Gymnasium — the classic "balance a pole on a cart" environment.

### Policy Network

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
            # No softmax here — we'll use Categorical which handles it
        )

    def forward(self, x):
        return self.net(x)  # Returns logits
```

The network takes in a state and outputs **logits** (raw scores) for each action. Remember — logits go through softmax to become probabilities.

---

### Collecting an Episode

```python
def collect_episode(env, policy, gamma=0.99):
    state, _ = env.reset()
    log_probs = []
    rewards = []

    done = False
    while not done:
        state_tensor = torch.FloatTensor(state)
        logits = policy(state_tensor)

        # Create a probability distribution over actions
        dist = Categorical(logits=logits)

        # Sample an action
        action = dist.sample()

        # This is log π_θ(A_t | S_t) — we'll need it for the gradient
        log_prob = dist.log_prob(action)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        log_probs.append(log_prob)
        rewards.append(reward)
        state = next_state

    # Compute discounted returns U_t for each timestep
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)

    return log_probs, returns
```

Notice how we compute returns **backwards** — that's the efficient way to accumulate $U_t = R_t + \gamma U_{t+1}$.

---

### The Update Step

```python
def update(optimizer, log_probs, returns):
    # Normalize returns (optional but helps training stability)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Compute the loss
    # Note: we MINIMIZE loss, so we negate the gradient ascent objective
    loss = []
    for log_prob, G in zip(log_probs, returns):
        loss.append(-log_prob * G)  # negative because we want gradient ASCENT

    loss = torch.stack(loss).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

This is the key line:

```python
loss = -log_prob * G
```

- `log_prob` is $\log \pi_\theta(A_t | S_t)$
- `G` is $U_t$
- The negative sign turns gradient **ascent** into gradient **descent** (which is what PyTorch's optimizer does)

When we call `loss.backward()`, PyTorch automatically computes $\nabla_\theta \log \pi_\theta(A_t|S_t) \cdot U_t$ for us!

---

### Training Loop

```python
env = gym.make("CartPole-v1")
policy = PolicyNetwork(state_dim=4, action_dim=2)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

for episode in range(1000):
    log_probs, returns = collect_episode(env, policy)
    update(optimizer, log_probs, returns)

    if episode % 100 == 0:
        print(f"Episode {episode}")
```

---

## Section 7: Limitations and What Comes Next (~ 2 min)

REINFORCE works, but it has two main weaknesses:

**1. High variance**
Because we use the raw return $U_t$, the gradient estimates are noisy. Two identical situations can give very different $U_t$ values depending on luck. This makes training slow and unstable.

**Fix:** Subtract a **baseline** (usually $V(s)$) from $U_t$:

$$\nabla_\theta J(\theta) \approx \mathbb{E}\left[\nabla_\theta \log \pi_\theta(A_t|S_t) \cdot (U_t - V(S_t))\right]$$

This is the motivation for **Actor-Critic** methods.

**2. Must wait for full episode**
We can't update until the episode ends, which is slow for long tasks.

**Fix:** Use **Temporal Difference** estimates instead of full returns — this leads to algorithms like **A2C** and **PPO**.

---

## Section 8: Summary (~ 1 min)

Let's recap what we covered:

| Concept | Key idea |
|---|---|
| Policy $\pi_\theta$ | Neural net that outputs action probabilities |
| Objective $J(\theta)$ | Expected total reward — we want to maximize this |
| Log-derivative trick | Converts $\nabla \pi$ into an expectation we can sample |
| REINFORCE gradient | $\mathbb{E}[\nabla \log \pi_\theta(A_t\|S_t) \cdot U_t]$ |
| PyTorch | `loss = -log_prob * G`, then `loss.backward()` |

REINFORCE is the simplest policy gradient algorithm, and understanding it deeply gives you the foundation to understand everything that comes after — Actor-Critic, PPO, and beyond.

Thanks for watching!