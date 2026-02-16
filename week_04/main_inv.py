from load_data import load_candles, compute_spread_zscore
from mdp import (
    InventoryMDPConfig,
    build_mdp_from_series_inv,
    value_iteration,
    backtest_inv,
    action_to_delta,
    state_index_to_name_inv,
    plot_policy_heatmap_inv,
)

def main_inv(
    lookback: int = 24 * 30,
    n_z_bins: int = 7,
    Q: int = 2,
    gamma: float = 0.99,
    cost_per_unit: float = 0.0005,
    lambda_inventory: float = 0.01,
    lambda_risk: float = 0.1,
):
    """Run inventory MDP: action = Δ ∈ {-2Q,...,2Q}, reward = pnl - inventory_penalty - risk_penalty."""
    config = InventoryMDPConfig(
        Q=Q,
        n_z_bins=n_z_bins,
        gamma=gamma,
        cost_per_unit=cost_per_unit,
        lambda_inventory=lambda_inventory,
        lambda_risk=lambda_risk,
    )

    print("Loading BTC/ETH 1h candles...")
    candles = load_candles()
    print(f"Candles from {candles['open_time'].min()} to {candles['open_time'].max()}")

    z = compute_spread_zscore(candles, lookback=lookback)
    valid = z.notna()
    print(f"Valid z-score count: {valid.sum()}")

    print(f"Building inventory MDP (Q={Q}, action = Δ ∈ [{-2*Q}, {2*Q}], λ_inv={config.lambda_inventory}, λ_risk={config.lambda_risk})...")
    T, R, count_sa = build_mdp_from_series_inv(z, config)
    print("Value iteration...")
    V, policy = value_iteration(T, R, gamma=config.gamma)

    print("\nOptimal policy (state = z_bin | inventory -> delta):")
    for s in range(config.n_states):
        state_name = state_index_to_name_inv(s, config.n_z_bins, config.Q)
        delta = action_to_delta(policy[s], config.Q)
        print(f"  {state_name} (s={s}) -> Δ={delta} (V={V[s]:.4f})")

    plot_policy_heatmap_inv(policy, config, save_path="policy_heatmap_inv.png")

    print("\nBacktesting (inventory)...")
    bt = backtest_inv(candles, z, policy, config)
    total_pnl = bt["pnl"].sum()
    cum = bt["cum_pnl"]
    print(f"Total PnL (spread units): {total_pnl:.4f}")
    print(f"Cumulative PnL end: {cum.iloc[-1]:.4f}")
    if cum.max() != cum.min():
        peak = cum.expanding().max()
        dd = cum - peak
        print(f"Max drawdown: {dd.min():.4f}")

    return {"candles": candles, "z": z, "T": T, "R": R, "V": V, "policy": policy, "backtest": bt}


if __name__ == "__main__":
    main_inv(Q=4)