from load_data import load_candles, compute_spread_zscore
from mdp import (
    MDPConfig,
    build_mdp_from_series,
    value_iteration,
    backtest,
    Action,
    state_index_to_name,
    state_index_to_position,
)

def main(
    lookback: int = 24 * 30,
    n_z_bins: int = 7,
    gamma: float = 0.99,
):
    config = MDPConfig(n_z_bins=n_z_bins, gamma=gamma)

    print("Loading BTC/ETH 1h candles...")
    candles = load_candles()
    print(f"Candles from {candles['open_time'].min()} to {candles['open_time'].max()}")

    z = compute_spread_zscore(candles, lookback=lookback)
    valid = z.notna()
    print(f"Valid z-score count: {valid.sum()}, mean z (later) = {z[valid].tail(5000).mean():.4f}")

    print("Building MDP from history...")
    T, R, count_sa = build_mdp_from_series(z, config)
    print("Value iteration...")
    V, policy = value_iteration(T, R, gamma=config.gamma)

    print("\nOptimal policy (state = z_bin | position -> action):")
    for s in range(config.n_states):
        state_name = state_index_to_name(s, config.n_z_bins)
        act = Action(policy[s])
        pos = state_index_to_position(s)
        next_pos = act.next_position(pos)
        print(f"  {state_name} (s={s}) -> {act.name} (position {pos} -> {next_pos}, V={V[s]:.4f})")

    print("\nBacktesting...")
    bt = backtest(candles, z, policy, config)
    total_pnl = bt["pnl"].sum()
    cum = bt["cum_pnl"]
    print(f"Total PnL (spread units): {total_pnl:.4f}")
    print(f"Cumulative PnL end: {cum.iloc[-1]:.4f}")
    if cum.max() != cum.min():
        peak = cum.expanding().max()
        dd = (cum - peak)
        print(f"Max drawdown: {dd.min():.4f}")

    return {
        "candles": candles,
        "z": z,
        "T": T,
        "R": R,
        "V": V,
        "policy": policy,
        "backtest": bt,
    }




if __name__ == "__main__":
    main()