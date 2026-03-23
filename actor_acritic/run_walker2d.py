import json
from pathlib import Path

from a2c_continuous import A2CConfig, Algorithm, plot_env_curves, train_a2c


ENV_ID = "Walker2d-v4"
OUTPUT_DIR = Path("actor_acritic/results_walker2d")
COMMON_NUM_ENVS = 8
COMMON_ROLLOUT_LEN = 128
COMMON_TOTAL_STEPS = 200_000
COMMON_TOTAL_UPDATES = COMMON_TOTAL_STEPS // (COMMON_NUM_ENVS * COMMON_ROLLOUT_LEN)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Synced from actor_acritic/results_full_rerun/summary.json
    best_cfg_by_algo: dict[Algorithm, dict] = {
        "td1": {
            "env_id": ENV_ID,
            "algo": "td1",
            "total_updates": 200,
            "rollout_len": 128,
            "num_envs": 4,
            "policy_lr": 3e-4,
            "value_lr": 5e-4,
            "gamma": 0.97,
            "lam": 0.0,
            "hidden_size": 128,
            "value_coef": 0.5,
            "entropy_coef": 0.0,
            "max_grad_norm": 0.5,
            "seed": 42,
            "device": "cpu",
        },
        "mc": {
            "env_id": ENV_ID,
            "algo": "mc",
            "total_updates": 200,
            "rollout_len": 128,
            "num_envs": 8,
            "policy_lr": 3e-4,
            "value_lr": 1e-3,
            "gamma": 0.99,
            "lam": 1.0,
            "hidden_size": 128,
            "value_coef": 0.5,
            "entropy_coef": 0.0,
            "max_grad_norm": 0.5,
            "seed": 42,
            "device": "cpu",
        },
        "gae": {
            "env_id": ENV_ID,
            "algo": "gae",
            "total_updates": 200,
            "rollout_len": 128,
            "num_envs": 8,
            "policy_lr": 3e-4,
            "value_lr": 5e-4,
            "gamma": 0.99,
            "lam": 0.95,
            "hidden_size": 128,
            "value_coef": 0.5,
            "entropy_coef": 0.0,
            "max_grad_norm": 0.5,
            "seed": 42,
            "device": "cpu",
        },
    }

    compare_results: dict[Algorithm, dict] = {}
    for algo in ("td1", "mc", "gae"):
        cfg = A2CConfig(**best_cfg_by_algo[algo])
        cfg.num_envs = COMMON_NUM_ENVS
        cfg.rollout_len = COMMON_ROLLOUT_LEN
        cfg.total_updates = COMMON_TOTAL_UPDATES
        compare_results[algo] = train_a2c(cfg)

    plot_env_curves(ENV_ID, compare_results, OUTPUT_DIR)

    summary = {
        "env_id": ENV_ID,
        "best_config": best_cfg_by_algo,
        "compare_score": {k: v["score"] for k, v in compare_results.items()},
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
