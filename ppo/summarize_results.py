import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize FourRooms RL results in Table-1 style.")
    parser.add_argument("--input", type=str, default="ppo/results_fourrooms.json")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    payload = json.loads(path.read_text())
    runs = payload.get("runs", {})

    print("\nEnvironment: Four Rooms")
    print("| Algorithm | (avg_reward, success_rate%) |")
    print("| --- | --- |")
    for key in ("reinforce", "ppo", "poly_ppo"):
        if key not in runs:
            continue
        m = runs[key]["eval"]
        avg_reward = float(m["avg_reward"])
        success_rate = float(m["success_rate"]) * 100.0
        pretty_key = {
            "reinforce": "REINFORCE",
            "ppo": "PPO",
            "poly_ppo": "Poly-PPO",
        }[key]
        print(f"| {pretty_key} | ({avg_reward:.3f}, {success_rate:.1f}) |")


if __name__ == "__main__":
    main()
