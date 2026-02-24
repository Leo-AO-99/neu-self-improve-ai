# Online Planning Assignment

### Part 0

MCTS-RAG: Enhancing Retrieval-Augmented Generation with Monte Carlo Tree Search

https://arxiv.org/abs/2503.20757

### Part 1

Replicating baseline (CoT, Standard RAG) with Qwen2.5-7B and Llama3.1-8B

```shell
pip install requirements.txt
python run_cot.py -m Qwen/Qwen2.5-7B
python run_standard_rag.py -m Qwen/Qwen2.5-7B
python run_mcts_rag.py -m Qwen/Qwen2.5-7B --num_rollouts 8 --num_votes 8
```

### Part 2: MCTS-RAG

Implementation of [MCTS-RAG](https://github.com/yale-nlp/MCTS-RAG): MCTS with actions A1 (Direct Answer) and A4 (Retrieval Reasoning). Final answer is chosen by reward-weighted voting over rollout trajectories.

```shell
python run_mcts_rag.py -m <model_ckpt> -d cwebqa -n 5   # limit 5 samples
python run_mcts_rag.py -m <model_ckpt> --num_rollouts 8 --num_votes 8
```

Options: `--retriever wikipedia|duckduckgo`, `--num_rollouts`, `--num_votes`, `--top_k`, `--disable_a1`, `--disable_a4`.

Answer accuracy 

| Method | Qwen2.5-7B | | | Llama 3.1-8B | | |
|:-------|:----:|:----:|:----:|:----:|:----:|:----:|
| | CWQA | GPQA | FMT | CWQA | GPQA | FMT |
| CoT | 24.0 | 38.0 | 38.0 | 47.0 | 19.0 | 42.5 |
| Standard RAG | 25.0 | 41.0 | 40.0 | 48.0 | 21.0 | 50.5 |
| MCTS-RAG | 39.0 | 52.0 | 50.0 | 51.0 | 30.0 | 55.5 |