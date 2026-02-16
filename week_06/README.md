# Online Planning Assignment

### Part 0

MCTS-RAG: Enhancing Retrieval-Augmented Generation with Monte Carlo Tree Search

https://arxiv.org/abs/2503.20757

### Part 1

Replicating baseline (CoT, Standard RAG) with Qwen2.5-7B and Llama3.1-8B (currently awaiting approval from Hugging Face)

I will update the data after I get approval.

```shell
pip install requirements.txt
python run_cot.py -m Qwen/Qwen2.5-7B
python run_standard_rag.py -m Qwen/Qwen2.5-7B
```

Answer accuracy 

| Method | Qwen2.5-7B | | | Llama 3.1-8B | | |
|:-------|:----:|:----:|:----:|:----:|:----:|:----:|
| | CWQA | GPQA | FMT | CWQA | GPQA | FMT |
| CoT | 24.0 | 38.0 | 38.0 | todo | todo | todo |
| Standard RAG | 26.0 | 31.0 | 62.0 | todo | todo | todo |

