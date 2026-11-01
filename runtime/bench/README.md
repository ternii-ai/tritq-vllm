# Speed benchmarks

| tool | what it measures | where |
|---|---|---|
| `bench_oai.py` (ours) | decode tokens/s per stream after the first token, 8 prompts (maths, code, prose) x 2 repeats, greedy, thinking off, up to 512 tokens, at 1 / 4 / 8 concurrent streams; aggregate = tokens / wall time | `python3 bench_oai.py http://127.0.0.1:8000 <model> <label> 1,4,8 2 >> results.jsonl` |
| community harness (`run_community_harness.sh`) | tonyd2wild `bench_categories.py` @6ad1c8f, unchanged: 40 short prompts in 8 categories, greedy, up to 900 tokens; figure = median prompt, median of 3 runs | see the script header |

Protocol for any comparison we publish: same machine and session, servers interleaved, two passes in opposite order (ABBA), report mean
AND median, state the concurrency, the speculative mode and every server flag. Name the comparator by Hugging Face repo and revision.
