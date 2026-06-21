# Numbers

Every rate in the README comes from `results/summary.json`, which is generated
from the run logs in this directory by `python -m ontime.bench.analysis`. The
logs are the evidence. Re-running the analysis reproduces the rates.

## Headline rates from the logs

| Rate | Value | Source |
| --- | --- | --- |
| Sonnet natural-language refusal | 0% | `full_run_sonnet_tool_nl_n*.jsonl` |
| Qwen natural-language refusal | 7.5% (6 of 80) | `full_run_qwen_tool_nl_n*.jsonl` |
| Qwen natural-language engage rate | 92.5% | `full_run_qwen_tool_nl_n*.jsonl` |
| Sonnet natural-language correctness | 95.8% | `full_run_sonnet_tool_nl_n*.jsonl` |
| Sonnet trivial-copy correctness | 100% | `full_run_sonnet_tool_trivial_n*.jsonl` |
| Qwen trivial-copy correctness | 98.75% | `full_run_qwen_tool_trivial_n*.jsonl` |
| Sonnet marshaling oracle match (natural language) | 72.5% | `full_run_sonnet_tool_nl_n*.jsonl` |
| Sonnet self-review fire rate | 34.6% (83 of 240) | `full_run_sonnet_tool_nl_n*.jsonl` |
| Schema floor correctness at n=16 and n=20 | 0% | `schema_diag_n16.jsonl`, `schema_diag_n20.jsonl` |

The interface jump lives in the marshaling layer. When the numbers are handed to
the model already shaped, the trivial-copy path reaches the optimum on every
instance. When the model reads the same problem from prose, the marshaling oracle
match drops to 72.5% for Sonnet, and correctness drops with it.

## Two figures that differ from an earlier draft

Two rates were stated differently in an earlier draft of the article. The logs do
not support those earlier figures, so the repo uses the log values and records the
difference here.

- Refusal rate on raw text. An earlier draft put this around 15%. The logs show
  0% for Sonnet and 7.5% for Qwen. The 7.5% is the open-7B figure to cite.
- Self-review fire rate. An earlier draft called it about a quarter. The logs show
  34.6% across the Sonnet natural-language runs, and it climbs with problem size
  to roughly 45% at the larger sizes. About a third is the figure to cite, with
  the note that it rises with size.

Reconcile the article prose with these two values before publishing.
