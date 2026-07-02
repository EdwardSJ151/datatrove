# examples/

## Purpose

Runnable pipeline templates (not production configs): Common Crawl / FineWeb processing, dedup, tokenization, stats, HF dataset filtering, bucket I/O, and synthetic LLM generation. Each script wires `src/datatrove` blocks to an executor with placeholder paths.

## Files (root level)

| Script | Workflow | Executor |
|--------|----------|----------|
| `fineweb.py` | Full FineWeb reproduction (WARC → filters → minhash → PII) | Slurm, chained `depends=` |
| `process_common_crawl_dump.py` | Slimmer CC dump (extract + Gopher + language) | Slurm |
| `tokenize_c4.py` | C4 HF → tokenize → merge | Slurm, 2 stages |
| `tokenize_from_hf_to_s3.py` | Generalized HF tokenization CLI | Slurm, 2 stages |
| `estimate_tokens.py` | Sample HF datasets for token rate estimates | None (standalone) |
| `smol_data.py` | Build smol-data subsets/mixtures/shuffles | Slurm + raw sbatch for shuffle |
| `minhash_deduplication.py` | 4-stage MinHash on S3 jsonl | Slurm, 4 stages |
| `sentence_deduplication.py` | Sentence-level dedup | Local, 3 stages |
| `exact_substrings.py` | ExactSubstr + external Rust tool | Local |
| `url_deduplication.py` | URL exact dedup | Local, CLI paths |
| `summary_stats.py` | Sampled doc/word/line stats + merge | Slurm, 2 stages |
| `filter_hf_dataset.py` | Filter HF parquet → push hub | Slurm |
| `bucket_synthetic_data.py` | HF bucket I/O patterns (demo) | Local (`.run()` commented) |

## Core concepts

**Templates, not turnkey.** Paths like `s3://some_s3_bucket`, partitions (`hopper-cpu`), and org names are placeholders — replace before running.

**Multi-stage Slurm.** Dedup, tokenization, FineWeb, and stats use `depends=` between executors. Task counts must match between signature and filter stages (see [dedup CLAUDE.md](../src/datatrove/pipeline/dedup/CLAUDE.md)).

**Executor matrix.**

| Pattern | Scripts |
|---------|---------|
| Local only | `sentence_deduplication.py`, `exact_substrings.py`, `url_deduplication.py` |
| Slurm only | `fineweb.py`, `process_common_crawl_dump.py`, `tokenize_*.py`, `minhash_deduplication.py`, `summary_stats.py`, `filter_hf_dataset.py`, `smol_data.py` |
| Local or Slurm | `inference/*` via `generate_data` flags |

### Synthetic inference (`inference/`)

| File | Role |
|------|------|
| `generate_data.py` | Main Typer CLI: HF reader → vLLM → Parquet on Hub + checkpoints |
| `generate_data2.py` | Duplicate of `generate_data.py` (benchmark path compatibility) |
| `finephrase.py` / `finephrase_en.py` | Multi-template FinePhrase launchers |
| `inference_chunked.py` | Custom rollouts, checkpoints, local + Slurm |
| `progress_monitoring.py` | `InferenceProgressMonitor` demo |
| `utils.py` | HF auth, `build_run_path`, config validation |

**Slurm 3-job architecture:** inference (GPU) → optional monitor (CPU) → datacard (CPU, `afterok`). **`output_dir`** is run metadata (logs/checkpoints), not Parquet — output goes to `hf://datasets/...`. Requires `uv sync --extra inference` and HF write token.

**Benchmark sweep (`inference/benchmark/`):** `launch_experiments.py` expands YAML → Cartesian product → `generate_data.main`; `analyze_results.py` parses throughput/GPU-days. Run dirs mirror `build_run_path()` encoding. Use YAML tiers to avoid combinatorial explosion; `benchmark_mode=True` skips HF output (perf tuning only).

## Complexities and pitfalls

- **`fineweb.py` scale:** 8000-task base job; `randomize_start_duration=180` avoids S3 list storms; minhash stage 2 uses `num_buckets * 50` workers.
- **`exact_substrings.py`:** Requires external Google Research tool between stages 2 and 3.
- **`smol_data.py shuffle`:** Uses raw `sbatch` + `datasets.push_to_hub`, not DataTrove pipeline.
- **`estimate_tokens.py`:** No datatrove imports — planning helper for `smol_data.py`.
- **Inference:** JSONL input / dual local+HF output not in CLI today; checkpointing needs `${chunk_index}` in writer template.

## See also

- [../src/datatrove/pipeline/CLAUDE.md](../src/datatrove/pipeline/CLAUDE.md) — block reference
- [../src/datatrove/pipeline/inference/CLAUDE.md](../src/datatrove/pipeline/inference/CLAUDE.md) — inference internals
- [../src/datatrove/executor/CLAUDE.md](../src/datatrove/executor/CLAUDE.md) — executor behavior
- [../CLAUDE.md](../CLAUDE.md) — repo navigation
- [../AGENTS.md](../AGENTS.md) — dev commands
- [inference/README.md](inference/README.md) — install and Slurm details
