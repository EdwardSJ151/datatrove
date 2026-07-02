# inference/

## Purpose

LLM inference at scale: async rollouts per document, server lifecycle (vLLM/SGLang/endpoint/dummy/custom), multi-node Ray, checkpoint/resume, metrics, Hugging Face dataset cards, and progress monitoring.

## Files

| Module | Class(es) | Role |
|--------|-----------|------|
| `run_inference.py` | `InferenceConfig`, `InferenceRunner` | Core async inference pipeline block |
| `types.py` | `InferenceResult`, `RolloutFunction`, errors | Rollout types and errors |
| `checkpointing.py` | `CheckpointManager`, `RequestCache` | Chunk checkpoints + request dedup cache |
| `metrics.py` | `MetricsKeeper`, `QueueSizesKeeper` | Throughput / queue metrics |
| `progress_monitor.py` | `InferenceProgressMonitor` | Live HF dataset card updates |
| `dataset_card_generator.py` | `InferenceDatasetCardGenerator`, `InferenceDatasetCardParams` | Final dataset README |
| `dataset_card_template.md` | Template | Card markdown skeleton |

### Inference servers (`servers/`)

| Module | Role |
|--------|------|
| `base.py` | `InferenceServer` ABC — start, stop, readiness, generate |
| `vllm_server.py` | Local vLLM subprocess |
| `sglang_server.py` | SGLang backend |
| `endpoint_server.py` | Remote OpenAI-compatible API (no local GPU) |
| `dummy_server.py` | Test server |
| `custom_server.py` | User hook |
| `compile_lock.py` | Prevent concurrent vLLM compiles on shared nodes |

`InferenceConfig` maps `tp`, `dp`, `pp` to backend-specific CLI flags. Multi-node vLLM forces Ray backend when TP spans nodes.

### Distributed Ray (`distributed/`)

| Module | Role |
|--------|------|
| `ray.py`, `utils.py` | Attach vLLM to Ray cluster; IP discovery |

Used when Slurm `nodes_per_task > 1`: `generate_data.py` sets `distributed-executor-backend=ray` and `RAY_ADDRESS=auto`. General pipeline Ray execution lives in `executor/ray.py`, not here.

## Core concepts

**Rollout function.** Async callable `(document, generate, **shared_context) -> RolloutResult`. `generate(payload)` sends one request to the server. Results stored under `document.metadata[metadata_key]` (default `rollout_results`).

**`InferenceRunner`.** Bounded concurrency via `max_concurrent_generations` and `max_concurrent_documents`. Uses thread pool + asyncio. Writes through a single `DiskWriter` with optional chunk checkpointing.

**Checkpointing.** When `checkpoints_local_dir` is set, documents are saved to local JSONL chunks (`records_per_chunk` docs). Output filename must include `${chunk_index}`. `RequestCache` (sqlite + xxhash) deduplicates rollouts on retry. Failed tasks resume from `last_chunk/{rank}.txt`.

**`skip_bad_requests=True`.** Provider bad requests (e.g. context overflow) skip output but still advance checkpoints when configured — prevents stuck chunks at scale.

**Dataset card pipeline.** Separate executor job reads `stats.json` and uploads README to HF dataset repo.

## Complexities and pitfalls

- **One `output_writer` only** — dual local+HF output requires a custom wrapper writer.
- **Readiness timeouts:** Large models need sufficient Slurm `time` for vLLM compile + inference.
- **Pickled rollouts on Slurm:** Imports inside rollout body if needed; set `PYTHONPATH` to `examples/inference` in Slurm `env_command`.
- **Multi-node Ray:** GPUs must be allocated per node; network/IP discovery failures stall startup.
- **Global `max_examples`:** Split across tasks in `generate_data.py` — per-rank reader limit is `(max_examples + tasks - 1) // tasks`.
- **Local execution:** Progress monitor is skipped (would block); datacard runs inline after inference.

## Examples that touch this code

- `examples/inference/generate_data.py`, `generate_data2.py` — full production path
- `examples/inference/finephrase.py`, `finephrase_en.py` — wrappers around `generate_data.main`
- `examples/inference/inference_chunked.py` — custom rollouts + checkpoints
- `examples/inference/progress_monitoring.py` — monitor + datacard jobs
- `examples/inference/benchmark/` — YAML hyperparameter sweeps

## See also

- [../writers/CLAUDE.md](../writers/CLAUDE.md) — Parquet/JSONL output
- [../readers/CLAUDE.md](../readers/CLAUDE.md) — HF / JSONL input
- [../../executor/CLAUDE.md](../../executor/CLAUDE.md) — Slurm 3-job architecture
- [../../../examples/CLAUDE.md](../../../examples/CLAUDE.md) — inference scripts section
