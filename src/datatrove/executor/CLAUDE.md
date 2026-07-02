# executor/

## Purpose

Run pipeline steps across ranks (tasks): local multiprocessing, Slurm job arrays, or Ray clusters. Owns completion tracking, per-rank logging, and stats serialization.

## Files

| Module | Class(es) | Role |
|--------|-----------|------|
| `base.py` | `PipelineExecutor` | Abstract run loop, completions, stats |
| `local.py` | `LocalPipelineExecutor` | Multiprocess on one machine |
| `slurm.py` | `SlurmPipelineExecutor` | `sbatch` array jobs, pickling, dependencies |
| `ray.py` | `RayPipelineExecutor`, `RankWorker`, `RayTaskManager` | Ray actors + placement groups |
| `__init__.py` | Exports | Public API |

## Core concepts

**`_run_for_rank(rank)`.** Sets `rank` / `world_size` env, chains pipeline steps (each step is `PipelineStep` or callable `(data, rank, world_size) -> generator`), aggregates `PipelineStats`, writes artifacts.

**`logging_dir` layout (per run):**
- `executor.json` — pipeline metadata, `world_size`
- `logs/{rank:05d}.log`
- `stats/{rank:05d}.json`
- `completions/{rank:05d}` — empty marker when rank finishes successfully

**`skip_completed=True`.** Skips ranks whose completion file already exists — enables relaunch after failures.

**Pipeline step chaining.** First step receives `data=None`; each subsequent step receives the generator from the previous step.

**Distributed env.** `get_distributed_env()` sets `datatrove_*` variables for multi-node local/Ray/Slurm workers.

## Complexities and pitfalls

- **Unique `logging_dir` per pipeline** — reuse overwrites stats and completion state.
- **Never change `tasks` on partial relaunch** — file shards and completions are rank-indexed.
- **`LocalPipelineExecutor` with `workers==1`:** Deep-copies pipeline per rank to avoid shared mutable state.
- **`SlurmPipelineExecutor`:** Must not launch from inside a compute node; pickles executor with `dill`; worker entry is `launch_pickled_pipeline`; supports `depends=`, `max_array_size` splitting, `tasks_per_job`, requeue on signals.
- **`RayPipelineExecutor`:** Placement groups, timeout/retry; spread across nodes not guaranteed when `nodes_per_task >= 2`.
- **Stats saved on master node only** in some distributed modes; use `merge_stats` tool post-run.
- **Raw callables in pipeline** must match signature or `ValueError` is raised.

## Examples that touch this code

- Local: `sentence_deduplication.py`, `url_deduplication.py`, `exact_substrings.py`, `bucket_synthetic_data.py`, `inference/inference_chunked.py`
- Slurm: `fineweb.py`, `minhash_deduplication.py`, `tokenize_c4.py`, `process_common_crawl_dump.py`, `summary_stats.py`, `filter_hf_dataset.py`, `smol_data.py`
- Both: `inference/generate_data.py`, `finephrase*.py`, `progress_monitoring.py`

## See also

- [../../CLAUDE.md](../../CLAUDE.md) — `DataFolder` for `logging_dir`
- [../pipeline/CLAUDE.md](../pipeline/CLAUDE.md) — `PipelineStep` contract
- [../utils/CLAUDE.md](../utils/CLAUDE.md) — CLI tools, `launch_pickled_pipeline`
- [../../../examples/CLAUDE.md](../../../examples/CLAUDE.md) — executor choice matrix
