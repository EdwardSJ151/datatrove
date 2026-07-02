# utils/ (+ tools/)

## Purpose

Shared library helpers for pipeline blocks and executors, plus command-line utilities for post-run inspection, stats merge, and Slurm worker entry.

## Library modules (`utils/`)

| Module | Role |
|--------|------|
| `logging.py` | Loguru setup, per-task log files, `DATATROVE_COLORIZE_LOGS` |
| `stats.py` | `Stats`, `PipelineStats`, `MetricStats`, `TimingStats` |
| `_import_utils.py` | Optional dep checks, `ASSETS_PATH`, availability helpers |
| `batching.py` | `batched()` shim for Python < 3.12 |
| `binaryio.py` | Struct parsing for dedup binary files |
| `text.py` | Normalization, n-grams, sentence/word splitting |
| `word_tokenizers.py` | Per-language tokenizers from assets CSV |
| `japanese_tokenizer.py` | Japanese tokenizer (lazy import) |
| `tokenization.py` | `load_tokenizer`, `PipelineStepWithTokenizer`, `chunk_doc_ends` |
| `hashing.py` | `HashConfig`, `create_hash_func()` |
| `hashes/sha1.py`, `hashes/xxhash.py` | Hash backends (use via `create_hash_func()`) |
| `typeshelper.py` | `Languages` enum, `StatHints`, extension helpers |
| `lid.py` | `FastTextLID`, `FT176LID`, `GlotLID` — cached model downloads |
| `perplexity.py` | `KenlmModel` for perplexity stats |
| `dataset.py` | `DatatroveFileDataset` — PyTorch Dataset over `.ds` files |
| `media.py` | `iter_pages()` — split text by media page offsets |
| `jobs.py` | Slurm job helpers — **not imported elsewhere in repo** |

### Bundled assets (`assets/`)

Static files at `ASSETS_PATH` (via `importlib.resources`): `tokenizer_assignment.csv`, URL filter word lists (`banned_words.txt`, etc.). Not accessed via `DataFolder`. `URLFilter` tarball may be absent in some checkouts.

## CLI tools (`tools/`)

Registered in `pyproject.toml` as console scripts:

| Script | Entry point | Role |
|--------|-------------|------|
| `check_dataset.py` | `check_dataset` | Validate `.ds` / `.index` / `.ds.loss` |
| `merge_stats.py` | `merge_stats` | Merge per-rank `stats/*.json` |
| `inspect_data.py` | `inspect_data` | Rich TUI for jsonl/parquet/csv/warc |
| `failed_logs.py` | `failed_logs` | Logs for incomplete ranks |
| `jobs_status.py` | `jobs_status` | Completion scan |
| `track_jobs.py` | `track_jobs` | Live progress |
| `launch_pickled_pipeline.py` | `launch_pickled_pipeline` | Slurm worker entry (`dill`) |

**Monitoring tools require `executor.json`** in the logging dir. **`merge_stats`** run manually after large multi-task jobs. **`launch_pickled_pipeline`** invoked by `SlurmPipelineExecutor`, not users directly. **`rich` CLI extra** needed for inspect/failed_logs/jobs_status/track_jobs.

### Rust MinHash clustering (`tools/fast_mh3/`)

Standalone binary for MinHash dedup stage 3 (union-find). Build with `cargo build --release` in `fast_mh3/`. Not imported from Python; must match stage 2 bucket layout (see `fast_mh3/README.md`).

## Core concepts

**`PipelineStats`.** Merges per-rank `Stats` JSON after each task completes.

**`HashConfig`.** Must match across all dedup stages — switching hash function/precision invalidates prior signatures.

**`cached_asset_path_or_download`.** Hub cache + `safely_create_file` for LID/perplexity models.

## Complexities and pitfalls

- **`Languages` enum is huge** — use attribute access.
- **`xxhash` is optional** — runtime check on `HashConfig(hash_fc="xxhash")`.
- **`load_tokenizer_assignments()` is `@lru_cache`d** — tests must `.cache_clear()` when mocking.
- **`DatatroveFileDataset` random access is slow** — sequential reads after shuffle-chunk tokenization.
- **`check_dataset` assumes Megatron-style `.ds` layout** — wrong tokenizer settings cause false failures.
- **Pickled pipelines** — import inside rollout/function body if Slurm unpickling fails.
- **CLI tools use `logger.remove()`** for clean Rich output.

## Examples that touch this code

- Indirect via pipeline blocks in most `examples/`
- `examples/minhash_deduplication.py`, `fineweb.py` — `HashConfig`
- `examples/summary_stats.py` — stats + `merge_stats` workflow
- `examples/tokenize_*.py` — `check_dataset` on output

## See also

- [../../CLAUDE.md](../../CLAUDE.md) — `Document`, `io.py`
- [../executor/CLAUDE.md](../executor/CLAUDE.md) — completion files, pickling
- [../pipeline/dedup/CLAUDE.md](../pipeline/dedup/CLAUDE.md) — hash-dependent dedup
- [../pipeline/CLAUDE.md](../pipeline/CLAUDE.md) — stats/tokens blocks
