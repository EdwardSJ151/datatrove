# readers/

## Purpose

Pipeline entry points: load external data into `Document` streams with rank/world_size sharding, optional adapters, and metadata defaults.

## Files

| Module | Class(es) | Role |
|--------|-----------|------|
| `base.py` | `BaseReader`, `BaseDiskReader` | Adapter contract, file listing, sharding |
| `jsonl.py` | `JsonlReader` | One JSON object per line |
| `parquet.py` | `ParquetReader` | PyArrow batch iteration |
| `csv.py` | `CsvReader` | CSV rows |
| `warc.py` | `WarcReader` | WARC HTTP payload → text (not binary media WARC) |
| `huggingface.py` | `HuggingFaceDatasetReader` | `datasets.load_dataset` / `load_from_disk` |
| `ipc.py` | `IpcReader` | Arrow IPC streams |

## Core concepts

**Adapter contract.** `BaseReader._default_adapter` maps a raw dict to `{text, id, media, metadata}`. Custom adapters receive `(self, data, path, id_in_file)`. The `text_key` and `id_key` constructor args select columns; if `text` is empty after adaptation, the row is skipped with a one-time warning.

**File-level sharding.** `BaseDiskReader.run()` calls `data_folder.get_shard(rank, world_size)` (or `get_shard_from_paths_file` when `paths_file` is set). Sharding is by **file**, not by line. One 100GB JSONL file on `tasks=10` means nine tasks do nothing.

**`paths_file`.** Explicit list of relative paths, one per line. Overrides glob/recursive listing; used for reproducible relaunches and custom file→task assignment.

**`HuggingFaceDatasetReader`.** `load_from_disk=True` uses `datasets.load_from_disk(path)` and ignores `dataset_options` / `streaming`. For Hub datasets, `dataset_options` must include `split` when the loaded object is a dict.

## Complexities and pitfalls

- **`tasks` vs file count:** Setting `tasks` greater than the number of input files wastes workers. Never change `tasks` mid-relaunch on partially completed jobs.
- **`shuffle_files=True`:** Debug/viz only. Breaks deterministic dedup signatures.
- **`limit`:** Applied per rank independently in HF reader; in disk readers, applied within each task's file shard.
- **Wrong `text_key`:** Silent skip of all documents after first warning.
- **WARC reader:** Heavy optional deps (`warcio`, `cchardet`, `python-magic`). Distinct from `pipeline/media/media_readers/warc.py` (binary payloads).

## Examples that touch this code

- `examples/fineweb.py`, `process_common_crawl_dump.py` — `WarcReader`
- `examples/minhash_deduplication.py`, `sentence_deduplication.py`, `url_deduplication.py`, `exact_substrings.py` — `JsonlReader`
- `examples/tokenize_c4.py`, `tokenize_from_hf_to_s3.py` — `JsonlReader` / `HuggingFaceDatasetReader`
- `examples/summary_stats.py` — `JsonlReader`
- `examples/filter_hf_dataset.py`, `bucket_synthetic_data.py` — `ParquetReader`
- `examples/inference/generate_data.py` — `HuggingFaceDatasetReader`

## See also

- [../CLAUDE.md](../CLAUDE.md) — `PipelineStep` base
- [../../CLAUDE.md](../../CLAUDE.md) — `DataFolder`, `get_shard`
- [../../executor/CLAUDE.md](../../executor/CLAUDE.md) — rank/world_size assignment
- [AGENTS.md](../../../../AGENTS.md) — adding a new reader
