# writers/

## Purpose

Persist `Document` objects (or serialized dicts) to local disk, S3, or Hugging Face paths via fsspec-backed writers.

## Files

| Module | Class(es) | Role |
|--------|-----------|------|
| `disk_base.py` | `DiskWriter` | Base: filename templates, HF retry, adapters |
| `jsonl.py` | `JsonlWriter` | JSONL output |
| `parquet.py` | `ParquetWriter` | Parquet with batching, CDC, page index |
| `huggingface.py` | `HuggingFaceDatasetWriter`, `HuggingFaceBucketWriter` | Staged local write → Hub/bucket upload |

## Core concepts

**Filename templates.** `output_filename` is a `string.Template` with `${rank}`, `${id}`, and any metadata field. Rank is zero-padded to 5 digits. Inference checkpointing requires `${chunk_index}` in the template when used with `InferenceRunner`.

**`expand_metadata`.** When `True`, metadata keys become top-level columns in the serialized dict instead of a nested `metadata` object.

**`DiskWriter.write()`.** Adapts the document, optionally rotates files at `max_file_size`, writes via format-specific `_write()`.

**HF paths.** `hf://datasets/...` and `hf://buckets/...` open through fsspec; uploads happen on file close. `HuggingFaceDatasetWriter` / `HuggingFaceBucketWriter` write locally first, then push via Xet/batch APIs — better for very large datasets.

**Exclusion pattern.** Filters accept `exclusion_writer: DiskWriter` to persist rejected documents for audit.

## Complexities and pitfalls

- **Missing `${rank}`:** Logged warning; parallel tasks can overwrite each other's outputs.
- **HF Hub transient errors:** `DiskWriter` retries open/upload on 429, 412, rate limits, LFS verify failures (exponential backoff, up to 12 attempts).
- **Concurrent HF commits:** High Slurm array concurrency on `hf://datasets/...` can cause commit races; retries mitigate but job staggering helps.
- **`HuggingFaceDatasetWriter.cleanup`:** Deletes local staging files after upload when `True`.
- **Binary mode required for `max_file_size`:** Parquet/JSONL writers use `mode="wb"` or equivalent.

## Examples that touch this code

- `examples/fineweb.py`, dedup examples — `JsonlWriter`
- `examples/inference/generate_data.py` — `ParquetWriter` on `hf://datasets/...`
- `examples/filter_hf_dataset.py` — `HuggingFaceDatasetWriter`
- `examples/bucket_synthetic_data.py` — `ParquetWriter`, `HuggingFaceBucketWriter`
- `examples/inference/inference_chunked.py` — `JsonlWriter` with `${chunk_index}`

## See also

- [../CLAUDE.md](../CLAUDE.md) — pipeline block base
- [../../CLAUDE.md](../../CLAUDE.md) — `DataFolder`, path schemes
- [../filters/CLAUDE.md](../filters/CLAUDE.md) — `exclusion_writer`
- [../inference/CLAUDE.md](../inference/CLAUDE.md) — checkpoint + writer integration
