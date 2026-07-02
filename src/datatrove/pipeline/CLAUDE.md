# pipeline/

## Purpose

All pipeline blocks inherit from `PipelineStep` in `base.py`. Blocks are composed into ordered lists and executed by an executor with rank/world_size sharding.

## Files

| Module | Role |
|--------|------|
| `base.py` | `PipelineStep` — stats, dependency checks, `run()` contract |

Subpackages with dedicated `CLAUDE.md`:

| Directory | Category |
|-----------|----------|
| `readers/` | Input |
| `writers/` | Output |
| `filters/` | Keep/drop |
| `dedup/` | Deduplication (+ decont) |
| `inference/` | LLM generation |

Code-only subpackages documented in this file:

| Directory | Category |
|-----------|----------|
| `extractors/` | HTML → text |
| `formatters/` | In-place text transform |
| `stats/` | Metrics collection |
| `tokens/` | Tokenization |
| `media/` | Binary/media |

## Core concepts

**`PipelineStep.__new__`.** Walks MRO for `_requires_dependencies` and calls `check_required_dependencies` before instantiation.

**Generator contract.** `run(data, rank, world_size)` yields `Document` objects (or passes through upstream generator). Filters omit dropped docs; writers consume and may re-yield.

**Stats on every block.** `self.stats = Stats(str(self))`; use `stat_update()`, `update_doc_stats()`, `track_time()` context manager.

**Typical single-job ordering:**

```text
readers → extractors → formatters → filters → (stats | tokens | dedup | inference) → writers
```

**Multi-job patterns (separate executor runs):**
- MinHash dedup: signature → buckets → cluster → filter
- Tokens: tokenize → merger
- Stats: collect → `StatsMerger`
- Decont: indexer → filter (see [dedup/CLAUDE.md](dedup/CLAUDE.md))
- Inference (Slurm): inference → monitor → datacard

### Extractors (`extractors/`)

Convert HTML to clean text via `BaseExtractor` + `ExtractorSandbox` (multiprocess timeout). `Trafilatura` is production default; `ReadabilityInscriptis` in `modular.py`. Replaces `doc.text` in place. Pair with `readers/warc.py`, not binary `media/` WARC readers. Used in `fineweb.py`, `process_common_crawl_dump.py`, dedup prep examples.

### Formatters (`formatters/`)

In-place transforms via `BaseFormatter.format(text)` — never drop documents. `FTFYFormatter`, `PIIFormatter`, `SymbolLinesFormatter`. Run after extractors, before filters; PII order vs quality filters matters. `fineweb.py` uses `PIIFormatter`.

### Stats (`stats/`)

Per-document metrics → per-rank JSON under `output_folder/{group}/{stat}/`, then `StatsMerger` for global `metric.json`. Key classes: `DocStats`, `LineStats`, `WordStats`, `SentenceStats`, `TokenStats`, `LangStats`, `PerplexityStats`, `ContaminationStats`. `TopKConfig` truncates histograms at merge time. **`tldextract` required** for fqdn URL stat groups. Separate Slurm job with `depends=` in `summary_stats.py`. Post-run: `merge_stats` CLI ([utils/CLAUDE.md](../utils/CLAUDE.md)).

### Tokens (`tokens/`)

Megatron-style binary training output. Two-stage: (1) `DocumentTokenizer` per rank → `.ds` + `.index` shards; (2) `DocumentTokenizerMerger` shuffle/merge. `TokensCounter` adds `metadata["token_count"]` without writing files. **Merger on S3 is slow** — use local FS; ~11 GB/cpu in examples. Validate output with `check_dataset` CLI. Examples: `tokenize_c4.py`, `tokenize_from_hf_to_s3.py`, `TokensCounter` in dedup/fineweb.

### Media (`media/`)

Binary handling on `Document.media[]` — parallel to text pipeline, no production examples (tests only).

| Subpackage | Key classes |
|------------|-------------|
| `media_readers/` | `BinaryReaderThreaded`, `ZstdReader`, `WarcReader` (raw bytes, not text WARC) |
| `media_writers/` | `BaseMediaWriter`, `ZstdWriter` |
| `filters/` | `MimeTypeFilter` on `media_bytes` |
| `readers/` | `HTTPFetchReader` |

Do not confuse `readers/warc.py` (HTML/text) with `media/media_readers/warc.py` (binary). `preserve_order=False` on threaded readers trades order for speed.

## Complexities and pitfalls

- **First block is usually a reader** with `data=None`; pre-built generators supported for tests.
- **Do not reuse `logging_dir`** across pipelines — stats and completion markers collide.
- **Extractor timeouts:** `ReadabilityInscriptis` defaults are very aggressive (0.1s).
- **Changing base class interfaces** requires project maintainer approval (AGENTS.md boundary).

## Child docs

- [readers/CLAUDE.md](readers/CLAUDE.md)
- [writers/CLAUDE.md](writers/CLAUDE.md)
- [filters/CLAUDE.md](filters/CLAUDE.md)
- [dedup/CLAUDE.md](dedup/CLAUDE.md)
- [inference/CLAUDE.md](inference/CLAUDE.md)

## See also

- [../../CLAUDE.md](../../CLAUDE.md) — `Document`, `io.py`, repo map
- [../executor/CLAUDE.md](../executor/CLAUDE.md) — how ranks invoke `run()`
- [../utils/CLAUDE.md](../utils/CLAUDE.md) — `PipelineStats`, CLI tools
- [AGENTS.md](../../../AGENTS.md) — adding new pipeline blocks
