# dedup/

## Purpose

Multi-stage and streaming deduplication: exact hash, MinHash LSH, sentence-span removal, exact substrings (suffix-array), Bloom-filter streaming, and eval contamination removal (decont).

## Files

| Module | Class(es) | Stage / role |
|--------|-----------|--------------|
| `exact_dedup.py` | `ExactDedupConfig`, `ExactDedupSignature`, `ExactFindDedups`, `ExactDedupFilter` | Exact hash dedup (+ URL variant) |
| `minhash.py` | `MinhashConfig`, `MinhashDedupSignature`, `MinhashDedupBuckets`, `MinhashDedupCluster`, `MinhashDedupFilter`, `MinhashBuildIndex` | 4-stage MinHash LSH |
| `sentence_dedup.py` | `SentDedupConfig`, `SentenceDedupSignature`, `SentenceFindDedups`, `SentenceDedupFilter` | C4-style sentence dedup |
| `exact_substrings.py` | `ESDatasetToSequence`, `ESMergeSequences`, `ESRangeRemover` | Exact substrings pipeline |
| `bloom_filter.py` | `BloomFilterConfig`, `SingleBloomFilter` | Single-pass streaming dedup |
| `decont/n_grams.py` | `NGramsDecontConfig`, `NGramsDecontIndexer`, `NGramsDecontFilter` | Eval n-gram contamination |

## Core concepts

**Multi-job pipelines.** Signature → find/bucket/cluster → filter are separate executor runs with binary intermediates on disk. Task counts and input readers must match between signature and filter stages.

**`only_dedup_in_index`.** Exact and sentence dedup can restrict matches to a pre-built index (dedup new data against an existing corpus).

**MinHash stages.** (1) signatures per doc, (2) bucket by LSH bands (`num_buckets` tasks), (3) union-find cluster (single high-memory task or Rust `tools/fast_mh3`), (4) filter using cluster map.

**Sentence dedup.** Removes duplicate *spans* inside documents, not whole-document duplicates; language-aware tokenization required.

**Exact substrings.** Stages 1–2 are DataTrove; stage 2.5 requires external [deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets) before stage 3.

**Bloom filter.** Single-pass; size and false-positive rate are tuning-critical.

### Decontamination (`decont/`)

Two-phase eval contamination removal via `NGramsDecontIndexer` then `NGramsDecontFilter` (extends `BaseFilter`). Builds n-gram hash indexes from LightEval tasks (`_requires_dependencies = ["lighteval"]`); filters training docs with overlapping n-grams. Indexing docs need `text=label`, `metadata={"query", "task"}`. Config flags `find_query_ngrams` / `find_overlap_ngrams` control what gets indexed. No `examples/` script — see `tests/pipeline/dedup/test_ngrams_decont.py`.

## Complexities and pitfalls

- **Never change `tasks` on partial relaunch** for signature/filter symmetry.
- **`HashConfig` must match** across all MinHash stages (seed, precision, hash function) — see [utils/CLAUDE.md](../../utils/CLAUDE.md).
- **Stage 2 task count = `num_buckets`** in MinHash; stage 3 is often one task with ~25 GB/cpu in production examples.
- **`shuffle_files=True` on readers** breaks dedup determinism.
- **Exact substrings:** `.bytearange` artifacts from external tool must exist before `ESRangeRemover`.

## Examples that touch this code

- `examples/minhash_deduplication.py`, `fineweb.py` — MinHash (4 stages)
- `examples/sentence_deduplication.py` — sentence dedup (local, 3 stages)
- `examples/url_deduplication.py` — exact URL dedup with custom normalizer
- `examples/exact_substrings.py` — exact substrings + external Rust step

## See also

- [../CLAUDE.md](../CLAUDE.md) — multi-job pipeline patterns
- [../../utils/CLAUDE.md](../../utils/CLAUDE.md) — hashing, CLI tools, `fast_mh3` Rust clustering
- [../readers/CLAUDE.md](../readers/CLAUDE.md) — file sharding for signature stage
- [../filters/CLAUDE.md](../filters/CLAUDE.md) — `NGramsDecontFilter` base
