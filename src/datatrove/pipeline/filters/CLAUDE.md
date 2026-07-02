# filters/

## Purpose

Document rejection blocks: return keep/drop decisions on each `Document`, with optional side output of rejected docs and batched inference hooks.

## Files

| Module | Class(es) | Basis |
|--------|-----------|-------|
| `base_filter.py` | `BaseFilter` | Abstract `filter(doc) -> bool` |
| `gopher_quality_filter.py` | `GopherQualityFilter` | Gopher quality heuristics |
| `gopher_repetition_filter.py` | `GopherRepetitionFilter` | Gopher repetition rules |
| `c4_filters.py` | `C4QualityFilter`, `C4ParagraphFilter`, `C4BadWordsFilter` | C4 paper filters |
| `fineweb_quality_filter.py` | `FineWebQualityFilter` | FineWeb-specific rules |
| `language_filter.py` | `LanguageFilter` | LID-based language gate |
| `url_filter.py` | `URLFilter` | URL blocklists / patterns |
| `regex_filter.py` | `RegexFilter` | Custom regex |
| `sampler_filter.py` | `SamplerFilter` | Random subsampling |
| `lambda_filter.py` | `LambdaFilter` | User predicate |
| `fasttext_filter.py` | `FastTextClassifierFilter` | FastText model |
| `unigram_log_probs.py` | `UnigramLogProbFilter` | Perplexity/unigram threshold |

## Core concepts

**`BaseFilter.run()`.** Calls `filter(doc)`; on `False`, optionally writes to `exclusion_writer` and updates stats. Supports `batch_filter(batch)` for vectorized paths; if `batch_size > 1` without override, logs warning and falls back to per-doc `filter()`.

**Tuple rejection.** `filter()` may return `(False, "reason")`. Reason becomes stat label `dropped_{reason}` and can land in excluded doc metadata.

**`LanguageFilter` + exclusion_writer.** Can route rejected docs to language-specific output paths via filename templating.

**CPU cost.** Repetition analysis, LID, and FastText filters are expensive at scale; tune `tasks` and partition accordingly.

## Complexities and pitfalls

- **Order in pipeline:** Filters run after readers/extractors/formatters in typical CC flows; PII formatting before vs after quality filters changes outcomes.
- **`SamplerFilter` rate:** Used in `summary_stats.py` and `smol_data.py` for cost control; rate must be tuned to target token budgets.
- **`URLFilter` assets:** Loads bundled word lists from `assets/`; optional tarball may be absent in some checkouts.
- **No `batch_filter` override:** Setting `batch_size > 1` without implementing `batch_filter` does not batch.

## Examples that touch this code

- `examples/fineweb.py`, `process_common_crawl_dump.py` — Gopher, C4, FineWeb, Language, URL
- `examples/sentence_deduplication.py`, `exact_substrings.py` — Gopher + Language
- `examples/summary_stats.py`, `smol_data.py` — `SamplerFilter`
- `examples/bucket_synthetic_data.py`, `filter_hf_dataset.py` — `LambdaFilter`

## See also

- [../CLAUDE.md](../CLAUDE.md) — `PipelineStep` stats; extractors and formatters
- [../writers/CLAUDE.md](../writers/CLAUDE.md) — exclusion writers
- [../../utils/CLAUDE.md](../../utils/CLAUDE.md) — URL filter word lists (bundled assets)
