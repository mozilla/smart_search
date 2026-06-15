# evaluation_pipeline_v2

Semantic / URL-bar history-search evaluation. These scripts drive a **real
Firefox ML build** through [`ml_driver`](https://github.com/Firefox-AI/ml_driver)
to measure **Recall@1**, **Precision**, and **displacement** of Firefox history
search as a function of the semantic distance threshold (and, in the as-you-type
test, query length). They source ground truth from the
[`Mozilla/history-search-retrieval`](https://huggingface.co/datasets/Mozilla/history-search-retrieval)
dataset and emit logs/CSVs + charts + a self-contained HTML report.

The actual querying is done by `ml_driver`, which loads synthetic history into a
Firefox profile, generates embeddings, and runs queries through
`PlacesSemanticHistoryManager` (semantic) and the URL-bar providers manager
(semantic + exact-match, combined by the muxer).

## Prerequisites

- **Python ≥ 3.12** — required by `ml_driver` (note: this is newer than the
  rest of `smart_search`, so use a dedicated environment).
- A **Firefox build with ML features enabled** — you pass its binary via
  `--firefox-bin` (e.g. `…/obj-*/dist/Nightly.app/Contents/MacOS/firefox`).
  A stock Firefox download will not work (no `PlacesSemanticHistoryManager`).
- A checkout of **`ml_driver`** (assumed as a sibling of this repo:
  `../ml_driver`; adjust paths below to your layout).
- **GitHub SSH access** — `ml_driver` depends on `ai-window-prompts-updater`,
  a git package pulled over SSH from `github.com/mozilla`.
- Network access to **HuggingFace** (the dataset is downloaded on first run and
  cached).

## Setup

`ml_driver` is `uv`-managed and pulls a git dependency, so the simplest reliable
path is to build its env with `uv sync`, then add the few extra packages these
scripts need.

```bash
# 1. Build ml_driver's environment (Python ≥3.12, installs ml_driver + git deps)
cd ../ml_driver
uv sync

# 2. Add the extra packages the evaluation scripts use
uv pip install huggingface-hub pyarrow matplotlib pandas
```

That creates `../ml_driver/.venv`. Activate it and run everything below
**from the `smart_search` repo root** (so `evaluation_pipeline_v2` is importable
as a package — the modules use package-relative imports and must be run with
`python -m …`, not as loose scripts):

```bash
cd ../smart_search                       # run from the smart_search repo root
source ../ml_driver/.venv/bin/activate   # the 3.12+ env with ml_driver installed
```

> Alternatively, make a dedicated env: `uv venv --python 3.12 && uv pip install -e
> ../ml_driver huggingface-hub pyarrow matplotlib pandas`. If your installer
> can't resolve `ai-window-prompts-updater`, install it explicitly:
> `uv pip install "ai-window-prompts-updater @ git+ssh://git@github.com/mozilla/remote-settings-ai-window-prompts-updater.git"`.

## Running the experiments

All commands are run from the `smart_search` repo root. Replace `$FF` with your
Firefox ML binary:

```bash
FF=/path/to/obj-*/dist/Nightly.app/Contents/MacOS/firefox
```

### 1. Semantic threshold sweep (PlacesSemanticHistoryManager)
Recall@1 / Precision vs distance threshold, per query/doc length bucket.

```bash
python -m evaluation_pipeline_v2.threshold_experiment \
    --firefox-bin "$FF" --max-profiles 5 --max-queries 40 --max-docs 150 \
    --thresholds 0.25 0.4 0.5 0.6 0.75 --out results/threshold.log
```

Sweep both embedding models sequentially (one shared vector DB; switching
`places.semanticHistory.embeddingType` drops/recreates the table). Writes
`threshold_static.log` and `threshold_contextual.log`:

```bash
python -m evaluation_pipeline_v2.threshold_experiment \
    --firefox-bin "$FF" --embedding-types static contextual \
    --max-profiles 5 --max-queries 40 --out results/threshold.log
```

### 2. URL-bar combined (semantic + exact-match history)
Runs the real address-bar pipeline (history providers only, combined by the
muxer).

```bash
# combined
python -m evaluation_pipeline_v2.urlbar_threshold_experiment \
    --firefox-bin "$FF" --max-profiles 5 --max-queries 40 --out results/urlbar.log

# exact-match-only baseline (semantic off — the current default behavior)
python -m evaluation_pipeline_v2.urlbar_threshold_experiment \
    --firefox-bin "$FF" --exact-only --out results/exact_only.log
```

### 3. Displacement / as-you-type
Eviction of exact matches by semantic results. Sweep the distance threshold, or
truncate the query to simulate typing (writes a CSV):

```bash
# threshold sweep
python -m evaluation_pipeline_v2.urlbar_displacement_experiment \
    --firefox-bin "$FF" --sweep threshold --out results/displacement_threshold.csv

# as-you-type (query truncated to N chars; fixed semantic threshold)
python -m evaluation_pipeline_v2.urlbar_displacement_experiment \
    --firefox-bin "$FF" --sweep truncation --truncations 5 8 12 14 16 10000 \
    --fixed-threshold 0.5 --visible-n 3 --out results/displacement_truncation.csv
```

Common flags: `--max-profiles`, `--max-queries`, `--max-docs` (relevant docs are
always kept), `--thresholds`, `--variant`, `--profiles <name…>`, `--no-headless`,
`--cache-dir`, `--log-level`. The driver sets the needed prefs internally and
uses `skip_fastly=False`, so **no `FASTLY_TOKEN`/FxA credentials are required**.

## Plotting & report

```bash
# recall or precision vs threshold (notebook-style line chart)
python -m evaluation_pipeline_v2.plot_threshold_results results/threshold.log --metric recall

# compare two runs (e.g. static vs contextual, or semantic vs urlbar)
python -m evaluation_pipeline_v2.plot_threshold_comparison \
    results/threshold_static.log results/threshold_contextual.log \
    --label-a "static (512-dim)" --label-b "contextual (384-dim)" --metric recall

# exact-match-only baseline bars
python -m evaluation_pipeline_v2.plot_threshold_bars results/exact_only.log

# displacement metrics from the CSV (recall|precision|eviction|rank_delta)
python -m evaluation_pipeline_v2.plot_displacement results/displacement_truncation.csv \
    --metric eviction --pooled --sweep-label "Query length (chars typed)"

# self-contained HTML report (embeds all PNGs + tables from a directory)
python -m evaluation_pipeline_v2.build_report --artifact-dir results --out results/report.html
```

## How the metrics are computed

- **Recall@1** — the top-1 result is a qrels-relevant doc. Clean primary metric
  (each query has exactly one labeled relevant doc).
- **Precision** — `relevant_retrieved / retrieved`, micro-averaged over the
  threshold-filtered result set. With one labeled relevant doc and `k≤2`, this is
  best read as an **over-retrieval / clutter** signal, not classic IR precision.
- **Eviction@N (displacement)** — of the exact/token-match results, the fraction
  pushed out of the combined top-N by the semantic results (which rank by
  frecency, not relevance).
