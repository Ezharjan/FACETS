# FACETS-PR

FACETS-PR is a reproducible framework for evaluating AI-assisted police
reports. It takes synthetic body-worn-camera (BWC) transcripts, asks a set of
local language models to draft incident reports from them, and then scores each
draft along six axes:

* **F -- Source Fidelity** (F1 NEHR, F2 VQR, F3 SAA, F4 PCS)
* **A -- Asymmetric Exculpatory Recall** (A1 ER/IR, A2 Asymmetry Index)
* **C -- Cognitive Disengagement** (C1 RHDR, C2 LRS)
* **E -- Equity / Dialect Parity** (E1 DPS, E2 SDxD)
* **T -- Traceability** (T1 DTC, T2 SCS)
* **S -- Suspicion Inflation** (S1 IIR, S2 PCES, S3 EMD, S4 BSI)

The sixteen metrics are combined into a single composite, the **FACETS Risk
Score (FRS)**, where lower is better.

The whole pipeline runs locally. Drafts are produced through
[Ollama](https://ollama.com); nothing here calls a hosted or proprietary API.

The pipeline, end to end:

1. builds the synthetic **BWCSyn-90** corpus,
2. drives six open-weight LLMs through Ollama with a deterministic drafting
   prompt,
3. computes all sixteen metrics plus the composite FRS,
4. runs the hypothesis tests (H1-H6) described below, and
5. writes the summary report and the result figures.

---

## 1. Prerequisites

* **Ollama** installed with the daemon running. See <https://ollama.com>. The
  runner will pull the six default models on first use if they are missing.

* A **Python environment**. The project was developed against a conda
  environment named `ml`; activate whatever environment you use before
  installing or running anything:

  ```bash
  conda activate ml
  ```

* An **NVIDIA GPU** is recommended but not required. The spaCy transformer
  pipeline (`en_core_web_trf`), the DeBERTa MNLI scorer, and
  sentence-transformers all move to CUDA automatically when
  `torch.cuda.is_available()` is true, and each Ollama model uses the GPU when
  one is detected. Everything still runs on CPU, just slower.

## 2. Installation

```bash
conda activate ml

# (Optional, recommended) install a CUDA build of PyTorch first.
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Project dependencies
pip install -r requirements.txt

# spaCy transformer pipeline (used by NEHR, SAA, PCS, PCES)
python -m spacy download en_core_web_trf

# Confirm CUDA is visible (optional)
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

`sentence-transformers` and the DeBERTa MNLI checkpoint download their weights
lazily on first use.

If you would rather pull the models up front instead of letting the runner do
it (the runner pulls anything missing automatically):

```bash
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull qwen2.5:7b
ollama pull phi3:mini
ollama pull gemma2:9b
```

## 3. Running it end to end

Run everything from the `src/` folder.

```bash
conda activate ml

# 1. Build the corpus (90 transcript bundles; deterministic; seed = 0xCDAA48).
python generate_corpus.py --out data/bwcsyn90.json

# 1b. (Optional) sanity-check the AAE renderings with the DeBERTa NLI model.
python generate_corpus.py --out data/bwcsyn90.json --validate-aae

# 2. Run the full evaluation across all six models.
#    Make sure Ollama is running before this step.
python run_evaluation.py --corpus data/bwcsyn90.json \
    --out results/facets_pr_results.csv \
    --write-drafts-to results/drafts.jsonl

# 3. Hypothesis tests + figures.
python analyze_results.py --csv results/facets_pr_results.csv \
    --figs results/figures \
    --report results/report.txt
```

Step 2 is the slow one: generating across six models takes hours on a single
GPU. Steps 1 and 3 are quick.

### Resuming an interrupted run

If step 2 is interrupted (power loss, an Ollama crash, Ctrl-C), pass `--resume`
to pick up where it stopped instead of starting over:

```bash
python run_evaluation.py --resume --corpus data/bwcsyn90.json \
    --out results/facets_pr_results.csv \
    --write-drafts-to results/drafts.jsonl
```

Resume granularity is the *bundle*: a `(model, item)` pair counts as done only
when all three variant rows (`sae`, `aae`, `sae_rh`) are present in the CSV.
Rows belonging to incomplete bundles are dropped from the CSV automatically
before the run continues, so no duplicate rows are produced. To force a model
to be re-evaluated from scratch, delete its rows from the CSV first.

### Robustness notes

* Degenerate drafts (empty, whitespace-only, or no word characters -- small
  models occasionally emit these) are handled gracefully by every spaCy-based
  metric rather than crashing the transformer pipeline.
* Generation is capped at `num_predict = 2048` tokens with an explicit
  `num_ctx = 4096` context window. This bounds runaway repetitive generations
  and prevents silent prompt truncation.

### Smaller / faster runs

To run only the two smallest models (a config that fits comfortably on a 16 GB
laptop), restrict the model list:

```bash
python run_evaluation.py --models llama3.2:3b phi3:mini \
    --corpus data/bwcsyn90.json --out results/small.csv
```

Add `--do-recall` to additionally compute C2 (LRS). This roughly doubles the
wall-clock cost, because each transcript then needs a second model call under
the memoryless recall prompt.

### Re-running the analysis only

Once the CSV exists, `analyze_results.py` can be re-run on its own at any time
to regenerate the hypothesis tests, the per-model and per-axis summary tables,
and the figures:

```bash
python analyze_results.py --csv results/facets_pr_results.csv
```

## 4. The hypothesis tests

`analyze_results.py` runs six tests over the results CSV and writes them to the
report:

| Test | Question |
| ---- | -------- |
| H1 | Do models drop verbatim quotes? (VQR well below 0.5) |
| H2 | Do models recall inculpatory facts more than exculpatory ones? (Asymmetry Index > 0) |
| H3 | Do models harden epistemic stance and insert legal conclusions? (EMD > 0, IIR > 0) |
| H4 | Does the same model treat AAE transcripts differently from SAE? (DPS_EMD > 0) |
| H5 | Do reviewer models leave impossible "red herring" content in the report? (RHDR < 1) |
| H6 | Do models spontaneously add the AI-use disclosure tag? (DTC rate) |

The report also includes per-model and per-axis summary tables and, when
`statsmodels` is installed, linear mixed-effects fits for EMD and IIR.

## 5. Outputs

| Path | What it is |
| ---- | ---------- |
| `data/bwcsyn90.json` | The annotated BWCSyn-90 corpus. |
| `results/facets_pr_results.csv` | One row per (model, item, variant); every metric column. |
| `results/drafts.jsonl` | (optional) raw LLM drafts for spot-checking and the H5 audit. |
| `results/report.txt` | H1-H6 tests, per-model and per-axis summaries, mixed-effects fits. |
| `results/figures/fig_heatmap_metrics.pdf` | Per-axis metric heatmap across models. |
| `results/figures/fig_frs_bar.pdf` | Composite FACETS Risk Score with bootstrap CIs. |
| `results/figures/fig_asymmetry.pdf` | Boxplots of the Asymmetry Index per model (H2). |
| `results/figures/fig_emd_distribution.pdf` | Violin plots of EMD per model (H3). |
| `results/figures/fig_dps_heatmap.pdf` | Dialect Parity Score heatmap (Axis E). |
| `results/figures/fig_rhdr_by_model.pdf` | Red-Herring Detection Rate (H5). |
| `results/figures/fig_difficulty_strata.pdf` | FRS stratified by difficulty condition. |

Figures are saved as vector PDFs with a tight bounding box.

## 6. Source layout

```
src/
+-- facets_pr/                   # importable package
|   +-- __init__.py
|   +-- prompts.py               # drafting / review / recall prompts
|   +-- lexicons.py              # closed-class word lists used by Axis S / T
|   +-- corpus/
|   |   +-- templates.py         # six incident templates
|   |   +-- aae.py               # rule-based SAE -> AAE transducer
|   |   +-- red_herrings.py      # 20-sentence canary bank
|   |   `-- generator.py         # builds BWCSyn-90, dumps / loads JSON
|   +-- metrics/
|   |   +-- common.py            # spaCy / DeBERTa NLI / sentence-encoder loaders
|   |   +-- fidelity.py          # F1 NEHR, F2 VQR, F3 SAA, F4 PCS
|   |   +-- asymmetric.py        # A1 ER / IR, A2 Asymmetry Index
|   |   +-- cognitive.py         # C1 RHDR, C2 LRS
|   |   +-- equity.py            # E1 DPS, E2 SDxD
|   |   +-- traceability.py      # T1 DTC, T2 SCS
|   |   +-- suspicion.py         # S1 IIR, S2 PCES, S3 EMD, S4 BSI
|   |   `-- composite.py         # FACETS Risk Score
|   +-- pipeline.py              # end-to-end evaluation loop
|   +-- analysis.py              # H1-H6 tests, mixed-effects, summary tables
|   `-- viz.py                   # result figures
+-- generate_corpus.py           # CLI: build / validate BWCSyn-90
+-- run_evaluation.py            # CLI: run the evaluation to CSV
+-- analyze_results.py           # CLI: stats + figures
+-- requirements.txt
+-- LICENSE
`-- README.md
```

## 7. The BWCSyn-90 corpus

BWCSyn-90 is built from six incident templates (domestic violence, drug,
financial, DUI, property, traffic) crossed with five difficulty conditions
(`clean`, `chaotic`, `multi_speaker`, `code_switched`, `extended_interview`),
giving 30 base transcripts. Each base transcript is rendered in three variants:

* **sae** -- the base transcript in Standard American English,
* **aae** -- civilian (non-officer) speech rewritten by the rule-based AAE
  transducer, used to probe dialect parity,
* **sae_rh** -- the SAE transcript with one factually-impossible "red herring"
  turn spliced in, used to probe reviewer attention.

That is 30 x 3 = 90 transcript bundles. Every turn carries ground-truth tags
(evidentiary, inculpatory, exculpatory) so the metrics have something to score
against. Evidentiary turns are wrapped in `<Q>...</Q>` markers in the rendered
transcript so the drafting prompt can be asked to preserve them verbatim.

## 8. Reproducibility

Deterministic generation uses the seed `0xCDAA48` throughout: corpus
construction, red-herring sampling, and the Ollama generation options. Changing
the seed does not change any metric definition, but it does change which red
herring is paired with which transcript and the models' sampling trajectories,
so results will differ run to run if you change it.

## 9. Adapting to another jurisdiction or template

* Add a new incident template by editing `corpus/templates.py` and adding
  another `_xx()` builder to `_BASE_BUILDERS`.
* Adapt the legal-conclusion lexicon (`lexicons.LEGAL_TRIGGERS`) to your local
  charging vocabulary; the IIR metric picks up the change with no other edits.
* Add more models by passing `--models` to `run_evaluation.py`.

## 10. License

Released under the MIT License. See [LICENSE](LICENSE) for the full text.

Copyright (c) 2026 Alexander Ezharjan.
