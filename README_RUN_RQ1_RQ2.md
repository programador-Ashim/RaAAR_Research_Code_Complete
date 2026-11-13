# RaAAR RQ1 + RQ2 — matched experiments

The scripts use **the same** `processed/damir_all.jsonl` and `processed/reqeval_test.jsonl` files you already have. Copy `rq12_train.py` and `rq12_report.py` into the **root of your existing** `RaAAR_Research_Code_Complete` folder. They do not replace `experiment_cv.py`, and all new outputs are under `rq12_outputs/`.

## Before running

1. Open Terminal and activate the **Python 3.10** environment, removing the old Python alias:

```bash
conda activate roberta-research
unalias python 2>/dev/null
cd ~/Desktop/Research/RaAAR_Research_Code_Complete
which python
python -c 'import torch, transformers, sklearn, numpy; print(torch.__version__, transformers.__version__)'
```

2. Verify input files:

```bash
ls -lh processed/damir_all.jsonl processed/reqeval_test.jsonl
```

3. Set the environment variables that were helpful on your Mac CPU:

```bash
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## A. Preflight — no training

```bash
python rq12_train.py --preflight --folds 5 --max_folds 5
```

Read `rq12_outputs/data_audit.json`. If DAMIR group IDs are actually not document IDs, revise the grouping rule **before treating CV as document-disjoint evidence**. The script currently uses `ID.split('-')[0]`, as your original data preparation does.

## B. Smoke tests — 1 fold, 1 seed, 1 epoch

RQ1: RoBERTa ambiguity detection:

```bash
python -u -X faulthandler rq12_train.py --task detection --model roberta \
  --epochs 1 --folds 5 --max_folds 1 --seeds 42 --batch_size 2 --grad_accum 4 --cpu
```

RQ2: RoBERTa antecedent ranking (separately trained):

```bash
python -u -X faulthandler rq12_train.py --task resolution --model roberta \
  --epochs 1 --folds 5 --max_folds 1 --seeds 42 --batch_size 2 --grad_accum 4 --cpu
```

RQ1 and RQ2 matched SpanBERT smoke tests:

```bash
python -u -X faulthandler rq12_train.py --task detection --model spanbert \
  --epochs 1 --folds 5 --max_folds 1 --seeds 42 --batch_size 2 --grad_accum 4 --cpu
python -u -X faulthandler rq12_train.py --task resolution --model spanbert \
  --epochs 1 --folds 5 --max_folds 1 --seeds 42 --batch_size 2 --grad_accum 4 --cpu
```

## C. Full matched experiments — CPU-intensive

**Do not start these until the smoke tests and grouping audit pass.** Your CPU will train 30 detection runs and 30 ranking runs: 2 backbones × 5 folds × 3 seeds per task. Seven epochs per run can be extremely time-consuming on a Mac CPU. Prefer a GPU machine when feasible.

**RQ1 full detection**:

```bash
python -u -X faulthandler rq12_train.py --task detection --model both \
  --epochs 7 --folds 5 --max_folds 5 --seeds 42 123 456 \
  --batch_size 2 --grad_accum 4 --cpu
```

**RQ2 full candidate ranking**:

```bash
python -u -X faulthandler rq12_train.py --task resolution --model both \
  --epochs 7 --folds 5 --max_folds 5 --seeds 42 123 456 \
  --batch_size 2 --grad_accum 4 --cpu
```

The script **skips existing results** if a matching `metrics.json` exists. If you run a 1-epoch pilot first in `rq12_outputs`, use a *new output directory* for final experiments so the 1-epoch metrics are not mistaken for full results. For example, append `--out_dir rq12_final_outputs` to both full commands and use the same directory in the report command below. Never run `--force` unless intentionally overwriting a result.

To preserve a reusable model checkpoint for your later RaAAR integration, add `--save_models` to the **specific training run** for which you want a checkpoint. Otherwise `predictions.csv`, `metrics.json`, and top-1 predictions are still saved.

## D. Produce RQ1 and RQ2 result tables

After full runs:

```bash
python rq12_report.py --out_dir rq12_final_outputs \
  --expected_folds 5 --expected_seeds 42 123 456
```

Open:

```bash
open rq12_final_outputs/RQ12_REPORT.md
open rq12_final_outputs/rq12_summary.csv
```

Each result folder has:

- `metrics.json` — detection precision, recall, F1, F2, specificity, balanced accuracy, ROC-AUC, counts, confusion matrix and always-positive baseline; or resolution pronoun-level top-1 accuracy.
- `predictions.csv` — raw candidate/example scores with gold labels.
- `damir_cv_top1.csv` and `reqeval_external_top1.csv` — **for resolution only**; selected versus gold antecedents per eligible pronoun.
- `data_audit.json` — fold/group counts and dataset sizes.

## Critical interpretation requirements

- The RQ1 decision threshold is chosen using **inner development data only**. The default balanced-accuracy objective discourages the original all-positive problem. It does not guarantee improvement; class counts and all-positive results still must be checked. F2 is always reported.
- Only *one* gold-correct candidate with 2+ labeled alternatives is included in RQ2. Report the number excluded. Top-1 accuracy is calculated per pronoun, not per candidate pair.
- ReqEval is held out as an **external test**, not used for training or threshold optimization.
- The current grouping uses an **unverified ID-prefix heuristic**. Audit it using original dataset metadata before a publication claim of document-independent validation.
- The report shows descriptive run variation; repeated ReqEval evaluations and folds do not create independent datasets. A final paper should also disclose exclusion counts, uncertainty estimates and limitations.
- Historical published SpanBERT_RE and MLensemble numbers cannot be treated as matched head-to-head results without reproducing their protocols.
- Exact-span and partial-span equivalence with the original paper are **not** implemented here; first complete candidate-level top-1, then add validated span-matching conventions.
- An actual zero-shot LLM baseline is separate and is **not** run by these scripts.

## E. Reproducible RQ1 baseline (recommended)

Besides the trivial always-positive baseline automatically included in every detection `metrics.json`, run a **class-balanced TF-IDF + logistic regression** baseline on exactly the same outer folds and seeds:

```bash
python rq1_baselines.py --out_dir rq12_final_outputs \
  --folds 5 --max_folds 5 --seeds 42 123 456
python rq12_report.py --out_dir rq12_final_outputs \
  --expected_folds 5 --expected_seeds 42 123 456
```

**This is NOT the published MLensemble.** If you need an exact MLensemble replication, obtain its original implementation, feature definitions and splits, and match its evaluation conditions; published scores alone are historical context, not directly comparable experimental results.
