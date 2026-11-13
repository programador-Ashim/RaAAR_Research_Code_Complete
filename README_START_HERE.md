# RaAAR: runnable research prototype (REFSQ paper companion)

**What this package does:** Actual DAMIR/ReqEval parsing; matched backbone experiments for ambiguity detection and binary candidate ranking; real prediction CSVs; grouped CV summaries; explicit synthetic examples for missing, outdated, contradictory and supported evidence; deterministic rule-based evidence gate; result evaluation and demo UI.

**What it does NOT prove:** This package does not reproduce the original ICSE 2022 MLensemble/SpanBERT_RE implementation, implement an automatic semantic evidence retriever, run an LLM API, generate a validated 60–80-case domain benchmark, or establish RaAAR improves accuracy. Those remain separate experimental obligations. The current rules require facts structured by a human; don't report their synthetic-demo classification as an empirical discovery.

## Mac Terminal: easiest way to run

Copy the contents of this ZIP to `~/Desktop/Research/RaAAR_Research_Code_Complete` (or another folder). The two XLSX files are already in `data/`.

```bash
conda activate roberta-research
unalias python 2>/dev/null
cd ~/Desktop/Research/RaAAR_Research_Code_Complete
which python
python -m pip install "sentencepiece>=0.1.99,<0.3"  # for slow RoBERTa/SpanBERT tokenizers
python -m unittest discover -s tests -v
python prepare_data.py
python raaar_demo.py
python evaluate_gate.py
python export_zero_shot.py
```

The evidence demo runs without Torch/Transformers and uses **illustrative** linguistic scores and **synthetic** facts. Its statuses should be supported, missing, outdated, contradictory, refuted. The annotation template is an EXAMPLE, not a real annotated benchmark.

### First one-fold training smoke test (Mac CPU)

```bash
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -u -X faulthandler experiment_cv.py \
  --model roberta --task detection --epochs 1 \
  --folds 5 --max_folds 1 --seeds 42 \
  --batch_size 2 --grad_accum 4 --cpu
python evaluate_results.py
```

This version uses `use_fast=False` **and a native `torch.utils.data.Dataset`**. Neither Hugging Face `Dataset.map()` nor Rust fast-tokenizer batch encoding is called. A segmentation fault elsewhere may still indicate a broken local native dependency.

### Full research runs: only after smoke tests succeed

```bash
python experiment_cv.py --model both --task detection --epochs 7 \
  --folds 5 --max_folds 5 --seeds 42 123 456 --batch_size 2 --grad_accum 4 --cpu
python experiment_cv.py --model both --task resolution --epochs 7 \
  --folds 5 --max_folds 5 --seeds 42 123 456 --batch_size 2 --grad_accum 4 --cpu
python evaluate_results.py
```

**CPU WARNING:** 60 seven-epoch model runs may require substantial compute; don't run this until ready. Add `--save_models` to save a loadable trained model for use in the demo. Without it, only metric/prediction files are saved. Training outputs are in `cv_outputs/<task>/<model>/seed_<N>/fold_<N>/`. Re-running the same identity can overwrite its files; back up `cv_outputs` as needed.

### Use a real trained resolution model (optional)

```bash
python raaar_demo.py --model-path cv_outputs/resolution/roberta/seed_42/fold_1/model
```

This loads actual candidate scores from that RESOLUTION model but still uses **synthetic structured domain evidence**. Model positive-class probabilities are scores, not automatically calibrated probabilities; the `.65` gate cutoff is illustrative. Genuine experiments must tune gates using development data only, never test gold labels.

### Zero-shot LLM

`python export_zero_shot.py` creates prompts and gold labels in `results/zero_shot_prompts.jsonl`. It **does not contact an LLM** or claim LLM evaluation results. Run a chosen LLM independently, save outputs and evaluate them using the same test cases and policy.

### Protocol caveats to fix before publishing

1. DAMIR labels are derived from source `ResolvedAs` annotation rules; compare those semantics with the published evaluation protocol.
2. `doc_key()` uses the substring of `id` before the first hyphen. This is an **unverified document grouping heuristic**; inspect real source-document IDs before claiming document-level leakage protection.
3. Some DAMIR examples have unknown detection labels and resolution uses only pronoun groups with exactly one correct candidate and at least two labeled candidates. Report filtering/exclusion counts.
4. F2-optimized dev thresholds can select an all-positive classifier. Compare always-positive baselines, predicted class counts, and AUC; do not call F2 improvement unless discrimination improves.
5. ReqEval is the same external 109-example evaluation in every fold; repeated ReqEval scores are correlated.
6. Exact/partial span accuracy requires explicit annotation conventions and is **not** implemented here; `top1_accuracy` is candidate-string selection.
7. Gate status accuracy on `annotation_template.jsonl` tests handcrafted rules on their matching synthetic fact patterns. It is a unit demonstration, NOT independent validation.
8. An actual REFSQ scientific-evaluation paper needs second annotation, κ, held-out evidence scenarios, case-level error analysis, and reliable baselines before reporting RaAAR effectiveness.

### Dashboard

```bash
python -m pip install streamlit pandas
python -m streamlit run dashboard.py
```

The dashboard shows **only real model metrics found in cv_outputs** and a clearly labeled synthetic evidence demonstration. It can run without training first.

### Independent annotation and error analysis

Enter independent labels in `evidence/independent_annotations.csv` then run:

```bash
python annotation_agreement.py
```

The CSV starts empty by design. Cohen's kappa is not available without human annotations. Use `evidence/error_analysis_template.csv` for manually checking at least 50 errors.

### Audit the evidence-data contract

Each knowledge-check case must have: `candidate` (the noun phrase), `capability` (normalized action), `target_version`, and fact records with `source_id`, `candidate`, `capability`, `polarity`, `valid_versions`, `authority`, `superseded_by`, and `excerpt`. A *current negative fact* is **refutation**, not missing evidence. A past authoritative fact is **outdated**, not current support. An unsupported fact or failed search cannot prove real-world nonexistence. Conflicts require incompatible CURRENT authoritative positive and negative facts for the same candidate/capability. This prototype assumes claims have already been normalized and does not perform automatic NLI.
