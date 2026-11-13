# RaAAR RQ1 / RQ2 evaluation summary

Run-level data: `rq12_all_runs.csv`. Summary: `rq12_summary.csv`.

**Do not publish incomplete study results as full matched comparisons.**

Statistics are descriptive mean ± sample SD of seed-wise fold means. For ReqEval, the same 109 records are evaluated in each run; tests are correlated.

Fold grouping uses the current *unverified* document-ID-prefix heuristic.

## detection / roberta / damir_cv
Completed 11/15 expected runs; complete=False.
- precision: 0.4305 ± 0.2264
- recall: 0.4536 ± 0.2653
- f1: 0.3946 ± 0.1844
- f2: 0.4163 ± 0.2148
- specificity: 0.5299 ± 0.2359
- balanced_accuracy: 0.4918 ± 0.0240
- roc_auc: 0.5131 ± 0.0209
- Missing: [(456, 2), (456, 3), (456, 4), (456, 5)]

## detection / roberta / reqeval_external
Completed 11/15 expected runs; complete=False.
- precision: 0.4484 ± 0.2924
- recall: 0.3656 ± 0.2018
- f1: 0.3668 ± 0.2013
- f2: 0.3588 ± 0.1935
- specificity: 0.6695 ± 0.2151
- balanced_accuracy: 0.5175 ± 0.0255
- roc_auc: 0.5336 ± 0.0237
- Missing: [(456, 2), (456, 3), (456, 4), (456, 5)]

## detection / spanbert / damir_cv
Completed 10/15 expected runs; complete=False.
- precision: 0.4113 ± 0.0711
- recall: 0.6850 ± 0.4277
- f1: 0.4337 ± 0.2679
- f2: 0.5519 ± 0.3459
- specificity: 0.3187 ± 0.4110
- balanced_accuracy: 0.5018 ± 0.0084
- roc_auc: 0.5114 ± 0.0059
- Missing: [(456, 1), (456, 2), (456, 3), (456, 4), (456, 5)]

## detection / spanbert / reqeval_external
Completed 10/15 expected runs; complete=False.
- precision: 0.4979 ± 0.0919
- recall: 0.4919 ± 0.4129
- f1: 0.4182 ± 0.2702
- f2: 0.4535 ± 0.3457
- specificity: 0.5106 ± 0.4393
- balanced_accuracy: 0.5013 ± 0.0132
- roc_auc: 0.5418 ± 0.0012
- Missing: [(456, 1), (456, 2), (456, 3), (456, 4), (456, 5)]

## detection / tfidf_lr / damir_cv
Completed 15/15 expected runs; complete=True.
- precision: 0.4352 ± 0.1087
- recall: 0.5254 ± 0.1497
- f1: 0.4430 ± 0.1087
- f2: 0.4819 ± 0.1283
- specificity: 0.5157 ± 0.1229
- balanced_accuracy: 0.5205 ± 0.0181
- roc_auc: 0.5264 ± 0.0013

## detection / tfidf_lr / reqeval_external
Completed 15/15 expected runs; complete=True.
- precision: 0.5019 ± 0.1561
- recall: 0.3215 ± 0.1684
- f1: 0.3545 ± 0.1487
- f2: 0.3289 ± 0.1585
- specificity: 0.6993 ± 0.1445
- balanced_accuracy: 0.5104 ± 0.0207
- roc_auc: 0.5758 ± 0.0084
