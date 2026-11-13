#!/usr/bin/env python3
"""Summarize real saved RQ1/RQ2 fold results; do not invent incomplete results."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def write_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def fmt(nums):
    vals=[float(x) for x in nums if x is not None]
    return f'{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}' if len(vals)>1 else (f'{vals[0]:.4f}' if vals else 'NA')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out_dir',type=Path,default=ROOT/'rq12_outputs')
    p.add_argument('--expected_folds',type=int,default=5)
    p.add_argument('--expected_seeds',nargs='+',type=int,default=[42,123,456])
    a=p.parse_args()
    rows=[]
    for f in sorted(a.out_dir.glob('*/**/metrics.json')):
        obj=json.loads(f.read_text(encoding='utf8'))
        for ds in ('damir_cv','reqeval_external'):
            met=obj.get(ds,{})
            r={'task':obj['task'],'model':obj['model'],'seed':int(obj['seed']),
               'fold':int(obj['fold']),'dataset':ds}
            for key in ('n','precision','recall','f1','f2','specificity','balanced_accuracy',
                        'accuracy','roc_auc','predicted_positive','predicted_negative',
                        'gold_positive','gold_negative','tp','fp','tn','fn',
                        'n_ranked_pronouns','top1_correct','top1_accuracy'):
                r[key]=met.get(key)
            rows.append(r)
    if not rows:
        raise SystemExit(f'No metrics.json found under {a.out_dir}. Run training first.')
    write_csv(a.out_dir/'rq12_all_runs.csv',rows)
    grouped=defaultdict(list)
    for r in rows:grouped[(r['task'],r['model'],r['dataset'])].append(r)
    summaries=[]
    print('\nRQ1/RQ2 RESULTS (mean ± SD over seeds, using within-seed fold means)')
    print('NOTES: Folds share data; ReqEval is SAME external set evaluated repeatedly. Repeated scores are correlated.')
    print('Never interpret repeated ReqEval results as independent test sets.\n')
    for (task,model,ds),data in sorted(grouped.items()):
        runs={(r['seed'],r['fold']) for r in data}
        missing=[(s,f) for s in a.expected_seeds for f in range(1,a.expected_folds+1) if (s,f) not in runs]
        scores=['precision','recall','f1','f2','specificity','balanced_accuracy','roc_auc'] if task=='detection' else ['top1_accuracy']
        seeds=sorted(set(r['seed'] for r in data))
        result={'task':task,'model':model,'dataset':ds,'completed_runs':len(data),
                'expected_runs':len(a.expected_seeds)*a.expected_folds,
                'complete':not missing,'missing_runs':str(missing)}
        for score in scores:
            seed_means=[]
            for seed in seeds:
                vals=[float(r[score]) for r in data if r['seed']==seed and r.get(score) is not None]
                if vals:seed_means.append(statistics.mean(vals))
            result[f'{score}_seed_mean_sd']=fmt(seed_means)
            result[f'{score}_runs_mean_sd_descriptive']=fmt([r[score] for r in data])
        summaries.append(result)
        print(f'{task.upper()} | {model} | {ds} | {len(data)}/{result["expected_runs"]} runs | '
              + ('COMPLETE' if not missing else 'INCOMPLETE'))
        for metric in scores:print(f'  {metric}: {result[f"{metric}_seed_mean_sd"]}')
    write_csv(a.out_dir/'rq12_summary.csv',summaries)
    # Keep metrics separate and mark incomplete; no bogus precision from candidate-level ranking.
    lines=['# RaAAR RQ1 / RQ2 evaluation summary','',
           'Run-level data: `rq12_all_runs.csv`. Summary: `rq12_summary.csv`.','',
           '**Do not publish incomplete study results as full matched comparisons.**','',
           'Statistics are descriptive mean ± sample SD of seed-wise fold means. '
           'For ReqEval, the same 109 records are evaluated in each run; tests are correlated.','',
           'Fold grouping uses the current *unverified* document-ID-prefix heuristic.','']
    for r in summaries:
        lines.append(f"## {r['task']} / {r['model']} / {r['dataset']}")
        lines.append(f"Completed {r['completed_runs']}/{r['expected_runs']} expected runs; complete={r['complete']}.")
        for key,value in r.items():
            if key.endswith('_seed_mean_sd'):lines.append(f'- {key.replace("_seed_mean_sd","")}: {value}')
        if not r['complete']:lines.append(f"- Missing: {r['missing_runs']}")
        lines.append('')
    (a.out_dir/'RQ12_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    print('\nFILES: '+str(a.out_dir/'rq12_all_runs.csv'))
    print('       '+str(a.out_dir/'rq12_summary.csv'))
    print('       '+str(a.out_dir/'RQ12_REPORT.md'))

if __name__=='__main__':main()
