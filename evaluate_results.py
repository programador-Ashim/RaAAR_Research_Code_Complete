"""Aggregate real fold/seed metrics. Never claims cross-validation when runs are missing."""
import argparse, csv, json, statistics
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cv-dir',type=Path,default=ROOT/'cv_outputs')
    p.add_argument('--output',type=Path,default=ROOT/'results'/'cv_summary.csv')
    args=p.parse_args()
    files=sorted(args.cv_dir.rglob('metrics.json'))
    if not files:
        print('No CV results found. Run experiment_cv.py first.');return
    grouped=defaultdict(list)
    for f in files:
        obj=json.loads(f.read_text())
        for dataset in ('damir_cv','reqeval_external'):
            if obj.get(dataset,{}).get('n',0):grouped[(obj['task'],obj['model'],dataset)].append(obj[dataset])
    fields=['task','model','dataset','n_runs','f1_mean','f1_sd','f2_mean','f2_sd',
            'accuracy_mean','always_positive_f2_mean','predicted_positive_mean',
            'top1_accuracy_mean','complete_5fold_3seed']
    output=[]
    for key, ms in sorted(grouped.items()):
        def mean_of(k):
            nums=[float(m[k]) for m in ms if m.get(k) is not None]
            return round(statistics.mean(nums),5) if nums else ''
        def sd_of(k):
            nums=[float(m[k]) for m in ms if m.get(k) is not None]
            return round(statistics.stdev(nums),5) if len(nums)>1 else ''
        task,model,dataset=key
        output.append(dict(task=task,model=model,dataset=dataset,n_runs=len(ms),
                f1_mean=mean_of('f1'),f1_sd=sd_of('f1'),f2_mean=mean_of('f2'),
                f2_sd=sd_of('f2'),accuracy_mean=mean_of('accuracy'),
                always_positive_f2_mean=mean_of('always_positive_f2'),
                predicted_positive_mean=mean_of('predicted_positive'),
                top1_accuracy_mean=mean_of('top1_accuracy'),
                complete_5fold_3seed='YES' if len(ms)>=15 else 'NO'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fields);writer.writeheader();writer.writerows(output)
    print('Saved',args.output)
    for x in output:print(x)
    print('IMPORTANT: ReqEval results across folds share the SAME 109 external examples; fold scores are not independent test sets.')

if __name__=='__main__':main()
