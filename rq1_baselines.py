#!/usr/bin/env python3
"""Reproducible RQ1 baselines on the *same* fold assignments as rq12_train.py.
Not a reimplementation of published MLensemble. Always-positive and TF-IDF LR.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from rq12_train import (read_jsonl,rows_for,doc_key,split_data,classification_metrics,
                        all_positive_baseline,select_threshold,write_csv)

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--processed_dir',type=Path,default=ROOT/'processed')
    p.add_argument('--out_dir',type=Path,default=ROOT/'rq12_final_outputs')
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--max_folds',type=int,default=5)
    p.add_argument('--seeds',nargs='+',type=int,default=[42,123,456])
    p.add_argument('--force',action='store_true')
    args=p.parse_args()
    damir=read_jsonl(args.processed_dir/'damir_all.jsonl')
    req=read_jsonl(args.processed_dir/'reqeval_test.jsonl')
    outer=list(GroupKFold(n_splits=args.folds).split(damir,groups=[doc_key(x) for x in damir]))
    if not 1<=args.max_folds<=args.folds:raise ValueError('Invalid --max_folds')
    for seed in args.seeds:
        for fold,ix in enumerate(outer[:args.max_folds],1):
            dst=args.out_dir/'detection'/'tfidf_lr'/f'seed_{seed}'/f'fold_{fold}'
            f=dst/'metrics.json'
            if f.exists() and not args.force:
                print('SKIP',f);continue
            dst.mkdir(parents=True,exist_ok=True)
            tr,dev,te=split_data(damir,ix,seed)
            train,valid,test,external=[rows_for(x,'detection') for x in (tr,dev,te,req)]
            if len({r['label'] for r in train})<2:raise ValueError('Training fold has one class')
            model=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=40000)),
                ('lr',LogisticRegression(class_weight='balanced',random_state=seed,max_iter=2000))])
            model.fit([r['text'] for r in train],[r['label'] for r in train])
            ds=model.predict_proba([r['text'] for r in valid])[:,1]
            threshold=select_threshold([r['label'] for r in valid],ds,'balanced_accuracy')
            output={'task':'detection','model':'tfidf_lr','seed':seed,'fold':fold,
                    'threshold':threshold,'baseline_description':'TF-IDF unigram/bigram + class-balanced logistic regression; NOT MLensemble'}
            predrows=[]
            for name,rows in [('damir_cv',test),('reqeval_external',external)]:
                scores=model.predict_proba([r['text'] for r in rows])[:,1]
                m=classification_metrics([r['label'] for r in rows],scores,threshold)
                m['always_positive_baseline']=all_positive_baseline(np.asarray([r['label'] for r in rows]))
                output[name]=m
                for r,s in zip(rows,scores):
                    predrows.append(dict(dataset=name,group=r['group'],doc=r['doc'],id=r['id'],
                        pronoun=r['pronoun'],candidate='',gold=r['label'],score=float(s),
                        predicted_binary=int(s>=threshold),threshold=threshold))
            write_csv(dst/'predictions.csv',predrows)
            f.write_text(json.dumps(output,indent=2),encoding='utf8')
            print('SAVED',f)
    print('Baseline runs completed. Include them with rq12_report.py.')

if __name__=='__main__':main()
