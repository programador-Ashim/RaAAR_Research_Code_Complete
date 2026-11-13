#!/usr/bin/env python3
"""RaAAR RQ1/RQ2 matched grouped cross-validation.

Run from a folder containing processed/damir_all.jsonl and processed/reqeval_test.jsonl.
This does not modify existing checkpoints or results; output goes into rq12_outputs.
"""
import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
BACKBONES = {'roberta': 'roberta-base', 'spanbert': 'SpanBERT/spanbert-base-cased'}


def doc_key(ex):
    # Retain the current project's grouping policy so results are comparable.
    # THIS IS A HEURISTIC. Verify against source document IDs before publishing.
    return str(ex['id']).split('-')[0]


def read_jsonl(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def example_key(ex):
    return '|'.join([str(ex.get('source','')), str(ex['id']),str(ex.get('position','')),
                     str(ex['pronoun']), hashlib.sha256(ex['context'].encode()).hexdigest()[:16]])


def rows_for(examples, task):
    rows=[]
    for ex in examples:
        base={'group':example_key(ex), 'doc':doc_key(ex), 'id':str(ex['id']),
              'pronoun':str(ex['pronoun']), 'context':str(ex['context'])}
        if task=='detection':
            if ex.get('ambiguity_label') not in (0,1):
                continue
            rows.append({**base,'text':f"Context: {ex['context']}\nTarget pronoun: {ex['pronoun']}",
                         'label':int(ex['ambiguity_label']),'candidate':''})
        else:
            valid=[c for c in ex.get('candidates',[]) if c.get('label') in ('correct','incorrect')]
            if len(valid)<2 or sum(c['label']=='correct' for c in valid)!=1:
                continue
            for c in valid:
                rows.append({**base,'text':f"Context: {ex['context']}\nTarget pronoun: {ex['pronoun']}\nCandidate antecedent: {c['text']}",
                             'label':int(c['label']=='correct'),'candidate':str(c['text'])})
    return rows


def softmax_pos(logits):
    arr=np.asarray(logits,dtype=float)
    if arr.ndim!=2 or arr.shape[1]!=2: raise ValueError(f'Unexpected logit shape: {arr.shape}')
    arr=arr-arr.max(axis=1,keepdims=True)
    exp=np.exp(arr)
    return exp[:,1]/exp.sum(axis=1)


def classification_metrics(y, score, threshold):
    y=np.asarray(y,dtype=int); score=np.asarray(score,dtype=float)
    if len(y)!=len(score): raise ValueError('Inconsistent gold/score lengths')
    pred=(score>=threshold).astype(int)
    tp=int(((pred==1)&(y==1)).sum());tn=int(((pred==0)&(y==0)).sum())
    fp=int(((pred==1)&(y==0)).sum());fn=int(((pred==0)&(y==1)).sum())
    p,r,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    f2=5*p*r/(4*p+r) if 4*p+r else 0.
    spec=tn/(tn+fp) if tn+fp else None
    balanced=(r+spec)/2 if spec is not None else None
    try: auc=float(roc_auc_score(y,score))
    except ValueError: auc=None
    return dict(n=int(len(y)),gold_positive=int(y.sum()),gold_negative=int((1-y).sum()),
                predicted_positive=int(pred.sum()),predicted_negative=int((1-pred).sum()),
                tp=tp,fp=fp,tn=tn,fn=fn,precision=float(p),recall=float(r),f1=float(f1),
                f2=float(f2),specificity=spec,balanced_accuracy=balanced,
                accuracy=float(accuracy_score(y,pred)),roc_auc=auc,threshold=float(threshold))


def all_positive_baseline(y):
    return classification_metrics(y,np.ones(len(y)),0.5)


def select_threshold(y,score,objective):
    # DEV ONLY. F2-only tuning has produced all-positive decisions in the original pilot.
    # Balanced accuracy is the default selection objective; report F2 regardless.
    candidates=np.unique(np.r_[np.arange(0.05,.951,.025),0.5])
    candidates=np.round(candidates,6)
    def key(t):
        d=classification_metrics(y,score,float(t))
        if objective=='f2': return (d['f2'], d['balanced_accuracy'] or 0, -abs(t-.5))
        return (d['balanced_accuracy'] if d['balanced_accuracy'] is not None else -1,
                d['f2'], -abs(t-.5))
    return float(max(candidates,key=key))


def ranking_predictions(rows, scores):
    groups=defaultdict(list)
    for row,score in zip(rows,scores):groups[row['group']].append((row,float(score)))
    results=[]
    for key,items in sorted(groups.items()):
        top=max(items,key=lambda pair:pair[1])
        gold=[r['candidate'] for r,s in items if r['label']==1]
        if len(gold)!=1: continue
        results.append(dict(group=key,id=top[0]['id'],doc=top[0]['doc'],pronoun=top[0]['pronoun'],
                            n_candidates=len(items),predicted_antecedent=top[0]['candidate'],
                            gold_antecedent=gold[0],top_score=float(top[1]),
                            is_correct=int(top[0]['label']==1)))
    n=len(results)
    return dict(n_ranked_pronouns=n,top1_correct=sum(x['is_correct'] for x in results),
                top1_accuracy=(sum(x['is_correct'] for x in results)/n if n else None)),results


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def group_report(damir,req,outer,fold_count):
    out={'damir_records':len(damir),'reqeval_records':len(req),
         'damir_detection_labeled':len(rows_for(damir,'detection')),
         'reqeval_detection_labeled':len(rows_for(req,'detection')),
         'damir_resolution_eligible_pronouns':len(set(x['group'] for x in rows_for(damir,'resolution'))),
         'damir_detection_excluded':len(damir)-len(rows_for(damir,'detection')),
         'damir_resolution_excluded':len(damir)-len(set(x['group'] for x in rows_for(damir,'resolution'))),
         'reqeval_resolution_eligible_pronouns':len(set(x['group'] for x in rows_for(req,'resolution'))),
         'reqeval_resolution_excluded':len(req)-len(set(x['group'] for x in rows_for(req,'resolution'))),
         'document_grouping':'HEURISTIC: id prefix before hyphen; verify source documents',
         'distinct_group_keys':len(set(doc_key(x) for x in damir)), 'folds':[]}
    for fold,(train_ix,test_ix) in enumerate(outer,1):
        tr={doc_key(damir[i]) for i in train_ix};te={doc_key(damir[i]) for i in test_ix}
        assert not(tr&te),'Outer fold document leakage detected'
        out['folds'].append({'fold':fold,'train_pronouns_all':len(train_ix),
                             'heldout_pronouns_all':len(test_ix), 'train_groups':len(tr),
                             'test_groups':len(te), 'test_examples':sum(1 for i in test_ix if damir[i].get('ambiguity_label') in(0,1))})
    return out


def split_data(damir,outer_ix,seed):
    train_ix,test_ix=outer_ix
    pool=[damir[i] for i in train_ix];test=[damir[i] for i in test_ix]
    splitter=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed)
    ti,di=next(splitter.split(pool,groups=[doc_key(x) for x in pool]))
    train=[pool[i] for i in ti];dev=[pool[i] for i in di]
    groups=[{doc_key(x) for x in sub} for sub in (train,dev,test)]
    assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    return train,dev,test


def perform_run(args,task,model_name,seed,fold,train_ex,dev_ex,test_ex,req):
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,DataCollatorWithPadding,Trainer,TrainingArguments

    class EncodedDataset(torch.utils.data.Dataset):
        def __init__(self,rows,tok):
            self.labels=[r['label'] for r in rows]
            # Native Dataset; no HF datasets/dill fingerprinting.
            self.tokens=tok([r['text'] for r in rows],truncation=True,max_length=args.max_length,
                            padding=False,return_token_type_ids=False)
        def __len__(self):return len(self.labels)
        def __getitem__(self,i):
            data={key:values[i] for key,values in self.tokens.items()}
            data['labels']=self.labels[i]
            return data

    class WeightedTrainer(Trainer):
        def __init__(self,*a,class_weights=None,**kw):
            super().__init__(*a,**kw);self.class_weights=class_weights
        def compute_loss(self,model,inputs,return_outputs=False,**kwargs):
            inputs=dict(inputs);labels=inputs.pop('labels');outputs=model(**inputs)
            loss=torch.nn.functional.cross_entropy(outputs.logits,labels,
                       weight=self.class_weights.to(outputs.logits.device))
            return (loss,outputs) if return_outputs else loss

    dst=args.out_dir/task/model_name/f'seed_{seed}'/f'fold_{fold}'
    metrics_file=dst/'metrics.json'
    if metrics_file.exists() and not args.force:
        print(f'SKIP (already completed): {metrics_file}',flush=True)
        return
    dst.mkdir(parents=True,exist_ok=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    train,dev,test,ext=[rows_for(x,task) for x in (train_ex,dev_ex,test_ex,req)]
    if not train or not dev or not test or len(set(r['label'] for r in train))<2:
        raise ValueError(f'Insufficient examples for {task} seed {seed} fold {fold}')
    cnt=Counter(r['label'] for r in train)
    weights=torch.tensor([len(train)/(2*cnt[k]) for k in(0,1)],dtype=torch.float)
    model_id=BACKBONES[model_name]
    print(f'RUN task={task} backbone={model_id} seed={seed} fold={fold} '
          f'train_rows={len(train)} dev_rows={len(dev)} test_rows={len(test)} reqeval_rows={len(ext)}',flush=True)
    tokenizer=AutoTokenizer.from_pretrained(model_id,use_fast=False)
    model=AutoModelForSequenceClassification.from_pretrained(model_id,num_labels=2)
    arguments=TrainingArguments(output_dir=str(dst/'checkpoints'), num_train_epochs=args.epochs,
              learning_rate=args.learning_rate,weight_decay=.01,per_device_train_batch_size=args.batch_size,
              per_device_eval_batch_size=args.batch_size,gradient_accumulation_steps=args.grad_accum,
              evaluation_strategy='epoch',save_strategy='epoch',save_total_limit=1,
              load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,
              report_to='none',seed=seed,data_seed=seed,fp16=False,dataloader_num_workers=0,
              no_cuda=args.cpu,logging_steps=20,disable_tqdm=args.no_progress)
    trainer=WeightedTrainer(model=model,args=arguments,
             train_dataset=EncodedDataset(train,tokenizer),eval_dataset=EncodedDataset(dev,tokenizer),
             data_collator=DataCollatorWithPadding(tokenizer),class_weights=weights)
    trainer.train()
    def predict(rows):
        if not rows:return np.asarray([],dtype=float)
        return softmax_pos(trainer.predict(EncodedDataset(rows,tokenizer)).predictions)
    dev_score=predict(dev)
    threshold=select_threshold([r['label'] for r in dev],dev_score,args.threshold_objective) if task=='detection' else 0.5
    results={'task':task,'model':model_name,'seed':seed,'fold':fold,'epochs':args.epochs,
             'train_rows':len(train),'dev_rows':len(dev),'threshold':threshold,
             'threshold_objective':args.threshold_objective if task=='detection' else 'not_applicable',
             'class_weights':weights.tolist(), 'train_class_counts':dict(cnt),
             'training_checkpoint':str(trainer.state.best_model_checkpoint),
             'document_grouping':'id-prefix heuristic (not yet validated)',
             'dataset_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    (args.processed_dir/'damir_all.jsonl',args.processed_dir/'reqeval_test.jsonl')}}
    all_preds=[]
    for ds,rows in (('damir_cv',test),('reqeval_external',ext)):
        scores=predict(rows)
        if task=='detection':
            metrics=classification_metrics([r['label'] for r in rows],scores,threshold)
            metrics['always_positive_baseline']=all_positive_baseline(np.asarray([r['label'] for r in rows]))
        else:
            metrics,ranked=ranking_predictions(rows,scores)
            write_csv(dst/f'{ds}_top1.csv',ranked)
        results[ds]=metrics
        for r,s in zip(rows,scores):
            all_preds.append(dict(dataset=ds,group=r['group'],doc=r['doc'],id=r['id'],pronoun=r['pronoun'],
                       candidate=r['candidate'],gold=r['label'],score=float(s),
                       predicted_binary=int(s>=threshold),threshold=threshold))
    write_csv(dst/'predictions.csv',all_preds)
    if args.save_models:
        trainer.save_model(str(dst/'model'));tokenizer.save_pretrained(str(dst/'model'))
    metrics_file.write_text(json.dumps(results,indent=2),encoding='utf8')
    if not args.keep_checkpoints:shutil.rmtree(dst/'checkpoints',ignore_errors=True)
    print(f'SAVED {metrics_file}',flush=True)
    print(json.dumps({'task':task,'model':model_name,'seed':seed,'fold':fold,
                      'damir_cv':results['damir_cv'],'reqeval_external':results['reqeval_external']},indent=2),flush=True)
    del trainer,model


def main():
    p=argparse.ArgumentParser(description='RaAAR matched RQ1 and RQ2 experiments')
    p.add_argument('--model',choices=['roberta','spanbert','both'],default='roberta')
    p.add_argument('--task',choices=['detection','resolution','both'],default='detection')
    p.add_argument('--epochs',type=int,default=7)
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--max_folds',type=int,default=1,
                   help='Number of folds actually run (1=smoke; 5=full); ALWAYS specify')
    p.add_argument('--seeds',nargs='+',type=int,default=[42])
    p.add_argument('--batch_size',type=int,default=2)
    p.add_argument('--grad_accum',type=int,default=4)
    p.add_argument('--max_length',type=int,default=256)
    p.add_argument('--learning_rate',type=float,default=2e-5)
    p.add_argument('--threshold_objective',choices=['balanced_accuracy','f2'],default='balanced_accuracy')
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--save_models',action='store_true')
    p.add_argument('--keep_checkpoints',action='store_true')
    p.add_argument('--no_progress',action='store_true')
    p.add_argument('--force',action='store_true',help='Overwrite matching experiment results')
    p.add_argument('--preflight',action='store_true',help='Print data and fold report without loading models')
    p.add_argument('--processed_dir',type=Path,default=ROOT/'processed')
    p.add_argument('--out_dir',type=Path,default=ROOT/'rq12_outputs')
    args=p.parse_args()
    if args.folds<2 or not 1<=args.max_folds<=args.folds:raise ValueError('Require 2+ folds and 1<=max_folds<=folds')
    if args.epochs<1 or args.batch_size<1 or args.grad_accum<1:raise ValueError('Invalid training settings')
    damir=read_jsonl(args.processed_dir/'damir_all.jsonl')
    req=read_jsonl(args.processed_dir/'reqeval_test.jsonl')
    groups=[doc_key(x) for x in damir]
    if len(set(groups))<args.folds:raise ValueError('Too few distinct group IDs')
    outer=list(GroupKFold(n_splits=args.folds).split(damir,groups=groups))
    report=group_report(damir,req,outer,args.folds)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/'data_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2),flush=True)
    if args.preflight:return
    if args.max_folds<args.folds or len(args.seeds)<3 or args.epochs<5:
        print('WARNING: pilot / smoke test only; not complete RQ1 or RQ2 evaluation.',flush=True)
    tasks=['detection','resolution'] if args.task=='both' else [args.task]
    models=list(BACKBONES) if args.model=='both' else [args.model]
    for seed in args.seeds:
        for fold,outer_ix in enumerate(outer[:args.max_folds],1):
            tr,dev,te=split_data(damir,outer_ix,seed)
            for task in tasks:
                for name in models:perform_run(args,task,name,seed,fold,tr,dev,te,req)
    print('ALL REQUESTED EXPERIMENTS COMPLETED.',flush=True)

if __name__=='__main__':main()
