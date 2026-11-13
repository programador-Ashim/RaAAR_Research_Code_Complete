"""Matched RoBERTa/SpanBERT grouped CV for detection or antecedent ranking.
Torch Dataset plus slow tokenizer avoids datasets/dill and Rust tokenizer segfaults.
A smoke test is NOT a publishable 5-fold, multi-seed experiment.
"""
import argparse, csv, json, random, shutil
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, roc_auc_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from prepare_data import doc_key

ROOT = Path(__file__).resolve().parent
BACKBONES = {'roberta': 'roberta-base', 'spanbert': 'SpanBERT/spanbert-base-cased'}

def read_jsonl(path):
    with open(path, encoding='utf8') as stream: return [json.loads(s) for s in stream if s.strip()]

def record_id(x): return f"{x['id']}|{x.get('position')}|{x['pronoun']}|{x['context'][:32]}"

def make_rows(examples, task):
    result = []
    for x in examples:
        base = {'group': record_id(x), 'doc': doc_key(x)}
        if task == 'detection':
            if x['ambiguity_label'] is not None:
                result.append({**base, 'text': f"Context: {x['context']}\nTarget pronoun: {x['pronoun']}",
                               'label': int(x['ambiguity_label']), 'candidate': ''})
        else:
            valid = [c for c in x['candidates'] if c['label'] in ('correct', 'incorrect')]
            if len(valid) < 2 or sum(c['label'] == 'correct' for c in valid) != 1: continue
            for c in valid:
                result.append({**base, 'text': f"Context: {x['context']}\nTarget pronoun: {x['pronoun']}\nCandidate: {c['text']}",
                               'label': int(c['label'] == 'correct'), 'candidate': c['text']})
    return result

class EncodedDataset(torch.utils.data.Dataset):
    """No Arrow, HF Dataset.map or tokenizer pickling."""
    def __init__(self, rows, tokenizer, max_length):
        self.labels = [int(row['label']) for row in rows]
        self.inputs = tokenizer([row['text'] for row in rows], max_length=max_length,
                                truncation=True, padding=False, return_token_type_ids=False)
    def __len__(self): return len(self.labels)
    def __getitem__(self, index):
        item = {key: value[index] for key, value in self.inputs.items()}
        item['labels'] = self.labels[index]
        return item

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = dict(inputs)
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels,
                                                 weight=self.class_weights.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss

def scores_from_logits(logits):
    a = np.asarray(logits, dtype=np.float64)
    a = a - a.max(axis=1, keepdims=True)
    z = np.exp(a)
    return z[:, 1] / z.sum(axis=1)

def detection_metrics(gold, scores, threshold):
    y = np.asarray(gold, dtype=int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    f2 = 5*p*r/(4*p+r) if 4*p+r else 0.
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    result = {'n': len(y), 'gold_ambiguous_or_positive': int(y.sum()),
              'predicted_positive': int(pred.sum()), 'predicted_negative': int((1-pred).sum()),
              'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
              'precision': float(p), 'recall': float(r), 'f1': float(f1), 'f2': float(f2),
              'accuracy': float(accuracy_score(y, pred)), 'threshold': float(threshold)}
    result['always_positive_f2'] = float(detection_metrics_baseline(y))
    try: result['roc_auc_diagnostic_only'] = float(roc_auc_score(y, scores))
    except ValueError: result['roc_auc_diagnostic_only'] = None
    return result

def detection_metrics_baseline(y):
    p = float(np.mean(y)) if len(y) else 0
    return 5*p/(4*p+1) if 4*p+1 else 0.

def tune_threshold(gold, scores):
    # F2-only tuning can prefer all-positive; always report trivial baseline.
    return float(max(((detection_metrics(gold,scores,float(t))['f2'], -abs(t-.5), float(t))
                      for t in np.arange(.05, .951, .025)))[2])

def top1(rows, scores):
    groups = defaultdict(list)
    for row, score in zip(rows, scores): groups[row['group']].append((row, float(score)))
    eligible = [group for group in groups.values() if sum(r['label'] for r, _ in group) == 1]
    return {'n_ranked_pronouns': len(eligible),
            'top1_accuracy': (sum(max(g, key=lambda rs: rs[1])[0]['label'] for g in eligible)/len(eligible)
                              if eligible else None)}

def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with path.open('w', newline='', encoding='utf8') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def run(args, model_label, seed, fold, train_ex, dev_ex, test_ex, external_ex):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train, dev, test, external = (make_rows(xs, args.task) for xs in (train_ex, dev_ex, test_ex, external_ex))
    if not train or not dev or len(set(x['label'] for x in train)) < 2:
        raise RuntimeError('Invalid train/dev partition; review label counts and document-group mappings')
    destination = args.out_dir / args.task / model_label / f'seed_{seed}' / f'fold_{fold}'
    destination.mkdir(parents=True, exist_ok=True)
    model_id = BACKBONES[model_label]
    print('Loading slow tokenizer:', model_id, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    counts = Counter(x['label'] for x in train)
    weight = torch.tensor([len(train)/(2*counts[c]) for c in (0,1)], dtype=torch.float32)
    settings = TrainingArguments(output_dir=str(destination/'checkpoints'),
        num_train_epochs=args.epochs, learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, weight_decay=.01, evaluation_strategy='epoch',
        save_strategy='epoch', save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model='eval_loss', greater_is_better=False,
        logging_steps=20, seed=seed, data_seed=seed, report_to='none', fp16=False,
        dataloader_num_workers=0, no_cuda=args.cpu, disable_tqdm=args.no_progress)
    trainer = WeightedTrainer(model=model, args=settings,
        train_dataset=EncodedDataset(train,tokenizer,args.max_length),
        eval_dataset=EncodedDataset(dev,tokenizer,args.max_length),
        data_collator=DataCollatorWithPadding(tokenizer), class_weights=weight)
    trainer.train()
    def predict(rows):
        if not rows: return np.asarray([])
        return scores_from_logits(trainer.predict(EncodedDataset(rows,tokenizer,args.max_length)).predictions)
    dev_scores = predict(dev)
    threshold = tune_threshold([r['label'] for r in dev], dev_scores) if args.task=='detection' else .5
    outcome = {'model':model_label, 'task':args.task, 'seed':seed, 'fold':fold,
               'train_rows':len(train),'dev_rows':len(dev), 'threshold':threshold,
               'class_weight_0':float(weight[0]), 'class_weight_1':float(weight[1]),
               'best_checkpoint':trainer.state.best_model_checkpoint}
    prediction_rows=[]
    for name, rows in [('damir_cv',test),('reqeval_external',external)]:
        scores=predict(rows)
        value=detection_metrics([x['label'] for x in rows],scores,threshold) if rows else {'n':0}
        if args.task=='resolution': value.update(top1(rows,scores))
        outcome[name]=value
        for row,s in zip(rows,scores):
            prediction_rows.append({'model':model_label,'task':args.task,'seed':seed,'fold':fold,
                                    'dataset':name,'group':row['group'],'doc':row['doc'],
                                    'candidate':row['candidate'],'gold':row['label'],
                                    'prob_positive':float(s),'predicted':int(s>=threshold),
                                    'threshold':threshold})
    csv_write(destination/'predictions.csv',prediction_rows)
    if args.save_models:
        trainer.save_model(str(destination/'model')); tokenizer.save_pretrained(str(destination/'model'))
    (destination/'metrics.json').write_text(json.dumps(outcome,indent=2))
    if not args.keep_checkpoints: shutil.rmtree(destination/'checkpoints', ignore_errors=True)
    print(json.dumps({'model':model_label,'seed':seed,'fold':fold,
                      'damir_cv':outcome['damir_cv'],'reqeval_external':outcome['reqeval_external']},indent=2),flush=True)
    del trainer,model

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['roberta','spanbert','both'],default='roberta')
    p.add_argument('--task',choices=['detection','resolution'],default='detection')
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--max_folds',type=int,default=1)
    p.add_argument('--seeds',nargs='+',type=int,default=[42])
    p.add_argument('--batch_size',type=int,default=2)
    p.add_argument('--grad_accum',type=int,default=4)
    p.add_argument('--max_length',type=int,default=256)
    p.add_argument('--learning_rate',type=float,default=2e-5)
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--save_models',action='store_true')
    p.add_argument('--keep_checkpoints',action='store_true')
    p.add_argument('--no_progress',action='store_true')
    p.add_argument('--processed_dir',type=Path,default=ROOT/'processed')
    p.add_argument('--out_dir',type=Path,default=ROOT/'cv_outputs')
    args=p.parse_args()
    if args.max_folds is not None and args.max_folds < args.folds:
        print('WARNING: Smoke test only, not complete grouped cross-validation.',flush=True)
    damir=read_jsonl(args.processed_dir/'damir_all.jsonl')
    req=read_jsonl(args.processed_dir/'reqeval_test.jsonl')
    damir=[x for x in damir if make_rows([x],args.task)]
    grouped=[doc_key(x) for x in damir]
    if len(set(grouped))<args.folds: raise ValueError('Insufficient document groups')
    folds=list(GroupKFold(n_splits=args.folds).split(damir,groups=grouped))
    for seed in args.seeds:
        for fold,(train_ids,test_ids) in enumerate(folds[:args.max_folds],1):
            pool=[damir[i] for i in train_ids]; test=[damir[i] for i in test_ids]
            inner=GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
            ti,di=next(inner.split(pool,groups=[doc_key(x) for x in pool]))
            train=[pool[i] for i in ti]; dev=[pool[i] for i in di]
            for name in (['roberta','spanbert'] if args.model=='both' else [args.model]):
                run(args,name,seed,fold,train,dev,test,req)

if __name__=='__main__': main()
