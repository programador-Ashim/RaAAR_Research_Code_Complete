#!/usr/bin/env python3
"""Small, isolated 10-pronoun RQ1/RQ2 smoke tests; NOT publication CV.
Place beside rq12_train.py. Doesn't modify original script or datasets.
"""
import argparse
import random
import sys
from pathlib import Path

import rq12_train as core

parser = argparse.ArgumentParser(description='Quick 10-pronoun smoke test')
parser.add_argument('--task', choices=['detection', 'resolution'], default='detection')
parser.add_argument('--model', choices=['roberta','spanbert','both'], default='roberta')
parser.add_argument('--n', type=int, default=10, help='Maximum pronoun instances per split')
parser.add_argument('--folds', type=int, default=5)
parser.add_argument('--max_folds', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
if args.n < 4:
    parser.error('--n must be at least 4')

# Select only cases relevant to the requested task. For ranking, retain ALL
# candidate rows for each sampled pronoun, so gold labels remain meaningful.
def sample_pronouns(examples, task, count, seed):
    rng = random.Random(seed)
    eligible = [ex for ex in examples if core.rows_for([ex], task)]
    if task == 'detection':
        pos = [ex for ex in eligible if ex['ambiguity_label'] == 1]
        neg = [ex for ex in eligible if ex['ambiguity_label'] == 0]
        rng.shuffle(pos)
        rng.shuffle(neg)
        selected = pos[:count//2] + neg[:count-count//2]
        selected_ids = {id(x) for x in selected}
        leftovers = [ex for ex in eligible if id(ex) not in selected_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[:max(0, count-len(selected))])
    else:
        rng.shuffle(eligible)
        selected = eligible[:count]
    return selected[:count]

real_split = core.split_data
real_perform = core.perform_run

def quick_split(damir, outer_ix, seed):
    tr, dev, te = real_split(damir, outer_ix, seed)
    return (sample_pronouns(tr, args.task, args.n, seed),
            sample_pronouns(dev, args.task, args.n, seed+1),
            sample_pronouns(te, args.task, args.n, seed+2))

def quick_perform(options, task, model_name, seed, fold, tr, dev, te, req):
    req_small = sample_pronouns(req, task, args.n, seed+3)
    print(f'QUICK TEST (NOT PUBLICATION): train={len(tr)} dev={len(dev)} '
          f'test={len(te)} reqeval={len(req_small)} pronouns', flush=True)
    return real_perform(options, task, model_name, seed, fold,
                        tr, dev, te, req_small)

core.split_data = quick_split
core.perform_run = quick_perform
sys.argv = ['rq12_train.py', '--task', args.task, '--model', args.model,
            '--epochs', str(args.epochs), '--folds', str(args.folds),
            '--max_folds', str(args.max_folds), '--seeds', str(args.seed),
            '--batch_size', '2', '--grad_accum', '1', '--max_length', '128',
            '--cpu', '--out_dir', str(Path('rq12_quick10_outputs'))]
core.main()
