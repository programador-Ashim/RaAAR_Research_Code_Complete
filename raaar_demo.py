"""Demonstrate supported/missing/outdated/contradictory RaAAR scenarios.
Linguistic scores are ILLUSTRATIVE unless --model-path is supplied.
"""
import argparse, json
from pathlib import Path
from raaar_evidence import gate, load_facts

ROOT=Path(__file__).resolve().parent

def rank_using_model(path, text, pronoun, candidates):
    import numpy as np
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    tokenizer=AutoTokenizer.from_pretrained(path,use_fast=False)
    model=AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    scores=[]
    for c in candidates:
        encoded=tokenizer(f'Context: {text}\nTarget pronoun: {pronoun}\nCandidate: {c}',
                          return_tensors='pt',max_length=256,truncation=True)
        with torch.no_grad():
            logits=model(**encoded).logits
        scores.append(float(torch.softmax(logits,dim=-1)[0,1]))
    winning=int(np.argmax(scores))
    return candidates[winning],scores[winning],dict(zip(candidates,scores))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scenario',choices=['all','supported','missing','outdated','contradictory','refuted'],default='all')
    p.add_argument('--model-path',type=Path,help='Saved RESOLUTION model checkpoint: cv_outputs/resolution/.../model')
    p.add_argument('--score',type=float,default=.82,help='Illustrative score if no model is provided')
    p.add_argument('--min-score',type=float,default=.65)
    p.add_argument('--output',type=Path,default=ROOT/'results'/'demo_results.json')
    a=p.parse_args()
    cases=json.loads((ROOT/'evidence'/'example_scenarios.json').read_text())
    results=[]
    for case in cases:
        if a.scenario!='all' and case['scenario']!=a.scenario:continue
        candidate=case['candidate'];score=a.score
        if a.model_path:
            candidate,score,ranking=rank_using_model(a.model_path,case['context'],case['pronoun'],case['candidates'])
        else: ranking={'note':'Illustrative score supplied by --score; no trained model used.'}
        outcome=gate(candidate,case['capability'],case['target_version'],score,
                     load_facts(ROOT/'evidence'/case['facts_file']),a.min_score)
        outcome.update({'case_id':case['id'],'scenario':case['scenario'],
                        'requirement':case['context'],'candidate_scores':ranking,
                        'expected_status':case['scenario'],
                        'synthetic':True})
        results.append(outcome)
        print(f"{case['id']}: expected={case['scenario']} observed={outcome['evidence']['status']} -> {outcome['decision']} ({outcome['reason']})")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(results,indent=2))
    print('Saved:',a.output)

if __name__=='__main__':main()
