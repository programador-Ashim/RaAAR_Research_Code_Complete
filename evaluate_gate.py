"""Evaluate annotation-based evidence diagnostics and selective resolution.
Input JSONL fields: candidate,capability,target_version,linguistic_score,facts,
expected_evidence_status,gold_antecedent (optional),selected_antecedent (optional).
For genuine benchmark results, predictions MUST be generated without looking at gold status.
"""
import argparse,csv,json
from collections import Counter
from pathlib import Path
from raaar_evidence import Fact,gate

ROOT=Path(__file__).resolve().parent

def safe_div(a,b):return a/b if b else None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=ROOT/'evidence'/'annotation_template.jsonl')
    p.add_argument('--output',type=Path,default=ROOT/'results'/'gate_evaluation.json')
    p.add_argument('--min-score',type=float,default=.65)
    args=p.parse_args()
    if not args.input.exists():
        raise SystemExit('Missing annotation file: '+str(args.input))
    records=[json.loads(s) for s in args.input.read_text().splitlines() if s.strip()]
    evaluated=[]
    for case in records:
        if 'expected_evidence_status' not in case or not case['expected_evidence_status']:
            continue
        prediction=gate(case['candidate'],case['capability'],case['target_version'],
                        case['linguistic_score'],[Fact.from_dict(f) for f in case['facts']],
                        args.min_score)
        evaluated.append((case,prediction))
    if not evaluated:
        print('No gold-labeled examples to evaluate. Fill evidence/annotation_template.jsonl first.');return
    correct=sum(c['expected_evidence_status']==p['evidence']['status'] for c,p in evaluated)
    output={'n':len(evaluated),'evidence_status_accuracy':correct/len(evaluated),
            'gold_counts':dict(Counter(c['expected_evidence_status'] for c,_ in evaluated)),
            'predicted_counts':dict(Counter(p['evidence']['status'] for _,p in evaluated)),
            'coverage':sum(p['decision']=='resolve' for _,p in evaluated)/len(evaluated)}
    output['abstention_rate']=1-output['coverage']
    labels=['supported','missing','outdated','contradictory','refuted']
    output['per_condition']={}
    for name in labels:
        tp=sum(c['expected_evidence_status']==name and p['evidence']['status']==name for c,p in evaluated)
        fp=sum(c['expected_evidence_status']!=name and p['evidence']['status']==name for c,p in evaluated)
        fn=sum(c['expected_evidence_status']==name and p['evidence']['status']!=name for c,p in evaluated)
        output['per_condition'][name]={'precision':safe_div(tp,tp+fp),'recall':safe_div(tp,tp+fn),
                                        'tp':tp,'fp':fp,'fn':fn}
    has_gold=all('gold_antecedent' in c for c,_ in evaluated)
    if has_gold:
        resolved=[(c,p) for c,p in evaluated if p['decision']=='resolve']
        wrong=sum(p['predicted_antecedent']!=c['gold_antecedent'] for c,p in resolved)
        output['wrong_confident_resolution_rate']=safe_div(wrong,len(resolved))
        output['selective_accuracy']=safe_div(len(resolved)-wrong,len(resolved))
        output['risk_coverage_curve']=[]
        for cutoff in [round(i/20,2) for i in range(0,21)]:
            accepted=[(c,p) for c,p in evaluated if p['evidence']['status']=='supported'
                      and p['linguistic_score']>=cutoff]
            wrong_here=sum(p['predicted_antecedent']!=c['gold_antecedent'] for c,p in accepted)
            output['risk_coverage_curve'].append({'threshold':cutoff,
                      'coverage':len(accepted)/len(evaluated), 'risk':safe_div(wrong_here,len(accepted))})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))
    if all(c.get('synthetic',False) for c,_ in evaluated):
        print('CAUTION: Evaluated cases are synthetic demonstrations, NOT empirical paper results.')

if __name__=='__main__':main()
