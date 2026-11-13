"""Create reproducible zero-shot prompt dataset; no invented LLM answers.
Use an external LLM separately and add model outputs for evaluation.
"""
import argparse,json
from pathlib import Path
def read_jsonl(path):
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=ROOT/'processed'/'reqeval_test.jsonl')
    p.add_argument('--output',type=Path,default=ROOT/'results'/'zero_shot_prompts.jsonl')
    a=p.parse_args()
    if not a.input.exists():raise SystemExit('Run prepare_data.py first')
    examples=read_jsonl(a.input)
    with a.output.open('w',encoding='utf8') as out:
        for x in examples:
            prompt=('Decide whether the TARGET pronoun is ambiguous in the requirement. '
                    'Reply with exactly AMBIGUOUS or UNAMBIGUOUS.\n'
                    f"Requirement: {x['context']}\nTarget pronoun: {x['pronoun']}")
            out.write(json.dumps({'id':x['id'],'task':'detection','prompt':prompt,
                                  'gold_label':x['ambiguity_label']},ensure_ascii=False)+'\n')
            cs=[c for c in x['candidates'] if c['label'] in ('correct','incorrect')]
            if len(cs)>1 and sum(c['label']=='correct' for c in cs)==1:
                prompt=('Select exactly one antecedent from the numbered options. '
                        'Reply only with the option number.\n'
                        f"Requirement: {x['context']}\nTarget pronoun: {x['pronoun']}\n"+
                        '\n'.join(f"{i+1}. {c['text']}" for i,c in enumerate(cs)))
                out.write(json.dumps({'id':x['id'],'task':'resolution','prompt':prompt,
                        'candidate_options':[c['text'] for c in cs],
                        'gold_option':next(i+1 for i,c in enumerate(cs) if c['label']=='correct')},
                        ensure_ascii=False)+'\n')
    print('Prompts:',a.output,'(LLM responses must be collected and evaluated separately.)')
if __name__=='__main__':main()
