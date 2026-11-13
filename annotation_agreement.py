"""Cohen's kappa for independently annotated evidence labels (requires genuine annotations)."""
import argparse,csv
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=Path('evidence')/'independent_annotations.csv')
    a=p.parse_args()
    with a.input.open(newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
    pairs=[(r['annotator_1'],r['annotator_2']) for r in rows
           if r.get('annotator_1') and r.get('annotator_2')]
    if not pairs:raise SystemExit('No double-annotated rows. Fill annotator_1 and annotator_2 independently.')
    print('Double-annotated cases:',len(pairs))
    print('Cohen kappa:',cohen_kappa_score([x for x,_ in pairs],[y for _,y in pairs]))
    print('Agreement:',sum(a==b for a,b in pairs)/len(pairs))
if __name__=='__main__':main()
