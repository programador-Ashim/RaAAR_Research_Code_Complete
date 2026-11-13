#!/usr/bin/env python3
"""Seed 10 DAMIR/ReqEval rows with *synthetic pipeline-test* evidence.

This does not establish real-world facts about DAMIR/ReqEval. It is for
integration testing the existing restricted-pattern evidence checker.
No model training, score fabrication, or modification of model outputs.
"""
import argparse
import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

REQUIRED = {'dataset','requirement_id','pronoun','candidate','capability','target_version','manifest','evidence_origin','linguistic_score','candidate_options','context'}
CONDITIONS = ['supported','missing','outdated','contradictory','supported']
KEYWORDS = ('sensor','controller','device','module','server','system','station','unit','instrument')

def select(rows, dataset, n):
    choices=[]
    for r in rows:
        if r['dataset'] != dataset:
            continue
        if any(r.get(k,'').strip() for k in ['candidate','capability','target_version','manifest','evidence_origin']):
            continue
        candidates = [c.strip() for c in r['candidate_options'].split(' | ') if c.strip()]
        matches = [c for c in candidates if any(k in c.casefold() for k in KEYWORDS)
                   and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 '’-]*", c)
                   and 3 <= len(c) <= 65]
        if matches:
            choices.append((r, min(matches,key=len)))
        if len(choices) >= n:
            return choices
    raise RuntimeError(f'Only {len(choices)} eligible unannotated rows found for {dataset}; need {n}.')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', default='.', type=Path)
    ap.add_argument('--annotations', default='evidence/dataset_evidence_annotations.csv')
    ap.add_argument('--count-per-dataset',default=5,type=int)
    args=ap.parse_args()
    project=args.project.resolve()
    csvpath=(project/args.annotations).resolve()
    with csvpath.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        fieldnames=reader.fieldnames
        if not REQUIRED.issubset(fieldnames or []):
            raise ValueError('Wrong CSV schema: '+str(sorted(REQUIRED-set(fieldnames or []))))
        rows=list(reader)
    selected=select(rows,'DAMIR',args.count_per_dataset)+select(rows,'ReqEval',args.count_per_dataset)
    outputroot=project/'evidence'/'damir_reqeval_synthetic_demo'
    if outputroot.exists():
        raise FileExistsError(f'{outputroot} already exists; refusing to overwrite previous evidence. Rename it before retrying.')
    outputroot.mkdir(parents=True)
    for i,(r,candidate) in enumerate(selected):
        scenario=CONDITIONS[i%len(CONDITIONS)]
        # Version labels, source claims and predicates in this demo are explicitly invented.
        key=f"{r['dataset']}:{r['requirement_id']}"
        location=outputroot/(f"case_{i+1:02d}")
        location.mkdir()
        docs=[]
        def write_doc(filename,text,source_id,versions,superseded_by=None):
            (location/filename).write_text(text+'\n',encoding='utf-8')
            entry={'source_id':source_id,'path':filename,'valid_versions':versions,'authority':'authoritative'}
            if superseded_by: entry['superseded_by']=superseded_by
            docs.append(entry)
        # Synthetic data uses a generic predicate only to exercise the extraction rules.
        positive=f'{candidate} sends data.'
        negative=f'{candidate} does not send data.'
        if scenario=='supported':
            write_doc('current.txt',positive,f'SYN-{i+1:02d}-POS',['2.0'])
        elif scenario=='missing':
            write_doc('current.txt','This synthetic document lists maintenance intervals but has no statement about data transmission.',f'SYN-{i+1:02d}-GENERAL',['2.0'])
        elif scenario=='outdated':
            write_doc('historical.txt',positive,f'SYN-{i+1:02d}-OLD',['1.0'],'SYN-CURRENT')
            write_doc('current.txt','This synthetic updated specification lists maintenance intervals only.',f'SYN-{i+1:02d}-CURRENT',['2.0'])
        elif scenario=='contradictory':
            write_doc('current_support.txt',positive,f'SYN-{i+1:02d}-POS',['2.0'])
            write_doc('current_refute.txt',negative,f'SYN-{i+1:02d}-NEG',['2.0'])
        manifest={'requirement_keys':[key], 'evidence_origin':'synthetic',
                  'study_note':'Artificial data created to test pipeline behavior; not source truth for DAMIR/ReqEval',
                  'scenario_label_for_testing_only':scenario, 'documents':docs}
        (location/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        r['candidate']=candidate
        r['capability']='send_data'
        r['target_version']='2.0'
        r['manifest']=str((location/'manifest.json').relative_to(project))
        r['evidence_origin']='synthetic'
        r['linguistic_score']=''  # Do not fabricate any trained model score.
        print(key, '->',scenario, '| candidate:',candidate)
    backup=csvpath.with_suffix('.backup_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.csv')
    shutil.copy2(csvpath,backup)
    with csvpath.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print('Backed up original CSV:', backup)
    print('Wrote',len(selected),'synthetic test annotations to:', csvpath)
    print('Now run: python audit_datasets_evidence.py')
    print('No model predictions or trained checkpoints were modified.')

if __name__=='__main__':main()
