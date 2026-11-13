#!/usr/bin/env python3
"""RaAAR controlled versioned-evidence study on real DAMIR/ReqEval pronoun records.

Only the requirement contexts and candidate lists come from DAMIR/ReqEval.
Versioned documents are researcher-constructed conditions, never original history.
Requires local raaar_evidence.py and evidence_pipeline.py for evaluation.
No model training or changes to existing output files.
"""
import argparse, csv, json, re
from collections import Counter
from pathlib import Path

CAPS = {
 'measure_temperature': ('measures temperature', 'measure temperature'),
 'send_data': ('sends data', 'send data'),
 'display_data': ('displays data', 'display data'),
 'activate_alarm': ('activates alarm', 'activate alarm'),
}
FIELDS = ['dataset','requirement_id','pronoun','candidate','candidate_options','context',
          'capability','domain_claim_basis','approved','reviewer_notes']

def read_records(project):
    out=[]
    for dataset,filename in [('DAMIR','damir_all.jsonl'),('ReqEval','reqeval_test.jsonl')]:
        path=project/'processed'/filename
        if not path.exists(): raise FileNotFoundError(f'Missing {path}; run prepare_data.py first.')
        with path.open(encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    obj=json.loads(line)
                    obj['dataset']=dataset
                    out.append(obj)
    return out

def init(project, worksheet, per_dataset):
    if worksheet.exists(): raise FileExistsError(f'{worksheet} already exists; cannot overwrite annotations.')
    rows=[]
    for ds in ('DAMIR','ReqEval'):
        eligible=[r for r in read_records(project) if r['dataset']==ds and r.get('candidates')]
        for r in eligible[:per_dataset]:
            options=[str(c.get('text','')).strip() for c in r['candidates'] if c.get('text')]
            rows.append({'dataset':ds,'requirement_id':str(r['id']),'pronoun':r.get('pronoun',''),
              'candidate':options[0] if options else '', 'candidate_options':' | '.join(options),
              'context':r.get('context',''),'capability':'','domain_claim_basis':'',
              'approved':'no','reviewer_notes':''})
    worksheet.parent.mkdir(parents=True,exist_ok=True)
    with worksheet.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
    print(f'Created {worksheet} with {len(rows)} actual pronoun records.')
    print('REVIEW EVERY ROW: choose a valid candidate, capability, document your basis, then set approved=yes.')
    print('Allowed capabilities:', ', '.join(CAPS))

def safe_slug(dataset, rid):
    return re.sub(r'[^A-Za-z0-9_-]+','_',f'{dataset}_{rid}')[:110]

def get_reviewed(worksheet, project):
    known={(r['dataset'],str(r['id'])):r for r in read_records(project)}
    with worksheet.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    accepted=[]
    for row in rows:
        key=(row['dataset'].strip(),row['requirement_id'].strip())
        if key not in known: raise ValueError(f'Not in original dataset: {key}')
        if row['approved'].strip().lower() != 'yes': continue
        candidate=row['candidate'].strip()
        options=[str(c.get('text','')).strip() for c in known[key].get('candidates',[])]
        if candidate not in options: raise ValueError(f'{key}: candidate not in original candidate options: {candidate!r}')
        if row['capability'].strip() not in CAPS: raise ValueError(f'{key}: unsupported capability {row["capability"]!r}; choose {list(CAPS)}')
        if len(row['domain_claim_basis'].strip())<15:
            raise ValueError(f'{key}: document the manual reasoning for why the capability is relevant (>=15 characters).')
        accepted.append(row)
    if not accepted: raise ValueError('No approved rows. Edit the worksheet: capability, domain_claim_basis, approved=yes.')
    return accepted

def generate(project, worksheet, study):
    approved=get_reviewed(worksheet,project)
    if study.exists(): raise FileExistsError(f'{study} exists. Keep previous study; choose another --study name.')
    study.mkdir(parents=True)
    index=[]
    for r in approved:
        ds, rid = r['dataset'].strip(),r['requirement_id'].strip()
        cand,cap=r['candidate'].strip(),r['capability'].strip()
        slug=safe_slug(ds,rid)
        for condition in ('supported','missing','outdated','contradictory'):
            folder=study/slug/condition
            folder.mkdir(parents=True)
            clause,neg=CAPS[cap]
            pos=f'The {cand} {clause}.\n'
            negative=f'The {cand} does not {neg}.\n'
            # This text does not assert that a real-world requirement historically had the indicated versions.
            if condition=='supported':
                docs=[('current.txt',pos,['2.0'],None)]
            elif condition=='missing':
                docs=[('current.txt','The project has not documented this component capability in the reviewed source collection.\n',['2.0'],None)]
            elif condition=='outdated':
                docs=[('historical.txt',pos,['1.0'],'current.txt'),
                      ('current.txt','The latest reviewed document contains no applicable supporting capability statement.\n',['2.0'],None)]
            else:
                docs=[('source_a.txt',pos,['2.0'],None),('source_b.txt',negative,['2.0'],None)]
            manifest={'requirement_keys':[f'{ds}:{rid}'],
                      'evidence_origin':'researcher_constructed_version_scenario',
                      'target_version':'2.0','condition_gold':condition,
                      'documents':[]}
            for number,(filename,text,valid,superseded) in enumerate(docs,1):
                (folder/filename).write_text(text,encoding='utf-8')
                manifest['documents'].append({'path':filename,'source_id':f'{slug}-{condition}-D{number}',
                        'valid_versions':valid,'authority':'authoritative','superseded_by':superseded})
            (folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            index.append({'dataset':ds,'requirement_id':rid,'pronoun':r['pronoun'],
                    'candidate':cand,'capability':cap,'target_version':'2.0',
                    'scenario_gold':condition,'evidence_origin':'researcher_constructed_version_scenario',
                    'claim_basis':r['domain_claim_basis'],
                    'manifest':str((folder/'manifest.json').resolve())})
    index_file=study/'scenarios.csv'
    with index_file.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(index[0]));w.writeheader();w.writerows(index)
    print(f'Generated {len(index)} scenarios for {len(approved)} reviewed DAMIR/ReqEval records.')
    print('Scenario index:',index_file)
    print('NOTE: Versions and evidence texts are constructed; original requirement wording remains unchanged.')

def evaluate(project,study):
    from evidence_pipeline import run
    index=study/'scenarios.csv'
    if not index.is_file(): raise FileNotFoundError(f'No scenario index: {index}; run --mode build first')
    output=[]
    with index.open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    for r in rows:
        # Score 0 prevents claiming a model-based accepted resolution; the evidence status is evaluated separately.
        result=run(Path(r['manifest']),r['candidate'],r['capability'],r['target_version'],0.0,.65)
        predicted=result['evidence']['status']
        output.append({**r,'status_predicted':predicted,
                       'evidence_check_correct':str(predicted==r['scenario_gold']).lower(),
                       'extracted_claims':len(result['extracted_claims']),
                       'documents_inspected':len(result['documents_inspected']),
                       'decision':'not_evaluated_no_trained_resolution_score'})
    dest=study/'evidence_results.csv'
    with dest.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    print('Study results:',dest)
    print('Status counts:',dict(Counter(r['status_predicted'] for r in output)))
    print('Scenario-label match:',sum(r['evidence_check_correct']=='true' for r in output),'/',len(output))
    print('This checks controlled evidence logic only; it does not evaluate real-world knowledge accuracy.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['init','build','evaluate'],required=True)
    p.add_argument('--project',type=Path,default=Path.cwd())
    p.add_argument('--per-dataset',type=int,default=5)
    p.add_argument('--worksheet',type=Path,default=Path('evidence/version_study_review.csv'))
    p.add_argument('--study',type=Path,default=Path('evidence/version_study_run1'))
    a=p.parse_args();project=a.project.resolve();worksheet=project/a.worksheet;study=project/a.study
    if a.mode=='init':init(project,worksheet,a.per_dataset)
    elif a.mode=='build':generate(project,worksheet,study)
    else:evaluate(project,study)
if __name__=='__main__':main()
