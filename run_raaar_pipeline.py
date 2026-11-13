#!/usr/bin/env python3
"""RaAAR prediction-to-evidence integration for DAMIR/ReqEval.

Uses *saved held-out predictions*, not made-up predictions or retraining.
For a specified model/seed/fold, detection and resolution predictions must both exist.
A detected unambiguous item is recorded but not passed as an ambiguous-item resolution.
Missing annotations are `not_assessed` (NOT 'missing').
If the selected ranked antecedent differs from the evidence annotation candidate,
no evidence is claimed for that selected candidate.

Example:
  python run_raaar_pipeline.py --model roberta --seed 42 --fold 1
"""
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path

from audit_datasets_evidence import load_records, read_annotations
from evidence_pipeline import run as check_evidence

FIELDS = ['dataset','requirement_id','pronoun','gold_ambiguity','detected_ambiguous',
          'detection_score','detection_threshold','ranked_candidate','ranked_candidate_score',
          'candidate_count','annotated_candidate','evidence_origin','evidence_status',
          'decision','reason','supporting_sources','refuting_sources','manifest','gold_resolution_correct']


def rows_from_csv(path, task):
    if not path.is_file():
        raise FileNotFoundError(f'Missing {task} predictions: {path}\n'
            f'Train {task} on this fold/seed with experiment_cv.py before integration. '
            'The existing detection-only run is insufficient for antecedent ranking.')
    with path.open(newline='',encoding='utf-8-sig') as fp:
        r=list(csv.DictReader(fp))
    r=[x for x in r if x.get('task') == task]
    if not r:
        raise ValueError(f'No {task} rows in {path}')
    return r


def join_key(row):
    # The original training pipeline uses record_id = ID|position|pronoun|context[:32]
    # The leading ID is stable for the processed data; dataset is crucial.
    dataset={'damir_cv':'DAMIR','reqeval_external':'ReqEval'}.get(row.get('dataset'))
    if not dataset: return None
    return dataset, row['group'].split('|',1)[0]


def run(project, model, seed, fold, threshold, allow_synthetic):
    project=project.resolve()
    root=project/'cv_outputs'
    dpath=root/'detection'/model/f'seed_{seed}'/f'fold_{fold}'/'predictions.csv'
    rpath=root/'resolution'/model/f'seed_{seed}'/f'fold_{fold}'/'predictions.csv'
    detection=rows_from_csv(dpath,'detection')
    resolution=rows_from_csv(rpath,'resolution')
    annotations=read_annotations(project/'evidence/dataset_evidence_annotations.csv')
    records=load_records(project)

    # Strict validation: do not combine mixed fold results or duplicate scores.
    dmap={}
    for r in detection:
        k=join_key(r)
        if k is None: continue
        if k in dmap: raise ValueError(f'Duplicate detection prediction: {k}')
        dmap[k]=r
    rmap=defaultdict(list)
    for r in resolution:
        k=join_key(r)
        if k is not None:rmap[k].append(r)
    seen=set(); output=[]
    for example in records:
        key=(example['dataset'],str(example['requirement_id']))
        if key in seen: raise ValueError(f'Duplicate requirement {key}')
        seen.add(key)
        det=dmap.get(key); ranking=rmap.get(key,[])
        annotation=annotations.get(key,{})
        record={k:'' for k in FIELDS}
        record.update(dataset=key[0],requirement_id=key[1],pronoun=example.get('pronoun',''),
                      gold_ambiguity=example.get('ambiguity_label'),
                      detected_ambiguous='',detection_score='',detection_threshold='',
                      ranked_candidate='',ranked_candidate_score='',candidate_count=len(ranking),
                      annotated_candidate=annotation.get('candidate',''),
                      evidence_origin=annotation.get('evidence_origin',''),
                      evidence_status='not_assessed',decision='human_review',reason='',
                      gold_resolution_correct='')
        if det is None:
            record['reason']='not_in_held_out_detection_predictions'
        else:
            record['detected_ambiguous']=det['predicted']
            record['detection_score']=det['prob_positive']
            record['detection_threshold']=det['threshold']
            if int(det['predicted'])==0:
                record['decision']='no_ambiguity_flagged'
                record['reason']='detection_predicted_unambiguous'
            elif not ranking:
                record['reason']='no_held_out_antecedent_ranking_predictions'
            else:
                # candidates should correspond to exactly one prediction each
                if len({s['candidate'].strip().casefold() for s in ranking})!=len(ranking):
                    raise ValueError(f'Duplicate ranking candidates for {key}')
                selected=max(ranking,key=lambda r:float(r['prob_positive']))
                candidate=selected['candidate'].strip()
                record['ranked_candidate']=candidate
                record['ranked_candidate_score']=selected['prob_positive']
                record['gold_resolution_correct']=selected.get('gold','')
                expected={str(c.get('text','')).strip().casefold() for c in example.get('candidates',[])}
                if candidate.casefold() not in expected:
                    record['reason']='ranked_candidate_not_in_dataset'
                elif not all((annotation.get(k) or '').strip() for k in ('candidate','capability','target_version','manifest','evidence_origin')):
                    record['reason']='no_complete_requirement_linked_evidence_annotation'
                elif candidate.casefold()!=annotation['candidate'].strip().casefold():
                    record['reason']='annotation_is_for_different_candidate'
                elif annotation['evidence_origin'].strip().lower() not in ('synthetic','authentic'):
                    record['reason']='invalid_evidence_origin'
                elif not allow_synthetic and annotation['evidence_origin'].strip().lower()=='synthetic':
                    record['reason']='synthetic_evidence_excluded_use_allow_synthetic'
                else:
                    path=Path(annotation['manifest'])
                    if not path.is_absolute():path=project/path
                    record['manifest']=str(path)
                    try:
                        manifest=json.loads(path.read_text(encoding='utf-8'))
                        if f'{key[0]}:{key[1]}' not in manifest.get('requirement_keys',[]):
                            raise ValueError('manifest_not_scoped_to_this_requirement')
                        if not manifest.get('documents'):raise ValueError('manifest_has_no_documents')
                        score=float(selected['prob_positive'])
                        result=check_evidence(path,candidate,annotation['capability'],
                                              str(annotation['target_version']),score,threshold)
                        evidence=result['evidence']
                        record['evidence_status']=evidence['status']
                        record['decision']=result['decision']
                        record['reason']=result['reason']
                        record['supporting_sources']='; '.join(x['source_id'] for x in evidence['supporting_sources'])
                        record['refuting_sources']='; '.join(x['source_id'] for x in evidence['refuting_sources'])
                    except (OSError,KeyError,TypeError,ValueError) as exc:
                        record['reason']='evidence_check_error: '+str(exc)
        output.append(record)
    dest=project/'results'/f'raaar_integrated_{model}_seed{seed}_fold{fold}.csv'
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        raise FileExistsError(f'{dest} already exists; rename it before rerunning')
    with dest.open('w',newline='',encoding='utf-8') as fp:
        wr=csv.DictWriter(fp,fieldnames=FIELDS);wr.writeheader();wr.writerows(output)
    summary={
      'model':model,'seed':seed,'fold':fold,'total_records':len(output),
      'with_detection_predictions':sum(x['detected_ambiguous']!='' for x in output),
      'with_ranked_antecedent':sum(bool(x['ranked_candidate']) for x in output),
      'evidence_status_counts':dict(Counter(x['evidence_status'] for x in output)),
      'decision_counts':dict(Counter(x['decision'] for x in output)),
      'reason_counts':dict(Counter(x['reason'] for x in output)),
      'warning':'One fold/seed; scores are not independently calibrated. '
                'Synthetic documents only test pipeline functionality, not empirical domain-knowledge prevalence. '
                'Do not evaluate antecedent correctness on samples lacking ranking predictions.'
    }
    outjson=dest.with_suffix('.summary.json')
    outjson.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print('CSV:',dest,'\nSummary:',outjson)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,default=Path.cwd())
    ap.add_argument('--model',choices=('roberta','spanbert'),default='roberta')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--fold',type=int,default=1)
    ap.add_argument('--min-score',type=float,default=.65)
    ap.add_argument('--allow-synthetic',action='store_true',help='Only for test demonstrations, not scientific evidence results')
    args=ap.parse_args()
    run(args.project,args.model,args.seed,args.fold,args.min_score,args.allow_synthetic)

if __name__=='__main__':main()
