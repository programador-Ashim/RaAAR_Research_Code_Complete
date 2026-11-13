"""RaAAR document-to-evidence pipeline (restricted-language research prototype).

Reads plain-text specification documents declared in a JSON manifest, extracts
*only* simple component/capability statements, and audits versioned evidence.
Facts carry source IDs and excerpts. Unrecognized claims remain unverified.
No ML, web retrieval, or dependence on a running model-training process.
"""
from __future__ import annotations
import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from raaar_evidence import Fact, gate

# A transparent, extensible dictionary of patterns for domain predicates.
# Do NOT claim this recognizes unrestricted technical requirements language.
CAPABILITY_PATTERNS = {
    'measure_temperature': r'measur(?:e|es)\s+(?:the\s+)?(?:ambient\s+)?temperature',
    'send_data': r'(?:send|sends|transmit|transmits)\s+(?:the\s+)?data',
    'display_data': r'(?:display|displays|show|shows)\s+(?:the\s+)?data',
    'activate_alarm': r'(?:activate|activates|trigger|triggers)\s+(?:the\s+)?alarm',
}
NEGATION = r'(?:does\s+not|doesn\u2019t|doesn\'t|cannot|can\s+not|never)'

def extract_claims(text: str, candidate: str, capability: str):
    """Find narrow natural-language templates: 'Sensor measures temperature',
    'Sensor does not measure temperature', 'Sensor never measures temperature'.
    Returns exact source excerpts; does not infer unmentioned or complex facts.
    """
    if capability not in CAPABILITY_PATTERNS:
        raise ValueError(f'Unsupported capability {capability!r}; supported: {sorted(CAPABILITY_PATTERNS)}')
    head = re.escape(candidate).replace(r'\ ', r'\s+')
    action = CAPABILITY_PATTERNS[capability]
    # Candidate must be sentence subject and directly precede predicate.
    positive = re.compile(rf'(?<!\w)(?:the\s+)?{head}\s+(?:can\s+)?({action})(?!\w)', re.I)
    negative = re.compile(rf'(?<!\w)(?:the\s+)?{head}\s+{NEGATION}\s+({action})(?!\w)', re.I)
    found=[]
    for raw in re.split(r'(?<=[.!?])\s+|\n+', text):
        sentence=raw.strip()
        if not sentence: continue
        if negative.search(sentence): found.append((False, sentence))
        elif positive.search(sentence): found.append((True, sentence))
    return found

def run(manifest_path: Path, candidate: str, capability: str, version: str, score: float, threshold: float):
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    facts=[]; doc_audit=[]
    for doc in manifest['documents']:
        path=(manifest_path.parent / doc['path']).resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        text=path.read_text(encoding='utf8')
        matches=extract_claims(text,candidate,capability)
        doc_audit.append({'source_id':doc['source_id'],'path':str(path),
                          'claims_extracted':len(matches), 'source_excerpt_available':bool(text.strip())})
        for polarity, excerpt in matches:
            facts.append(Fact(source_id=doc['source_id'],candidate=candidate,capability=capability,
                              polarity=polarity,valid_versions=[str(v) for v in doc['valid_versions']],
                              authority=doc.get('authority','authoritative'),
                              superseded_by=doc.get('superseded_by'),excerpt=excerpt))
    result=gate(candidate,capability,version,score,facts,threshold)
    result['documents_inspected']=doc_audit
    result['extracted_claims']=[asdict(f) for f in facts]
    result['method']='restricted-pattern extraction from local plain-text documents (not full semantic extraction)'
    result['source_manifest']=str(manifest_path)
    result['limitations']=[
        'Only declared text files and listed verb/capability templates are recognized.',
        'Unmatched statements do not prove the corresponding capability is absent.',
        'Versions and supersession metadata must be supplied and checked by researchers.',
        'No automatic pronoun antecedent extraction or trained ranking model is used.',
        'The supplied --score is illustrative unless taken from a separately trained resolution model.',
    ]
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--candidate',required=True)
    p.add_argument('--capability',choices=sorted(CAPABILITY_PATTERNS),required=True)
    p.add_argument('--version',required=True)
    p.add_argument('--score',type=float,default=.82,help='Illustrative antecedent score; not a model prediction')
    p.add_argument('--min-score',type=float,default=.65)
    p.add_argument('--output',type=Path,default=Path('results/evidence_document_audit.json'))
    a=p.parse_args()
    result=run(a.manifest,a.candidate,a.capability,a.version,a.score,a.min_score)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding='utf8')
    print('Evidence status:',result['evidence']['status'])
    print('RaAAR decision:',result['decision'])
    print('Documents inspected:',len(result['documents_inspected']))
    print('Facts extracted:',len(result['extracted_claims']))
    for f in result['extracted_claims']:
        print(' -',f['source_id'],'POSITIVE' if f['polarity'] else 'NEGATIVE',repr(f['excerpt']))
    print('Full audit:',a.output)

if __name__=='__main__':main()
