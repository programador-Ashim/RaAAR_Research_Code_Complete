"""Convert actual DAMIR.xlsx and ReqEval.xlsx into reproducible JSONL files."""
import argparse, json, re
from collections import defaultdict, Counter
from pathlib import Path
import openpyxl
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent

def norm(v): return re.sub(r'\s+', ' ', str(v if v is not None else '')).strip()
def doc_key(row):
    """HEURISTIC grouping; audit IDs against original document metadata before publication."""
    return str(row['id']).split('-')[0]

def sheet_rows(path, sheet):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = wb[sheet].iter_rows(values_only=True)
    columns = [norm(x) for x in next(rows)]
    output = [dict(zip(columns, row)) for row in rows if any(v is not None for v in row)]
    wb.close()
    return output

def damir_examples(path):
    groups = defaultdict(list)
    for row in sheet_rows(path, 'Sheet1'):
        key = (norm(row['Id']), norm(row['Context']), norm(row['Pronoun']), norm(row['Position']))
        groups[key].append({'text': norm(row['Candidate Antecedent']), 'label': norm(row['ResolvedAs']).lower()})
    examples = []
    for (ident, context, pronoun, position), candidates in groups.items():
        labels = [c['label'] for c in candidates]
        label = 1 if labels and all(x == 'inconclusive' for x in labels) else (0 if 'correct' in labels else None)
        examples.append({'id': ident, 'context': context, 'pronoun': pronoun, 'position': position,
                         'source': 'DAMIR', 'candidates': candidates, 'ambiguity_label': label})
    return examples

def reqeval_examples(path):
    candidates = defaultdict(list)
    for row in sheet_rows(path, 'Resolution'):
        candidates[norm(row['ID'])].append({'text': norm(row['CandidateAntecedent(ajk)']),
                                            'label': norm(row['ResolvedAs']).lower()})
    examples = []
    for row in sheet_rows(path, 'Detection'):
        ident = norm(row['ID'])
        label = norm(row['DetectedAs']).upper()
        if label not in {'AMBIGUOUS', 'UNAMBIGUOUS'}: raise ValueError((ident, label))
        examples.append({'id': ident, 'context': norm(row['Context(cj)']),
                         'pronoun': norm(row['Pronoun (pj)']), 'position': None,
                         'source': 'ReqEval', 'candidates': candidates[ident],
                         'ambiguity_label': int(label == 'AMBIGUOUS')})
    return examples

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf8') as out:
        for row in rows: out.write(json.dumps(row, ensure_ascii=False) + '\n')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    p.add_argument('--processed-dir', type=Path, default=ROOT / 'processed')
    args = p.parse_args()
    damir = damir_examples(args.data_dir / 'DAMIR.xlsx')
    req = reqeval_examples(args.data_dir / 'ReqEval.xlsx')
    docs = sorted(set(doc_key(x) for x in damir))
    trainval_docs, test_docs = train_test_split(docs, test_size=.2, random_state=42)
    train_docs, valid_docs = train_test_split(trainval_docs, test_size=.2, random_state=42)
    partitions = {'damir_all': damir, 'reqeval_test': req,
                  'train': [x for x in damir if doc_key(x) in train_docs],
                  'validation': [x for x in damir if doc_key(x) in valid_docs],
                  'damir_test': [x for x in damir if doc_key(x) in test_docs]}
    for key, rows in partitions.items():
        write_jsonl(args.processed_dir / f'{key}.jsonl', rows)
        print(key, 'instances:', len(rows), 'labels:', dict(Counter(str(x['ambiguity_label']) for x in rows)))
    (args.processed_dir / 'split_docs.json').write_text(json.dumps({'train': train_docs,
        'validation': valid_docs, 'damir_test': test_docs, 'grouping_method': 'UNVERIFIED id-prefix heuristic'}, indent=2))
    print('IMPORTANT: document grouping by ID prefix must be audited before making publication claims.')

if __name__ == '__main__': main()
