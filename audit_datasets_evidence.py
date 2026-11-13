#!/usr/bin/env python3
"""RaAAR: audit evidence for every DAMIR and ReqEval pronoun record.

Only reports missing/outdated/contradictory when a specific, scoped evidence corpus
was supplied by the researcher. An absent annotation is NOT proof of missing facts.

Requires existing raaar_evidence.py and evidence_pipeline.py from RaAAR v2.
No model training is required.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

FIELDS = ["dataset", "requirement_id", "pronoun", "candidate", "capability", "target_version",
          "manifest", "evidence_origin", "linguistic_score", "candidate_options", "context"]
OUTPUT_FIELDS = ["dataset", "requirement_id", "pronoun", "candidate", "capability", "target_version",
                 "evidence_origin", "status", "decision", "reason", "documents_inspected",
                 "claims_extracted", "supporting_sources", "refuting_sources", "manifest", "error"]
VALID_ORIGIN = {"authentic", "synthetic"}


def read_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(f"Dataset missing: {path}; run prepare_data.py in your project first")
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from e


def load_records(project):
    sources = [("DAMIR", project / "processed" / "damir_all.jsonl"),
               ("ReqEval", project / "processed" / "reqeval_test.jsonl")]
    records = []
    for dataset, file in sources:
        for obj in read_jsonl(file):
            obj = dict(obj)
            obj["dataset"] = dataset
            obj["requirement_id"] = str(obj["id"])
            records.append(obj)
    pairs = [(r["dataset"], r["requirement_id"]) for r in records]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Duplicate (dataset, requirement_id) IDs; inspect data preparation")
    return records


def create_template(records, destination, overwrite=False):
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Annotation file already exists: {destination}. Not overwriting your work.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "dataset": r["dataset"], "requirement_id": r["requirement_id"],
                "pronoun": r.get("pronoun", ""),
                "candidate": "", "capability": "", "target_version": "", "manifest": "",
                "evidence_origin": "", "linguistic_score": "",
                "candidate_options": " | ".join(str(c.get("text", "")) for c in r.get("candidates", [])),
                "context": r.get("context", ""),
            })
    return destination


def read_annotations(path):
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}. First run --init.")
    keyed = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for required in ("dataset", "requirement_id", "candidate", "capability", "target_version", "manifest", "evidence_origin"):
            if required not in (reader.fieldnames or []):
                raise ValueError(f"Missing CSV column {required}")
        for line_number, row in enumerate(reader, 2):
            key = (row["dataset"].strip(), row["requirement_id"].strip())
            if key in keyed:
                raise ValueError(f"Duplicate annotation {key} on CSV line {line_number}; keep one per pronoun")
            keyed[key] = row
    return keyed


def audit_records(records, annotations, annotation_csv, project, threshold):
    # Import only when running an actual evidence audit. Does not load Torch or Hugging Face.
    from evidence_pipeline import run as check_evidence
    output = []
    used = set()
    for rec in records:
        key = (rec["dataset"], rec["requirement_id"])
        row = annotations.get(key, {})
        used.add(key)
        base = {"dataset": key[0], "requirement_id": key[1], "pronoun": rec.get("pronoun", ""),
                "candidate": row.get("candidate", "").strip(),
                "capability": row.get("capability", "").strip(),
                "target_version": row.get("target_version", "").strip(),
                "evidence_origin": row.get("evidence_origin", "").strip().lower(),
                "status": "not_assessed", "decision": "human_review", "reason": "", "documents_inspected": 0,
                "claims_extracted": 0, "supporting_sources": "", "refuting_sources": "",
                "manifest": row.get("manifest", "").strip(), "error": ""}
        core = ["candidate", "capability", "target_version", "manifest", "evidence_origin"]
        if not all(base[k] for k in core):
            base["reason"] = "missing_evidence_annotation"
            output.append(base)
            continue
        if base["evidence_origin"] not in VALID_ORIGIN:
            base["reason"] = "invalid_evidence_origin"
            base["error"] = "evidence_origin must be authentic or synthetic"
            output.append(base)
            continue
        candidates = {str(c.get("text", "")).strip().casefold() for c in rec.get("candidates", [])}
        if base["candidate"].casefold() not in candidates:
            base["reason"] = "candidate_not_in_requirement"
            base["error"] = "The supplied candidate must appear in this requirement's candidates"
            output.append(base)
            continue
        manifest = Path(base["manifest"])
        if not manifest.is_absolute():
            manifest = (project / manifest).resolve()
        if not manifest.is_file():
            base["reason"] = "manifest_not_found"
            base["error"] = str(manifest)
            output.append(base)
            continue
        try:
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            # IMPORTANT: require explicit scope. A generic document set must NOT
            # be silently reused for unrelated DAMIR/ReqEval requirements.
            scopes = manifest_data.get("requirement_keys", [])
            expected = f"{key[0]}:{key[1]}"
            if expected not in scopes:
                raise ValueError(f"Manifest must list {expected!r} in requirement_keys")
            if not manifest_data.get("documents"):
                raise ValueError("Manifest has no documents; insufficient evidence, not proven missing")
            raw_score = (row.get("linguistic_score") or "").strip()
            # No trained resolution score supplied? Use 0 for conservative abstention.
            score = float(raw_score) if raw_score else 0.0
            if not 0 <= score <= 1:
                raise ValueError("linguistic_score must lie between 0 and 1")
            result = check_evidence(manifest, base["candidate"], base["capability"],
                                    base["target_version"], score, threshold)
            assessment = result["evidence"]
            base["status"] = assessment["status"]
            base["decision"] = result["decision"]
            base["reason"] = result["reason"] if raw_score else "no_trained_resolution_score"
            base["documents_inspected"] = len(result["documents_inspected"])
            base["claims_extracted"] = len(result["extracted_claims"])
            base["supporting_sources"] = "; ".join(f["source_id"] for f in assessment["supporting_sources"])
            base["refuting_sources"] = "; ".join(f["source_id"] for f in assessment["refuting_sources"])
        except (ValueError, KeyError, OSError, TypeError) as exc:
            base["reason"] = "evidence_check_failed"
            base["error"] = str(exc)
        output.append(base)
    extra = set(annotations) - used
    if extra:
        print(f"WARNING: {len(extra)} annotation row(s) do not match any dataset requirement", file=sys.stderr)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd(), help="Existing RaAAR project folder")
    parser.add_argument("--annotations", default="evidence/dataset_evidence_annotations.csv")
    parser.add_argument("--output", default="results/damir_reqeval_evidence_audit.csv")
    parser.add_argument("--init", action="store_true", help="Create annotation template once; never overwrite")
    parser.add_argument("--min-score", type=float, default=.65)
    args = parser.parse_args()
    project = args.project.resolve()
    records = load_records(project)
    annotations_path = project / args.annotations
    output_path = project / args.output
    counts = Counter(r["dataset"] for r in records)
    print("Loaded:", ", ".join(f"{name}={count}" for name, count in sorted(counts.items())))
    if args.init:
        create_template(records, annotations_path)
        print("CREATED:", annotations_path)
        print("Complete candidate, capability, version, scoped manifest and origin for evidence-backed cases.")
        return
    annotations = read_annotations(annotations_path)
    results = audit_records(records, annotations, annotations_path, project, args.min_score)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    stats = {d: dict(Counter(r["status"] for r in results if r["dataset"] == d)) for d in ("DAMIR", "ReqEval")}
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({"total":len(records), "by_dataset":stats,
                                        "warning":"not_assessed means no validated evidence, NOT missing knowledge; synthetic results are demonstrations, not empirical proof"}, indent=2), encoding="utf-8")
    print("Status summary:", json.dumps(stats, indent=2))
    print("CSV:", output_path)
    print("Summary:", summary_path)

if __name__ == "__main__":
    main()
