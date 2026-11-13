"""Auditable evidence checker and selective decision prototype.
This is an operational RULE-BASED prototype, not a validated NLI or retrieval model.
Facts use structured candidate/capability tuples; false absence != negation.
"""
from dataclasses import dataclass, asdict
from typing import Optional
import json
from pathlib import Path

@dataclass
class Fact:
    source_id: str
    candidate: str
    capability: str
    polarity: bool
    valid_versions: list
    authority: str = 'authoritative'
    superseded_by: Optional[str] = None
    excerpt: str = ''

    @classmethod
    def from_dict(cls, d): return cls(**d)

def analyze_evidence(candidate, capability, version, facts):
    """Priority: CURRENT contradiction > CURRENT support > stale evidence > missing support.
    Outputs multiple flags so coverage, staleness, and conflicts can be audited independently.
    Non-authoritative facts are recorded but excluded from automatic resolution.
    This is *not* a semantic contradiction detector; facts must be annotated.
    """
    related=[f for f in facts if f.candidate.casefold()==candidate.casefold()
             and f.capability.casefold()==capability.casefold()]
    current=[f for f in related if version in f.valid_versions and not f.superseded_by
             and f.authority == 'authoritative']
    positive=[f for f in current if f.polarity]
    negative=[f for f in current if not f.polarity]
    historic=[f for f in related if f not in current]
    contradiction=bool(positive and negative)
    missing=not bool(current)
    outdated=missing and any((version not in f.valid_versions or f.superseded_by) and f.authority == 'authoritative' for f in historic)
    # A valid negative assertion says candidate does not have capability; no positive support.
    supported=bool(positive) and not contradiction
    status=('contradictory' if contradiction else 'supported' if supported else
            'refuted' if negative else 'outdated' if outdated else 'missing')
    return {'candidate':candidate,'capability':capability,'target_version':version,
            'status':status,'flags':{'missing_current_evidence':missing,
                                   'outdated_evidence_found':outdated,
                                   'contradictory_current_claims':contradiction},
            'supporting_sources':[asdict(f) for f in positive],
            'refuting_sources':[asdict(f) for f in negative],
            'historical_or_inapplicable_sources':[asdict(f) for f in historic],
            'limitations':'Structured matching only; no automatic fact extraction or NLP retrieval.'}

def gate(candidate, capability, version, linguistic_score, facts, min_score=.65):
    audit=analyze_evidence(candidate,capability,version,facts)
    accepted=audit['status']=='supported' and linguistic_score >= min_score
    return {'predicted_antecedent':candidate,'linguistic_score':float(linguistic_score),
            'decision':'resolve' if accepted else 'human_review',
            'reason':'supported_and_confident' if accepted else
                     ('low_linguistic_confidence' if audit['status']=='supported' else audit['status']),
            'evidence':audit}

def load_facts(path):
    content=json.loads(Path(path).read_text(encoding='utf8'))
    return [Fact.from_dict(f) for f in content['facts']]
