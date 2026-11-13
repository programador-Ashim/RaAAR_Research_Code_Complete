import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from evidence_pipeline import extract_claims, run

ROOT=Path(__file__).resolve().parent.parent
class DocumentEvidenceTests(unittest.TestCase):
    def test_extraction_polarity(self):
        s='The sensor measures temperature. The sensor does not measure temperature.'
        self.assertEqual([p for p,_ in extract_claims(s,'sensor','measure_temperature')],[True,False])
    def test_five_scenarios(self):
        for scenario,decision in [('supported','resolve'),('missing','human_review'),
                                  ('outdated','human_review'),('contradictory','human_review'),
                                  ('refuted','human_review')]:
            with self.subTest(scenario=scenario):
                result=run(ROOT/'evidence'/f'{scenario}_documents.json','sensor',
                           'measure_temperature','2.0',.82,.65)
                self.assertEqual(result['evidence']['status'],scenario)
                self.assertEqual(result['decision'],decision)
if __name__=='__main__':unittest.main()
