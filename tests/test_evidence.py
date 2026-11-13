import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from raaar_evidence import gate,load_facts,Fact,analyze_evidence

class EvidenceTests(unittest.TestCase):
    def test_four_conditions(self):
        expected={'supported':'resolve','missing':'human_review',
                  'outdated':'human_review','contradictory':'human_review','refuted':'human_review'}
        for name,decision in expected.items():
            with self.subTest(name=name):
                output=gate('sensor','measure_temperature','2.0',.82,
                            load_facts(ROOT/'evidence'/f'{name}.json'))
                self.assertEqual(output['evidence']['status'],name)
                self.assertEqual(output['decision'],decision)
    def test_low_confidence_abstains(self):
        r=gate('sensor','measure_temperature','2.0',.4,load_facts(ROOT/'evidence/supported.json'))
        self.assertEqual(r['decision'],'human_review')
    def test_non_authoritative_not_support(self):
        f=Fact(source_id='x',candidate='sensor',capability='measure_temperature',
               polarity=True,valid_versions=['2.0'],authority='unknown')
        self.assertEqual(analyze_evidence('sensor','measure_temperature','2.0',[f])['status'],'missing')

if __name__=='__main__': unittest.main()
