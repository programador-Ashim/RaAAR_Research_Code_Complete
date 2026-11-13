import unittest
from pathlib import Path
import numpy as np
from rq12_train import (read_jsonl,rows_for,classification_metrics,all_positive_baseline,
                        ranking_predictions,select_threshold,group_report,split_data,doc_key)
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parent
FIXTURE=ROOT/'processed'

class RQ12Tests(unittest.TestCase):
    def test_all_positive_baseline(self):
        y=[1,1,0]
        got=all_positive_baseline(y)
        self.assertEqual(got['predicted_positive'],3)
        self.assertEqual(got['specificity'],0.0)
        self.assertAlmostEqual(got['f2'],10/11)

    def test_top1(self):
        rows=[dict(group='a',id='x',doc='d',pronoun='it',candidate='sensor',label=1),
              dict(group='a',id='x',doc='d',pronoun='it',candidate='display',label=0)]
        met,pred=ranking_predictions(rows,[0.9,0.1])
        self.assertEqual(met['top1_accuracy'],1.0)
        self.assertEqual(pred[0]['predicted_antecedent'],'sensor')

    def test_threshold_dev_only_example(self):
        y=[1,0,1,0]; s=[.9,.1,.8,.2]
        t=select_threshold(y,s,'balanced_accuracy')
        self.assertEqual(classification_metrics(y,s,t)['balanced_accuracy'],1.0)

    @unittest.skipUnless(FIXTURE.exists(),'Fixture processed/ files missing')
    def test_real_data_and_groups(self):
        damir=read_jsonl(FIXTURE/'damir_all.jsonl')
        req=read_jsonl(FIXTURE/'reqeval_test.jsonl')
        self.assertEqual(len(damir),587)
        self.assertEqual(len(req),109)
        self.assertEqual(len(rows_for(damir,'detection')),564)
        self.assertEqual(len(set(x['group'] for x in rows_for(damir,'resolution'))),266)
        outer=list(GroupKFold(n_splits=5).split(damir,groups=[doc_key(x) for x in damir]))
        for fold in outer:
            tr,dev,test=split_data(damir,fold,42)
            a,b,c=[{doc_key(x) for x in part} for part in (tr,dev,test)]
            self.assertFalse(a&b or a&c or b&c)

if __name__=='__main__':unittest.main(verbosity=2)
