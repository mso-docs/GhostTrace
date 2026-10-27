import unittest
from ghosttrace.analysis.ratings import weighted_kappa

class Agreement(unittest.TestCase):
    def test_perfect_nonconstant(self):
        result=weighted_kappa([(0,0),(1,1),(2,2)])
        self.assertEqual(result['weighted_kappa'],1);self.assertEqual(result['agreement'],1)
    def test_hand_computable_quadratic_weights(self):
        # Marginals A=[.5,0,.5], B=[0,.5,.5]: observed=.125, expected=.375 => kappa=2/3.
        result=weighted_kappa([(0,1),(2,2)])
        self.assertAlmostEqual(result['weighted_kappa'],2/3);self.assertEqual(result['agreement'],.5)
    def test_undefined_not_zero(self):
        self.assertIsNone(weighted_kappa([(1,1),(1,1)])['weighted_kappa'])
        self.assertEqual(weighted_kappa([])['reason'],'no_paired_ratings')
    def test_invalid_scale(self):
        with self.assertRaises(ValueError):weighted_kappa([(3,1)])

class BlindCohort(unittest.TestCase):
    def test_shared_balanced_cohort_per_rater_hidden_metadata_and_reliability(self):
        import json
        import tempfile
        import uuid
        from ghosttrace.storage.db import Store
        from ghosttrace.config import load_config,canonical,now
        from ghosttrace.engine.planner import plan
        from ghosttrace.analysis.ratings import blinded_tasks,rating_summary,DIMENSIONS
        with tempfile.TemporaryDirectory() as temp:
            store=Store(temp);exp=plan(store,load_config('configs/balanced.yaml'))
            with store.connect() as db:
                for unit in exp['ordered_units']:
                    if unit['mode']!='greedy_generation':continue
                    record={**unit,'experiment_id':exp['id'],'status':'completed','text':'TEST response with no condition metadata.','scenario':{'text':'TEST scenario'},'created_at':now()}
                    db.execute('INSERT INTO units VALUES(?,?,?,?,?,?,?)',(unit['id'],exp['id'],canonical(record).decode(),'completed',None,None,now()))
            first=blinded_tasks(store,exp['id'],'rater-A');second=blinded_tasks(store,exp['id'],'rater-B')
            self.assertEqual(len(first['items']),36)
            self.assertEqual({x['run_unit_id'] for x in first['items']},{x['run_unit_id'] for x in second['items']})
            self.assertEqual(first['cohort_id'],second['cohort_id'])
            for item in first['items']:
                self.assertFalse({'condition','template_id','model_profile','activation_artifact'}&set(item))
            for index,item in enumerate(first['items'][:3]):
                for rater in ['rater-A','rater-B']:
                    score=index%3
                    rating={'id':str(uuid.uuid4()),'run_unit_id':item['run_unit_id'],'rater_id':rater,'rubric_version':'expressed-behavior-v1','dimension_scores':{d:score for d in DIMENSIONS},'created_at':now()}
                    store.put_record('ratings',rating)
            remaining=blinded_tasks(store,exp['id'],'rater-A');self.assertEqual(len(remaining['items']),33)
            self.assertEqual(remaining['cohort_id'],first['cohort_id'])
            summary=rating_summary(store,exp['id']);self.assertEqual(summary['rater_count'],2)
            self.assertEqual(summary['pairwise_agreement'][0]['paired_response_count'],3)
            self.assertEqual(summary['pairwise_agreement'][0]['dimensions']['regret']['weighted_kappa'],1)
