"""Regression cases for evidence evaluation: fabricated separation must never pass."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location('evidence_evaluator', Path(__file__).parents[2] / 'scripts/evaluate_evidence.py')
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


def fixtures():
    manifest = {'scenarios':[{'family_id':'a','split':'development'}], 'model_profile':{}, 'config':{'templates':['A']}}
    analysis = {'measurement_status':'not_run', 'settings':{'metric_version':'normalized-l2-v1','bootstrap_draws':2000,'split':'development'}, 'aggregates':{'remorseful-neutral':{'primary':{'n':0,'mean':None,'lower':None,'upper':None}}}}
    return manifest, analysis, []


class EvidenceEvaluation(unittest.TestCase):
    def test_unrun_is_pending_never_measured(self):
        m, a, r = fixtures()
        checks = evaluator.evaluate_records(m, a, r)
        self.assertEqual(next(c for c in checks if c['id'] == 'measurement_available')['status'], 'pending')
        self.assertFalse(any(c['status'] == 'fail' for c in checks))
        strict = evaluator.evaluate_records(m, a, r, require_measured=True)
        self.assertEqual(next(c for c in strict if c['id'] == 'measurement_available')['status'], 'fail')

    def test_empty_zero_bars_are_rejected(self):
        m, a, r = fixtures()
        a['aggregates']['remorseful-neutral']['primary']['mean'] = 0
        checks = evaluator.evaluate_records(m, a, r)
        self.assertEqual(next(c for c in checks if c['id'] == 'missingness_not_zero')['status'], 'fail')

    def test_fixture_cannot_claim_real_measurement(self):
        m, a, r = fixtures()
        m['fixture_type'] = 'illustrative'
        a['measurement_status'] = 'measured'
        checks = evaluator.evaluate_records(m, a, r)
        self.assertEqual(next(c for c in checks if c['id'] == 'fixture_separation')['status'], 'fail')
        self.assertEqual(next(c for c in checks if c['id'] == 'three_conditions')['status'], 'fail')

    def test_split_leakage_and_nonfinite_values_fail(self):
        m, a, r = fixtures()
        m['scenarios'].append({'family_id':'a','split':'holdout'})
        a['leaked'] = float('nan')
        checks = evaluator.evaluate_records(m, a, r)
        self.assertEqual(next(c for c in checks if c['id'] == 'family_split_separation')['status'], 'fail')
        self.assertEqual(next(c for c in checks if c['id'] == 'finite_json')['status'], 'fail')

    def test_archive_inventory_and_traversal(self):
        m, a, _ = fixtures()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'safe.zip'
            files = {'manifest.json':json.dumps(m).encode(), 'analysis.json':json.dumps(a).encode()}
            inventory = {'includes_raw_activations':False, 'file_inventory_with_checksums':{n:hashlib.sha256(v).hexdigest() for n, v in files.items()}}
            with zipfile.ZipFile(path, 'w') as z:
                for n, v in files.items():z.writestr(n,v)
                z.writestr('inventory.json',json.dumps(inventory))
            self.assertEqual(evaluator.evaluate_bundle(path)['status'], 'pending')
            with zipfile.ZipFile(path, 'a') as z:z.writestr('../escape.txt','bad')
            result = evaluator.evaluate_bundle(path)
            self.assertEqual(result['status'], 'fail')
            self.assertEqual(result['checks'][0]['status'], 'fail')

    def test_corrupt_archive_is_structured_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bad.zip';path.write_bytes(b'not a ZIP')
            result = evaluator.evaluate_bundle(path)
            self.assertEqual(result['status'], 'fail')
            self.assertEqual(result['checks'][0]['id'], 'readable_contract')
