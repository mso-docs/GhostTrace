import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from ghosttrace.storage.artifacts import write_activation,read_activation
from ghosttrace.config import EngineError,load_config
from ghosttrace.storage.db import Store
from ghosttrace.engine.planner import plan
from ghosttrace.analysis.report import analyze,export_bundle

class ArtifactContracts(unittest.TestCase):
    def test_artifact_checksum_quota_and_array_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'trace.npz';array=np.arange(24,dtype=np.float32).reshape(2,3,4)
            sha=write_activation(path,array,100000,temp)
            np.testing.assert_array_equal(read_activation(path,sha),array)
            with self.assertRaises(EngineError):read_activation(path,'0'*64)
            with self.assertRaises(EngineError):write_activation(path,array,1,temp)
    def test_not_run_is_missing_not_zero_and_export_hashes(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            store=Store(temp);exp=plan(store,load_config('configs/smoke.yaml'))
            result=analyze(store,exp['id'])
            self.assertEqual(result['measurement_status'],'not_run')
            self.assertEqual(result['expected_unit_count'],24)
            self.assertEqual(len(result['exclusions_with_reasons']),24)
            again=analyze(store,exp['id'])
            self.assertNotEqual(result['id'],again['id'])
            self.assertEqual(len(list((store.root/exp['config_hash']/'analyses').glob('*/metrics.json'))),2)
            self.assertIsNone(result['aggregates']['remorseful-neutral']['primary']['mean'])
            self.assertEqual(result['aggregates']['remorseful-neutral']['primary']['n'],0)
            exported=export_bundle(store,result['id'])
            with zipfile.ZipFile(store.root/exported['relative_path']) as archive:
                inventory=json.loads(archive.read('inventory.json'))
                for name,sha in inventory['file_inventory_with_checksums'].items():self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(),sha)
    def test_exact_budget_frozen_templates_and_draft_review(self):
        with tempfile.TemporaryDirectory() as temp:
            store=Store(temp);exp=plan(store,load_config('configs/balanced.yaml'))
            self.assertEqual(exp['estimated_units'],444)
            self.assertEqual(len(exp['scenarios']),30);self.assertEqual(len(exp['control_scenarios']),12)
            self.assertEqual(len(exp['prompt_templates']),7)
            self.assertTrue(any('human review' in w for w in exp['warnings']))
