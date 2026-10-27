"""Changed environments must not reuse a successful preflight or mix resumed units."""
import copy
import tempfile
import unittest
from ghosttrace.config import load_config
from ghosttrace.engine.planner import plan
from ghosttrace.engine.runner import run_job,require_compatible_backend
from ghosttrace.models.profiles import get_profile
from ghosttrace.storage.db import Store
from test_recovery import FixtureAdapter

class DriftAdapter(FixtureAdapter):
    def __init__(self,store,field,value):super().__init__(store);self.field=field;self.value=value
    def load(self,profile):self.profile={**copy.deepcopy(profile),self.field:self.value};return self.profile

class BackendDrift(unittest.TestCase):
    def test_dtype_version_thread_drift_blocks_before_first_unit_and_on_resume(self):
        for field,value in [('compute_dtype','float16'),('torch_version','changed'),('threads',3),('package_versions',{'torch':'changed'})]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as temp:
                store=Store(temp);profile={**get_profile('qwen25-05b-cpu'),'revision':'a'*40,'tokenizer_revision':'a'*40,'preflight_passed':True,'device':'cpu','compute_dtype':'float32','torch_version':'validated','threads':2,'package_versions':{'torch':'validated'}};store.put_record('profiles',profile)
                config=load_config('configs/smoke.yaml');config['model_profile']=profile['id'];exp=plan(store,config)
                store.queue(exp['id'],{},exp['estimated_units']);first=store.claim();adapter=DriftAdapter(store,field,value);run_job(store,first,adapter)
                failed=store.job(first['id']);self.assertEqual(failed['error_code'],'CONFLICT');self.assertEqual(adapter.calls,0);self.assertEqual(store.unit_rows(exp['id']),[])
                store.queue(exp['id'],{},exp['estimated_units']);second=store.claim();resumed=DriftAdapter(store,field,value);run_job(store,second,resumed)
                self.assertEqual(store.job(second['id'])['error_code'],'CONFLICT');self.assertEqual(resumed.calls,0)
    def test_nonnumerical_preflight_timing_and_source_hash_noise_are_excluded(self):
        validated={'compute_dtype':'float32','threads':2,'torch_version':'same','load_and_test_seconds':1,'code_provenance':{'source_hash':'old'}}
        actual={**validated,'load_and_test_seconds':2,'code_provenance':{'source_hash':'new'}}
        require_compatible_backend(validated,actual)

    def test_source_drift_is_blocked_for_canonical_holdout_only(self):
        from ghosttrace.config import EngineError
        validated={'compute_dtype':'float32','code_provenance':{'source_sha256':'old'}}
        actual={'compute_dtype':'float32','code_provenance':{'source_sha256':'new'}}
        require_compatible_backend(validated,actual,canonical_holdout=False)
        with self.assertRaises(EngineError) as caught:require_compatible_backend(validated,actual,canonical_holdout=True)
        self.assertEqual(caught.exception.code,'CONFLICT')
        self.assertIn('code_provenance.source_sha256',caught.exception.details['mismatched_fields'])

    def test_adapter_cannot_evade_guard_by_mutating_input_profile_in_place(self):
        class InPlaceAdapter(FixtureAdapter):
            def load(self,profile):self.profile=profile;profile['threads']=999;return profile
        with tempfile.TemporaryDirectory() as temp:
            store=Store(temp);profile={**get_profile('qwen25-05b-cpu'),'revision':'a'*40,'tokenizer_revision':'a'*40,'preflight_passed':True,'device':'cpu','compute_dtype':'float32','threads':2};store.put_record('profiles',profile)
            config=load_config('configs/smoke.yaml');config['model_profile']=profile['id'];exp=plan(store,config)
            store.queue(exp['id'],{},exp['estimated_units']);job=store.claim();adapter=InPlaceAdapter(store);run_job(store,job,adapter)
            self.assertEqual(store.job(job['id'])['error_code'],'CONFLICT');self.assertEqual(adapter.calls,0)
            self.assertEqual(store.get_record('profiles',profile['id'])['threads'],2)
