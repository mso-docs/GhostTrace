"""Synthetic adapter contract test; these arrays are test fixtures, never demo evidence."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from ghosttrace.config import load_config
from ghosttrace.storage.db import Store
from ghosttrace.engine.planner import plan
from ghosttrace.engine.runner import run_job
from ghosttrace.models.profiles import get_profile

class CharacterTokenizer:
    def encode(self,text,add_special_tokens=False):return list(text.encode('ascii',errors='replace'))
    def convert_ids_to_tokens(self,ids):return [chr(i) for i in ids]
    def __call__(self,text,**kwargs):return {'offset_mapping':[(i,i+1) for i in range(len(text))]}
class FixtureAdapter:
    def __init__(self,store,cancel_job=None):self.tokenizer=CharacterTokenizer();self.store=store;self.cancel_job=cancel_job;self.calls=0
    def load(self,profile):self.profile=profile;return profile
    def describe(self):return self.profile
    def unload(self):pass
    def seed(self,seed):pass
    def render(self,messages):return '\n'.join(x['content'] for x in messages)+'\nassistant: '
    def tokenize(self,text):return self.tokenizer.encode(text)
    def generate(self,ids,settings,seed):
        self.calls+=1;return {'text':'TEST FIXTURE','generated_ids':[65],'generated_token_count':1,'truncated':False,'stop_reason':'eos','seconds':0.01}
    def capture(self,ids,site,positions):
        self.calls+=1
        if self.cancel_job:self.store.update_job(self.cancel_job,cancel_requested=1);self.cancel_job=None
        return np.ones((24,len(positions),4),dtype=np.float32)
    def score_continuation(self,prefix,continuation):return 1.0

class Recovery(unittest.TestCase):
    def test_cooperative_cancel_verified_resume_and_corruption_block(self):
        with tempfile.TemporaryDirectory() as temp:
            store=Store(temp)
            profile={**get_profile('qwen25-05b-cpu'),'revision':'a'*40,'tokenizer_revision':'a'*40,'preflight_passed':True,'device':'cpu','compute_dtype':'float32'}
            store.put_record('profiles',profile)
            config=load_config('configs/smoke.yaml');config['model_profile']=profile['id'];config['limits']['max_input_tokens']=2048;config['limits']['max_total_tokens']=2144
            exp=plan(store,config)
            first=store.queue(exp['id'],{},exp['estimated_units']);job=store.claim()
            adapter=FixtureAdapter(store,job['id']);run_job(store,job,adapter)
            self.assertEqual(store.job(job['id'])['state'],'cancelled')
            first_rows=[r for r in store.unit_rows(exp['id']) if r['status']=='completed']
            self.assertGreater(len(first_rows),0)
            old_records={r['id']:r['record'] for r in first_rows}
            resumed=store.queue(exp['id'],{},exp['estimated_units'],idempotency_key='resume-test');job2=store.claim()
            next_adapter=FixtureAdapter(store);run_job(store,job2,next_adapter)
            self.assertEqual(store.job(job2['id'])['state'],'completed')
            self.assertEqual(next_adapter.calls,exp['estimated_units']-len(first_rows))
            for r in store.unit_rows(exp['id']):
                if r['id'] in old_records:self.assertEqual(r['record'],old_records[r['id']])
            artifact=next(r for r in first_rows if r['artifact_path'])
            (store.root/artifact['artifact_path']).write_bytes(b'corrupt')
            store.queue(exp['id'],{},exp['estimated_units'],idempotency_key='corruption-test');job3=store.claim();run_job(store,job3,FixtureAdapter(store))
            self.assertEqual(store.job(job3['id'])['error_code'],'ARTIFACT_CORRUPT')
