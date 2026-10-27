import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from ghosttrace.config import EngineError,canonical,digest,validate_config,validate_dataset
from ghosttrace.storage.db import Store
from ghosttrace.analysis.metrics import distances
from ghosttrace.analysis.bootstrap import bootstrap_scenarios
from ghosttrace.engine.render import align_continuation

CONFIG={'schema_version':'1.0','name':'test','dataset':'test','model_profile':'qwen25-05b-cpu','conditions':['neutral','remorseful','unremorseful'],'templates':['A'],'modes':['fixed_continuation'],'capture':{'site':'residual_post_block_pre_final_norm','continuation_tokens':16,'storage_dtype':'float32','use_cache':False},'generation':{'do_sample':False,'max_new_tokens':96},'limits':{'max_input_tokens':512,'max_total_tokens':640,'batch_size':1},'analysis':{'bootstrap_draws':2000,'primary_metric':'normalized_l2'}}
class Tokenizer:
    def encode(self,text,add_special_tokens=False):return list(text.encode())

class Contracts(unittest.TestCase):
    def test_canonical_hash_order_and_nonfinite(self):
        self.assertEqual(digest({'a':1,'b':2}),digest({'b':2,'a':1}))
        with self.assertRaises(ValueError):canonical({'x':math.nan})
    def test_strict_config(self):
        self.assertEqual(validate_config(CONFIG)['name'],'test')
        c=copy.deepcopy(CONFIG);c['unexpected']=True
        with self.assertRaises(EngineError):validate_config(c)
        c=copy.deepcopy(CONFIG);c['capture']['use_cache']=True
        with self.assertRaises(EngineError):validate_config(c)
    def test_dataset_split_leakage(self):
        a={'id':'a','family_id':'family','split':'development','text':'test','common_continuation':'long enough'}
        b={**a,'id':'b','split':'holdout'}
        with self.assertRaises(EngineError):validate_dataset([a,b])
    def test_alignment_rejects_ids_and_limit(self):
        result=align_continuation(Tokenizer(),'prefix','abcdefghijklmnop')
        self.assertEqual(result['continuation_offset'],6)
        with self.assertRaises(EngineError):align_continuation(Tokenizer(),'prefix','abcdefghijklmnop',[0])
        with self.assertRaises(EngineError):align_continuation(Tokenizer(),'prefix','abcdefghijklmnop',max_tokens=10)
    def test_identity_and_hand_computable(self):
        self.assertEqual(distances([1,2],[1,2])['normalized_l2'],0)
        self.assertAlmostEqual(distances([1,0],[0,1])['normalized_l2'],math.sqrt(2),places=7)
        self.assertIsNone(distances([0,0],[1,0])['cosine_distance'])
    def test_scenario_bootstrap_deterministic(self):
        a=bootstrap_scenarios([1,2,3]);self.assertEqual(a,bootstrap_scenarios([1,2,3]));self.assertEqual(a['n'],3)
        self.assertEqual(bootstrap_scenarios([])['mean'],None)
    def test_queue_idempotency_claim_resume(self):
        with tempfile.TemporaryDirectory() as path:
            store=Store(path)
            a=store.queue(None,{'profile':'x'},1,'preflight','same')
            b=store.queue(None,{'profile':'x'},1,'preflight','same')
            self.assertEqual(a,b)
            with self.assertRaises(EngineError):store.queue(None,{'profile':'different'},1,'preflight','same')
            claimed=store.claim();self.assertEqual(claimed['id'],a['id']);self.assertIsNone(store.claim())
            store.update_job(a['id'],heartbeat_at='2000-01-01T00:00:00+00:00')
            self.assertEqual(store.interrupt_stale(),1);self.assertEqual(store.job(a['id'])['state'],'interrupted')
            resumed=store.queue(None,{'profile':'x'},1,'preflight','resume')
            self.assertEqual(store.job(resumed['id'])['attempt'],2)
if __name__=='__main__':unittest.main()
