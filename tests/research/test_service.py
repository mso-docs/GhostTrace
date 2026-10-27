"""Synthetic adapter contracts. No arrays in these tests are released as model evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from ghosttrace.config import EngineError
from ghosttrace.storage.db import Store
from ghosttrace.research.schemas import StudyCreate
from ghosttrace.research.service import freeze_study,queue_capture,queue_intervention,run_captures,run_intervention,get_bundle,save_hypothesis
from ghosttrace.research.bundles import export_research,import_research

PROFILE={'id':'test-research','repo_id':'fixture/OnlyTests','revision':'a'*40,'tokenizer_revision':'a'*40,'layers':2,'hidden_size':4,'device':'cpu','compute_dtype':'float32','threads':2,'preflight_passed':True}
CONFIG={'schema_version':'2.0','name':'Test copying','question':'Which component affects this target?','hypothesis':'A literal residual patch may change the target.','model_profile_id':PROFILE['id'],'conditions':[{'id':'clean','name':'Clean','messages':[{'role':'user','content':'Copy A'}]},{'id':'corrupt','name':'Corrupt','messages':[{'role':'user','content':'Copy B'}]}],'target':{'kind':'next_token_logit_diff','positive':'A','negative':'B'},'capture':{'layers':[0,1],'positions':[-1],'sites':['resid_pre','attn_out','resid_mid','mlp_out','resid_post']}}
class Tokenizer:
    all_special_ids=[]
    def convert_ids_to_tokens(self,ids):return [chr(i) for i in ids]
class FakeMechanistic:
    def __init__(self):self.tokenizer=Tokenizer();self.loaded=False;self.calls=0
    def load(self,profile):self.profile=copy.deepcopy(profile);self.loaded=True;return self.profile
    def describe(self):return self.profile
    def unload(self):self.loaded=False
    def seed(self,seed):pass
    def render(self,messages):return '\n'.join(m['content'] for m in messages)+'!'
    def tokenize(self,text):return [ord(c) for c in text]
    def inspect(self,token_ids,*,layers,positions,target,attention=False,top_k=5):
        self.calls+=1;arrays={};components=[]
        for layer in layers:
            for site in ['resid_pre','attn_out','resid_mid','mlp_out','resid_post']:
                key=f'L{layer}.{site}';arrays[key]=np.ones((len(positions),4),np.float32)*(layer+1)
                components.append({'site':site,'layer':layer,'array_key':key,'shape':list(arrays[key].shape),'axis_names':['token','hidden'],'token_indices':positions})
        return {'artifact_arrays':arrays,'components':components,'metrics':{'target_score':0.25,'logit_lens':[{'layer':layer,'logit_difference':0.25} for layer in layers],'logit_parity_max_abs_error':0},'graph':{'nodes':[{'id':'L0.resid_post','label':'Residual after block 0','layer':0,'site':'resid_post','kind':'component'}],'edges':[]},'metadata':{'test_fixture':True}}
    def intervene(self,donor,recipient,**kwargs):
        self.calls+=1;return {'metrics':{'baseline':0,'donor':1,'patched':.5,'delta':.5,'recovery':.5,'controls':{'self_patch':0}},'graph':{'nodes':[],'edges':[]},'artifact_arrays':{},'metadata':{'test_fixture':True}}
    def attribute(self,*args,**kwargs):return self.intervene(*args,**kwargs)
    def path_patch(self,*args,**kwargs):return self.intervene(*args,**kwargs)

class Service(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.store=Store(self.temp.name);self.store.put_record('profiles',PROFILE)
    def tearDown(self):self.temp.cleanup()
    def test_freeze_capture_graph_tiles_and_verified_reuse(self):
        study=freeze_study(self.store,CONFIG);self.assertEqual(study['id'],freeze_study(self.store,CONFIG)['id'])
        queue_capture(self.store,study['id']);job=self.store.claim();adapter=FakeMechanistic();result=run_captures(self.store,job,adapter)
        self.assertFalse(adapter.loaded);self.assertEqual(len(result['capture_ids']),2);bundle=get_bundle(self.store,study['id'])
        self.assertEqual(len(bundle['captures']),2);capture=bundle['captures'][0];self.assertEqual(capture['tiles'][0]['values'],[[1,1,1,1]])
        self.assertIn('omitted_value_count',capture['tiles'][0]);self.assertTrue(bundle['graph']['nodes'])
        self.store.update_job(job['id'],state='completed');queue_capture(self.store,study['id']);again=self.store.claim();reuse=FakeMechanistic();run_captures(self.store,again,reuse);self.assertEqual(reuse.calls,0)
    def test_alignment_exclusion_and_exact_intervention(self):
        study=freeze_study(self.store,CONFIG)
        spec={'donor_condition_id':'clean','recipient_condition_id':'corrupt','site':'resid_post','layer':0,'positions':[-1],'operation':'patch'}
        queue_intervention(self.store,study['id'],spec);job=self.store.claim();result=run_intervention(self.store,job,FakeMechanistic());self.assertEqual(get_bundle(self.store,study['id'])['interventions'][0]['raw_delta'],.5)
        self.store.update_job(job['id'],state='completed');spec['positions']=[-2];queue_intervention(self.store,study['id'],spec)
        with self.assertRaises(EngineError) as caught:run_intervention(self.store,self.store.claim(),FakeMechanistic())
        self.assertEqual(caught.exception.code,'TOKEN_ALIGNMENT_FAILED')
    def test_portable_research_roundtrip_is_read_only_and_relocated(self):
        study=freeze_study(self.store,CONFIG);queue_capture(self.store,study['id']);run_captures(self.store,self.store.claim(),FakeMechanistic());exported=export_research(self.store,study['id'])
        with tempfile.TemporaryDirectory() as root:
            target=Store(root);result=import_research(target,self.store.root/exported['relative_path']);self.assertTrue(result['verification']['immutable_bindings_verified']);self.assertGreater(result['verification']['capture_tiles_recomputed'],0)
            self.assertEqual(get_bundle(target,study['id'])['study'],study)
            with target.connect() as db:self.assertEqual(db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0],0);self.assertEqual(db.execute('SELECT COUNT(*) FROM profiles').fetchone()[0],0)
            again=export_research(target,study['id']);self.assertTrue((target.root/again['relative_path']).is_file())
    def test_hypotheses_are_annotations_and_invalid_study_rejected(self):
        invalid=copy.deepcopy(CONFIG);invalid['conditions'][1]['id']='clean'
        with self.assertRaises(ValueError):StudyCreate.model_validate(invalid)
        study=freeze_study(self.store,CONFIG);hypothesis=save_hypothesis(self.store,study['id'],{'question':'Which?','prior_hypothesis':'A prior test hypothesis','protocol':'Selected block patch','expected_effect':'Possibly no effect'})
        self.assertTrue(hypothesis['registered_before_results']);self.assertEqual(hypothesis['evidence_type'],'human_interpretation')
