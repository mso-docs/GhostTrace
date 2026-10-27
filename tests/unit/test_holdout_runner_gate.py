"""Constructed receipt fixtures test gating only; no human review is performed."""
import copy
import tempfile
import unittest
import uuid
from ghosttrace.config import canonical,digest,now
from ghosttrace.engine.runner import run_job
from ghosttrace.storage.db import Store

class NeverLoadAdapter:
    def __init__(self):self.loaded=False
    def load(self,profile):self.loaded=True;raise AssertionError('Model must not load before review gates pass')
    def unload(self):pass

class HoldoutRunnerGate(unittest.TestCase):
    def fixture(self):
        profile={'id':'test-fixture','preflight_passed':True}
        config={'model_profile':profile['id']}
        scenarios=[{'id':'test-heldout','split':'holdout','review_status':'reviewed'}]
        controls=[{'id':'test-control','review_status':'reviewed'}]
        templates=[{'id':'test-template','review_status':'reviewed'}]
        review={'reviewer':'TEST FIXTURE ONLY','reviewed_dataset_hash':digest(scenarios),'reviewed_templates_hash':digest(templates),'reviewed_controls_hash':digest(controls)}
        receipt={'config_hash':digest(config),'dataset_hash':digest(scenarios),'template_hash':digest(templates),'human_review_hash':digest(review),'protocol_sha256':'test-protocol'}
        exp={'id':str(uuid.uuid4()),'config_hash':'a'*64,'config':config,'scenarios':scenarios,'control_scenarios':controls,'prompt_templates':templates,'dataset_hash':digest(scenarios),'model_profile_hash':digest(profile),'preregistration_hash':'test-protocol','human_review':review,'human_review_hash':digest(review),'preregistration_receipt':receipt,'preregistration_receipt_hash':digest(receipt)}
        return profile,exp

    def assert_blocked(self,change):
        profile,exp=self.fixture();change(exp)
        with tempfile.TemporaryDirectory() as root:
            store=Store(root);store.put_record('profiles',profile)
            with store.connect() as db:db.execute('INSERT INTO experiments VALUES(?,?,?,?)',(exp['id'],exp['config_hash'],canonical(exp).decode(),now()))
            store.queue(exp['id'],{},1);job=store.claim();adapter=NeverLoadAdapter();run_job(store,job,adapter)
            self.assertFalse(adapter.loaded);self.assertEqual(store.job(job['id'])['error_code'],'INVALID_CONFIG')

    def test_unreviewed_factual_control_blocks_before_model_load(self):
        self.assert_blocked(lambda exp:exp['control_scenarios'][0].update(review_status='pending_human_review'))
    def test_control_hash_mismatch_blocks_before_model_load(self):
        self.assert_blocked(lambda exp:exp['human_review'].update(reviewed_controls_hash='wrong'))
    def test_preregistration_template_hash_mismatch_blocks_before_model_load(self):
        self.assert_blocked(lambda exp:exp['preregistration_receipt'].update(template_hash='wrong'))
    def test_frozen_review_receipt_hash_mismatch_blocks_before_model_load(self):
        self.assert_blocked(lambda exp:exp.update(human_review_hash='wrong'))
