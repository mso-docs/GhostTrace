import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from ghosttrace.api.app import create_app
from ghosttrace.research.service import run_captures,run_intervention
from ghosttrace.worker.main import work
from test_service import PROFILE,CONFIG,FakeMechanistic

class ResearchAPI(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.app=create_app(self.temp.name);self.store=self.app.state.store;self.store.put_record('profiles',PROFILE);self.client=TestClient(self.app);self.headers={'X-GhostTrace-Token':self.client.get('/api/v1/health').json()['mutation_token']}
    def tearDown(self):self.client.close();self.temp.cleanup()
    def test_typed_general_study_capture_tiles_and_bundle_export_import(self):
        created=self.client.post('/api/v2/research/studies',json=CONFIG,headers=self.headers);self.assertEqual(created.status_code,201,created.text);study=created.json();self.assertEqual(study['schema_version'],'2.0')
        queued=self.client.post(f"/api/v2/research/studies/{study['id']}/captures",json={},headers=self.headers).json();self.assertEqual(queued['kind'],'research_capture')
        with patch('ghosttrace.research.worker.run_captures',side_effect=lambda store,job:run_captures(store,job,FakeMechanistic())):work(self.temp.name,once=True)
        job=self.client.get('/api/v1/jobs/'+queued['id']).json();self.assertEqual(job['state'],'completed',job);capture_id=job['result']['capture_ids'][0]
        tile=self.client.get(f'/api/v2/research/captures/{capture_id}/tiles?array=L0.resid_post').json();self.assertEqual(tile['values'],[[1,1,1,1]])
        bundle=self.client.get(f"/api/v2/research/studies/{study['id']}/bundle").json();self.assertEqual(len(bundle['captures']),2)
        exported=self.client.post(f"/api/v2/research/studies/{study['id']}/exports",json={'include_arrays':True},headers=self.headers).json();work(self.temp.name,once=True);job=self.store.job(exported['id']);self.assertEqual(job['state'],'completed',job)
        download=self.client.get('/api/v2/research/exports/'+job['result']['export_id']+'/download');self.assertEqual(download.status_code,200)
        imported=self.client.post('/api/v2/research/imports',content=download.content,headers={**self.headers,'Content-Type':'application/zip'});self.assertEqual(imported.status_code,202,imported.text);work(self.temp.name,once=True);self.assertEqual(self.store.job(imported.json()['id'])['state'],'completed')
    def test_unknown_fields_and_candidate_limits_are_rejected(self):
        response=self.client.post('/api/v2/research/studies',json={**CONFIG,'invented_capability':True},headers=self.headers);self.assertEqual(response.status_code,422)
        invalid={**CONFIG,'target':{'kind':'continuation_loglikelihood','positive_ids':list(range(65)),'negative_ids':[1]}}
        self.assertEqual(self.client.post('/api/v2/research/studies',json=invalid,headers=self.headers).status_code,422)
        capabilities=self.client.get('/api/v2/research/capabilities').json();self.assertIn('attn_in',capabilities['sites']);self.assertIn('mlp_in',capabilities['sites'])
    def test_v1_protocol_restrictions_remain_independent(self):
        from ghosttrace.config import load_config
        config=load_config('configs/smoke.yaml');config['conditions']=['blue','red']
        response=self.client.post('/api/v1/experiments',json=config,headers=self.headers);self.assertEqual(response.status_code,422)
