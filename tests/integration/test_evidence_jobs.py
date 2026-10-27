import tempfile
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from ghosttrace.api.app import create_app
from ghosttrace.config import load_config
from ghosttrace.engine.planner import plan
from ghosttrace.worker.main import work
from ghosttrace.analysis.report import export_bundle,analyze

class EvidenceJobs(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.app=create_app(self.root);self.client=TestClient(self.app)
        self.headers={'X-GhostTrace-Token':self.client.get('/api/v1/health').json()['mutation_token']}
        self.exp=plan(self.app.state.store,load_config('configs/smoke.yaml'))
    def tearDown(self):self.client.close();self.temp.cleanup()
    def test_analysis_export_and_summary_import_are_real_queued_jobs(self):
        response=self.client.post(f"/api/v1/experiments/{self.exp['id']}/analyses",json={'split':'development'},headers={**self.headers,'Idempotency-Key':'analysis'})
        self.assertEqual(response.status_code,202,response.text);queued=response.json();self.assertEqual(queued['kind'],'analysis')
        self.assertEqual(self.client.get(f"/api/v1/jobs/{queued['id']}").json()['state'],'queued')
        work(self.root,once=True)
        completed=self.client.get(f"/api/v1/jobs/{queued['id']}").json();self.assertEqual(completed['state'],'completed',completed)
        analysis_id=completed['result']['analysis_id'];self.assertEqual(self.client.get(f'/api/v1/analyses/{analysis_id}').status_code,200)
        exported=self.client.post('/api/v1/exports',json={'analysis_id':analysis_id,'include_activations':False},headers=self.headers).json()
        work(self.root,once=True);job=self.client.get(f"/api/v1/jobs/{exported['id']}").json();self.assertEqual(job['state'],'completed',job)
        export_id=job['result']['export_id'];self.assertEqual(self.client.get(f'/api/v1/exports/{export_id}').json()['status'],'completed')
        archive=self.client.get(f'/api/v1/exports/{export_id}/download');self.assertEqual(archive.status_code,200)
        imported=self.client.post('/api/v1/imports',content=archive.content,headers={**self.headers,'Content-Type':'application/zip'})
        self.assertEqual(imported.status_code,202,imported.text)
        work(self.root,once=True);job=self.client.get(f"/api/v1/jobs/{imported.json()['id']}").json();self.assertEqual(job['state'],'completed',job)
        metadata=self.client.get('/api/v1/imports/'+job['result']['import_id']).json();self.assertFalse(metadata['recomputed']);self.assertTrue(metadata['recorded_experiment'])
    def test_invalid_import_fails_job_and_no_partial_registration(self):
        imported=self.client.post('/api/v1/imports',content=b'not a zip',headers={**self.headers,'Content-Type':'application/zip'}).json()
        work(self.root,once=True);job=self.client.get(f"/api/v1/jobs/{imported['id']}").json()
        self.assertEqual(job['state'],'failed');self.assertEqual(job['error_code'],'INVALID_BUNDLE')
    def test_chunked_oversize_request_is_rejected_before_json_parsing(self):
        def chunks():
            yield b' ' * 3_000_000
            yield b' ' * 3_000_000
        response=self.client.post('/api/v1/experiments',content=chunks(),headers={**self.headers,'Content-Type':'application/json'})
        self.assertEqual(response.status_code,413,response.text)
        self.assertEqual(response.json()['error']['code'],'PAYLOAD_TOO_LARGE')
    def test_ratings_empty_state_and_balancing_metadata(self):
        response=self.client.get(f"/api/v1/experiments/{self.exp['id']}/rating-tasks?rater_id=tester").json()
        self.assertEqual(response['eligible_response_count'],0);self.assertEqual(response['items'],[])
        self.assertEqual(self.client.get(f"/api/v1/experiments/{self.exp['id']}/ratings").json()['items'],[])

    def test_openapi_typed_config_and_custom_port_same_origin(self):
        schema=self.client.get('/openapi.json').json()
        request_schema=schema['paths']['/api/v1/experiments']['post']['requestBody']['content']['application/json']['schema']
        self.assertEqual(request_schema['$ref'],'#/components/schemas/ExperimentConfigSchema')
        response=self.client.get('/api/v1/health',headers={'Host':'localhost:8001','Origin':'http://localhost:8001'})
        self.assertEqual(response.status_code,200,response.text)
        invalid=self.client.post('/api/v1/experiments',json={},headers={**self.headers,'Content-Length':'nonnumeric'})
        self.assertEqual(invalid.status_code,400)

    def test_holdout_analysis_requires_hash_bound_human_preregistration_receipts(self):
        exp=plan(self.app.state.store,load_config('configs/balanced.yaml'))
        queued=self.client.post(f"/api/v1/experiments/{exp['id']}/analyses",json={'split':'holdout'},headers=self.headers).json()
        work(self.root,once=True)
        job=self.client.get(f"/api/v1/jobs/{queued['id']}").json()
        self.assertEqual(job['state'],'failed');self.assertEqual(job['error_code'],'CONFLICT')

    def test_invalid_dataset_path_is_structured_422(self):
        config=load_config('configs/smoke.yaml');config['dataset']=str(self.root/'missing-dataset')
        response=self.client.post('/api/v1/experiments',json=config,headers=self.headers)
        self.assertEqual(response.status_code,422,response.text);self.assertEqual(response.json()['error']['code'],'INVALID_CONFIG')

    def test_ratings_strict_ordinal_eligibility_and_rater_bound_display(self):
        import copy
        from ghosttrace.config import canonical,digest,now
        store=self.app.state.store
        unit=next(u for u in self.exp['ordered_units'] if u['mode']=='greedy_generation')
        record={**unit,'experiment_id':self.exp['id'],'status':'completed','text':'TEST output','created_at':now()}
        def save(value):
            with store.connect() as db:db.execute('INSERT OR REPLACE INTO units VALUES(?,?,?,?,?,?,?)',(value['id'],self.exp['id'],canonical(value).decode(),value['status'],None,None,now()))
        save(record)
        body={'run_unit_id':unit['id'],'rater_id':'rater-A','blinded_display_id':digest({'experiment':self.exp['id'],'rater':'rater-A','unit':unit['id'],'version':'blind-v1'})[:20],'dimension_scores':{'regret':1,'responsibility':1,'repair':1,'justification_deflection':1}}
        for invalid in [True,1.0]:
            changed=copy.deepcopy(body);changed['dimension_scores']['regret']=invalid
            response=self.client.post('/api/v1/ratings',json=changed,headers=self.headers);self.assertEqual(response.status_code,422,response.text)
        response=self.client.post('/api/v1/ratings',json={**body,'blinded_display_id':'wrong-rater-task'},headers=self.headers);self.assertEqual(response.status_code,422)
        for invalid in [{**record,'mode':'fixed_continuation'},{**record,'status':'failed'},{**record,'control':'neutral_repeat'}]:
            save(invalid);response=self.client.post('/api/v1/ratings',json=body,headers=self.headers);self.assertEqual(response.status_code,422,response.text)
        save(record)
        response=self.client.post('/api/v1/ratings',json=body,headers=self.headers);self.assertEqual(response.status_code,201,response.text)
        repeated=self.client.post('/api/v1/ratings',json=body,headers=self.headers);self.assertEqual(response.json()['id'],repeated.json()['id'])
