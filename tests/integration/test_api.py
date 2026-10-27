import json
from pathlib import Path
import tempfile
import unittest
from fastapi.testclient import TestClient
from ghosttrace.api.app import create_app
from ghosttrace.config import load_config

class APIContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.app=create_app(self.temp.name);self.client=TestClient(self.app)
        self.token=self.client.get('/api/v1/health').json()['mutation_token']
        self.headers={'X-GhostTrace-Token':self.token}
    def tearDown(self):self.client.close();self.temp.cleanup()
    def test_health_does_not_load_model(self):
        value=self.client.get('/api/v1/health').json()
        self.assertFalse(value['model_loaded']);self.assertEqual(value['worker'],'unavailable')
    def test_localhost_and_origin(self):
        self.assertEqual(self.client.get('/api/v1/health',headers={'Host':'attacker.example'}).status_code,403)
        self.assertEqual(self.client.get('/api/v1/health',headers={'Origin':'https://attacker.example'}).status_code,403)
    def test_mutation_token(self):
        self.assertEqual(self.client.post('/api/v1/datasets/validate',json=[]).status_code,403)
    def test_invalid_preflight_and_pagination(self):
        r=self.client.post('/api/v1/preflights',json={'profile_id':'qwen25-05b-cpu','revision':'main'},headers=self.headers)
        self.assertEqual(r.status_code,422);self.assertEqual(r.json()['error']['code'],'INVALID_CONFIG')
        self.assertEqual(self.client.get('/api/v1/models?limit=101').status_code,422)
    def test_freeze_smoke_and_missing_model(self):
        config=load_config('configs/smoke.yaml')
        r=self.client.post('/api/v1/experiments',json=config,headers=self.headers)
        self.assertEqual(r.status_code,201,r.text)
        record=r.json();self.assertEqual(record['estimated_units'],24)
        same=self.client.post('/api/v1/experiments',json=config,headers=self.headers).json()
        self.assertEqual(record['id'],same['id'])
        queued=self.client.post(f"/api/v1/experiments/{record['id']}/jobs",json={},headers={**self.headers,'Idempotency-Key':'test'})
        self.assertEqual(queued.status_code,409) # profile is not registered; never substitute sample inference.
    def test_worker_job_lifecycle_and_events(self):
        r=self.client.post('/api/v1/preflights',json={'profile_id':'qwen25-05b-cpu','revision':'a'*40},headers=self.headers)
        self.assertEqual(r.status_code,202,r.text);job=r.json()['id']
        cancelled=self.client.post(f'/api/v1/jobs/{job}/cancel',headers=self.headers).json();self.assertEqual(cancelled['state'],'cancelled')
        resumed=self.client.post(f'/api/v1/jobs/{job}/resume',headers=self.headers).json()
        repeated=self.client.post(f'/api/v1/jobs/{job}/resume',headers=self.headers).json();self.assertEqual(resumed['id'],repeated['id'])
        events=self.client.get(f'/api/v1/jobs/{job}/events').json()['items'];self.assertEqual(events[0]['type'],'queued')
