import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from ghosttrace.config import canonical,load_config,now
from ghosttrace.storage.db import Store
from ghosttrace.engine.planner import plan
from ghosttrace.engine.runner import run_job
from ghosttrace.models.profiles import get_profile
from ghosttrace.analysis.report import analyze,export_bundle
from ghosttrace.storage.replay import import_bundle
from test_recovery import FixtureAdapter

class OneFailureAdapter(FixtureAdapter):
    def __init__(self,store):super().__init__(store);self.failed=False
    def generate(self,ids,settings,seed):
        if not self.failed:self.failed=True;raise RuntimeError('TEST first-attempt generation failure')
        return super().generate(ids,settings,seed)

class PortableHistory(unittest.TestCase):
    def test_failed_attempt_retained_after_resume_and_import_remains_inert(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);store=Store(root/'source')
            profile={**get_profile('qwen25-05b-cpu'),'hidden_size':4,'revision':'a'*40,'tokenizer_revision':'a'*40,'preflight_passed':True,'device':'cpu','compute_dtype':'float32'};store.put_record('profiles',profile)
            config=load_config('configs/smoke.yaml');config['model_profile']=profile['id'];config['limits']['max_input_tokens']=2048;config['limits']['max_total_tokens']=2144
            exp=plan(store,config);store.queue(exp['id'],{},exp['estimated_units']);first=store.claim();run_job(store,first,OneFailureAdapter(store));self.assertEqual(store.job(first['id'])['state'],'failed')
            store.queue(exp['id'],{},exp['estimated_units']);second=store.claim();run_job(store,second,FixtureAdapter(store));self.assertEqual(store.job(second['id'])['state'],'completed')
            analysis=analyze(store,exp['id']);exported=export_bundle(store,analysis['id'],False);archive=store.root/exported['relative_path']
            with zipfile.ZipFile(archive) as z:
                jobs=[json.loads(line) for line in z.read('jobs.jsonl').splitlines() if line]
                self.assertEqual({j['state'] for j in jobs},{'failed','completed'})
                attempts=[json.loads(z.read(name)) for name in z.namelist() if name.startswith('attempts/')]
                failed=[r for r in attempts if r['status']=='failed'];self.assertEqual(len(failed),1);self.assertIn('TEST first-attempt',failed[0]['error']['message'])
                events=[json.loads(line) for line in z.read('events.jsonl').splitlines() if line]
                failure=next(e for e in events if e['type']=='unit_failed');self.assertEqual(failure['payload']['error']['code'],'UNIT_FAILED')
            target=Store(root/'target');imported=import_bundle(target,archive)
            self.assertTrue(imported['execution_history']['inert'])
            with target.connect() as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0],0);self.assertEqual(db.execute('SELECT COUNT(*) FROM events').fetchone()[0],0)
            # Re-export must retain historical failed attempts even without active local jobs.
            exported_again=export_bundle(target,analysis['id'],False)
            with zipfile.ZipFile(target.root/exported_again['relative_path']) as z:self.assertTrue(any(name.startswith('attempts/') for name in z.namelist()))
