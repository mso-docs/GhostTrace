"""An actual killed worker process must preserve completed immutable units.

The child uses a synthetic adapter; it cannot create measured demo evidence.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from ghosttrace.config import load_config
from ghosttrace.storage.db import Store
from ghosttrace.engine.planner import plan
from ghosttrace.engine.runner import run_job
from ghosttrace.models.profiles import get_profile
from test_recovery import FixtureAdapter

ROOT=Path(__file__).parents[2]
CHILD='''
import sys,time
from ghosttrace.storage.db import Store
from ghosttrace.engine.runner import run_job
from test_recovery import FixtureAdapter
class SlowAfterFirst(FixtureAdapter):
 def before(self):
  if self.calls>=1:time.sleep(60)
 def capture(self,*args,**kwargs):self.before();return super().capture(*args,**kwargs)
 def generate(self,*args,**kwargs):self.before();return super().generate(*args,**kwargs)
store=Store(sys.argv[1]);job=store.claim();run_job(store,job,SlowAfterFirst(store))
'''


class ProcessRecovery(unittest.TestCase):
    def test_sigkill_preserves_committed_unit_and_resume_skips_it(self):
        with tempfile.TemporaryDirectory() as temp:
            store=Store(temp)
            profile={**get_profile('qwen25-05b-cpu'),'revision':'a'*40,'tokenizer_revision':'a'*40,'preflight_passed':True,'device':'cpu','compute_dtype':'float32'}
            store.put_record('profiles',profile)
            config=load_config('configs/smoke.yaml');config['model_profile']=profile['id'];config['limits']['max_input_tokens']=2048;config['limits']['max_total_tokens']=2144
            experiment=plan(store,config)
            first=store.queue(experiment['id'],{},experiment['estimated_units'])
            environment={**os.environ,'PYTHONPATH':os.pathsep.join([str(ROOT/'src'),str(ROOT/'tests/integration')])}
            process=subprocess.Popen([sys.executable,'-c',CHILD,temp],cwd=ROOT,env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                deadline=time.monotonic()+10
                rows=[]
                while time.monotonic()<deadline:
                    rows=[r for r in store.unit_rows(experiment['id']) if r['status']=='completed']
                    if rows:break
                    if process.poll() is not None:
                        out,err=process.communicate();self.fail(f'Child ended before a unit committed: {out} {err}')
                    time.sleep(.02)
                self.assertEqual(len(rows),1,'The child must commit exactly one unit before blocking.')
                original={r['id']:r['record'] for r in rows}
                process.kill();process.communicate(timeout=5)
                self.assertNotEqual(process.returncode,0)
                store.update_job(first['id'],heartbeat_at='2000-01-01T00:00:00+00:00')
                self.assertEqual(store.interrupt_stale(),1)
                self.assertEqual(store.job(first['id'])['state'],'interrupted')
                resumed=store.queue(experiment['id'],{},experiment['estimated_units'],idempotency_key='after-real-process-death')
                adapter=FixtureAdapter(store);run_job(store,store.claim(),adapter)
                self.assertEqual(store.job(resumed['id'])['state'],'completed')
                self.assertEqual(adapter.calls,experiment['estimated_units']-1)
                for unit in store.unit_rows(experiment['id']):
                    if unit['id'] in original:self.assertEqual(unit['record'],original[unit['id']])
            finally:
                if process.poll() is None:process.kill();process.communicate(timeout=5)
