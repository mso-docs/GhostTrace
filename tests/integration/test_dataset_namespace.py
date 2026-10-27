import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile
from ghosttrace.config import EngineError,load_config,digest,canonical
from ghosttrace.storage.db import Store
from ghosttrace.engine.planner import plan
from ghosttrace.analysis.report import analyze,export_bundle
from ghosttrace.storage.replay import verify_bundle,import_bundle

class DatasetNamespace(unittest.TestCase):
    def test_custom_main_control_id_collision_rejected_before_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dataset=root/'dataset';shutil.copytree('datasets/synthetic-v1',dataset)
            scenarios=[json.loads(line) for line in (dataset/'scenarios.jsonl').read_text().splitlines()]
            controls=[json.loads(line) for line in (dataset/'factual-controls.jsonl').read_text().splitlines()]
            controls[0]['id']=scenarios[0]['id'];controls[0]['sha256']=digest({k:v for k,v in controls[0].items() if k!='sha256'})
            (dataset/'factual-controls.jsonl').write_text('\n'.join(json.dumps(r) for r in controls)+'\n')
            config=load_config('configs/balanced.yaml');config['dataset']=str(dataset);store=Store(root/'data')
            with self.assertRaises(EngineError) as caught:plan(store,config)
            self.assertEqual(caught.exception.code,'INVALID_CONFIG')
            with store.connect() as db:self.assertEqual(db.execute('SELECT COUNT(*) FROM experiments').fetchone()[0],0)
    def test_custom_cross_namespace_family_split_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dataset=root/'dataset';shutil.copytree('datasets/synthetic-v1',dataset)
            scenarios=[json.loads(line) for line in (dataset/'scenarios.jsonl').read_text().splitlines()]
            controls=[json.loads(line) for line in (dataset/'factual-controls.jsonl').read_text().splitlines()]
            controls[0]['family_id']=next(r['family_id'] for r in scenarios if r['split']=='holdout');controls[0]['sha256']=digest({k:v for k,v in controls[0].items() if k!='sha256'})
            (dataset/'factual-controls.jsonl').write_text('\n'.join(json.dumps(r) for r in controls)+'\n')
            config=load_config('configs/balanced.yaml');config['dataset']=str(dataset)
            with self.assertRaises(EngineError) as caught:plan(Store(root/'data'),config)
            self.assertEqual(caught.exception.code,'INVALID_CONFIG')
    def test_untrusted_archive_cross_namespace_collision_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=Store(root/'source');exp=plan(source,load_config('configs/smoke.yaml'));analysis=analyze(source,exp['id']);exported=export_bundle(source,analysis['id'])
            with zipfile.ZipFile(source.root/exported['relative_path']) as archive:files={n:archive.read(n) for n in archive.namelist()}
            manifest=json.loads(files['manifest.json']);manifest['control_scenarios']=[dict(manifest['scenarios'][0])]
            # Re-sign every outer checksum: semantic validation must still reject the overlap.
            files['manifest.json']=canonical(manifest);inventory=json.loads(files['inventory.json']);inventory['file_inventory_with_checksums']={n:hashlib.sha256(v).hexdigest() for n,v in files.items() if n!='inventory.json'};files['inventory.json']=canonical(inventory)
            malicious=root/'collision.zip'
            with zipfile.ZipFile(malicious,'w') as archive:
                for name,value in files.items():archive.writestr(name,value)
            target=Store(root/'target')
            with self.assertRaises(EngineError) as caught:import_bundle(target,malicious)
            self.assertEqual(caught.exception.code,'INVALID_CONFIG')
            with target.connect() as db:self.assertEqual(db.execute('SELECT COUNT(*) FROM experiments').fetchone()[0],0)
