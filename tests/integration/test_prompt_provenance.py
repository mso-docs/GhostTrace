import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from ghosttrace.config import EngineError,load_config,digest
from ghosttrace.engine.planner import plan
from ghosttrace.storage.db import Store
from ghosttrace.storage.replay import _manifest

class PromptProvenance(unittest.TestCase):
    def test_label_only_instruction_change_creates_new_immutable_experiment_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dataset=root/'dataset';shutil.copytree('datasets/smoke-v1',dataset);shutil.copyfile('datasets/synthetic-v1/templates.json',dataset/'templates.json')
            config=load_config('configs/smoke.yaml');config['dataset']=str(dataset);store=Store(root/'data');first=plan(store,config)
            templates=json.loads((dataset/'templates.json').read_text());control=next(t for t in templates if t['family']=='control');control['system_text']+=' TEST revised instruction.';control['sha256']=digest({k:v for k,v in control.items() if k!='sha256'});(dataset/'templates.json').write_text(json.dumps(templates))
            second=plan(store,config)
            self.assertEqual(first['template_hashes'],second['template_hashes'])
            self.assertNotEqual(first['prompt_templates_hash'],second['prompt_templates_hash']);self.assertNotEqual(first['config_hash'],second['config_hash'])
            self.assertEqual(store.experiment(first['id']),first)
    def test_manifest_instruction_tampering_rejected_even_with_outer_checksum_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            exp=plan(Store(temp),load_config('configs/smoke.yaml'));tampered=copy.deepcopy(exp);next(t for t in tampered['prompt_templates'] if t['family']=='control')['system_text']='Tampered label-only condition'
            with self.assertRaises(EngineError) as caught:_manifest(tampered)
            self.assertEqual(caught.exception.code,'ARTIFACT_CORRUPT')
    def test_selected_family_hash_is_verified_directly_for_legacy_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            exp=plan(Store(temp),load_config('configs/smoke.yaml'));exp.pop('prompt_templates_hash');exp['prompt_templates'][0]['system_text']='Tampered selected family'
            with self.assertRaises(EngineError) as caught:_manifest(exp)
            self.assertEqual(caught.exception.code,'ARTIFACT_CORRUPT')

    def test_invalid_template_sha_duplicate_and_missing_ids_fail_before_freeze(self):
        for invalid in ['checksum','duplicate','missing']:
            with self.subTest(invalid=invalid),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);dataset=root/'dataset';shutil.copytree('datasets/smoke-v1',dataset)
                templates=json.loads(Path('datasets/synthetic-v1/templates.json').read_text())
                if invalid=='checksum':templates[0]['system_text']+=' Changed without a new content checksum.'
                elif invalid=='duplicate':templates[-1]=copy.deepcopy(templates[0])
                else:templates=[t for t in templates if t['id']!='label-only-neutral']
                (dataset/'templates.json').write_text(json.dumps(templates));config=load_config('configs/smoke.yaml');config['dataset']=str(dataset)
                with self.assertRaises(EngineError) as caught:plan(Store(root/'data'),config)
                self.assertEqual(caught.exception.code,'INVALID_CONFIG')
