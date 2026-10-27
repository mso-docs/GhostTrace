"""Portable evidence round trips use synthetic adapters strictly inside tests."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from ghosttrace.config import EngineError,load_config,canonical
from ghosttrace.storage.db import Store
from ghosttrace.storage.replay import import_bundle,verify_bundle,replay_experiment
from ghosttrace.engine.planner import plan
from ghosttrace.engine.runner import run_job
from ghosttrace.analysis.report import analyze,export_bundle
from ghosttrace.models.profiles import get_profile
from test_recovery import FixtureAdapter


def measured_fixture(root,templates=None):
    store=Store(root)
    profile={**get_profile('qwen25-05b-cpu'),'hidden_size':4,'revision':'a'*40,'tokenizer_revision':'a'*40,'preflight_passed':True,'device':'cpu','compute_dtype':'float32'}
    store.put_record('profiles',profile)
    config=load_config('configs/smoke.yaml');config['model_profile']=profile['id'];config['limits']['max_input_tokens']=2048;config['limits']['max_total_tokens']=2144
    if templates:config['templates']=templates
    exp=plan(store,config);store.queue(exp['id'],{},exp['estimated_units']);run_job(store,store.claim(),FixtureAdapter(store))
    analysis=analyze(store,exp['id'])
    return store,exp,analysis

def rewrite(source,target,transform):
    with zipfile.ZipFile(source) as z:files={n:z.read(n) for n in z.namelist()}
    transform(files)
    inventory=json.loads(files['inventory.json']);inventory['file_inventory_with_checksums']={n:hashlib.sha256(c).hexdigest() for n,c in files.items() if n!='inventory.json'};files['inventory.json']=canonical(inventory)
    with zipfile.ZipFile(target,'w') as z:
        for name,content in files.items():z.writestr(name,content)

class ReplayContracts(unittest.TestCase):
    def test_raw_export_import_metric_recompute_and_idempotency(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source,exp,analysis=measured_fixture(root/'source')
            exported=export_bundle(source,analysis['id'],True);archive=source.root/exported['relative_path']
            target=Store(root/'target');imported=import_bundle(target,archive)
            self.assertTrue(imported['recomputed']);self.assertEqual(imported['verification']['status'],'matched')
            self.assertEqual(imported['experiment_id'],exp['id']);self.assertEqual(target.experiment(exp['id']),exp)
            self.assertEqual(import_bundle(target,archive)['id'],imported['id'])
            replayed=replay_experiment(target,exp['id'],analysis['id']);self.assertTrue(replayed['recomputed'])
            self.assertFalse((target.root/'models').exists())
            self.assertEqual(target.get_record('units',source.unit_rows(exp['id'])[0]['id']),json.loads(source.unit_rows(exp['id'])[0]['record']))
    def test_secondary_behavioral_metrics_and_wording_heatmap(self):
        with tempfile.TemporaryDirectory() as temp:
            source,exp,analysis=measured_fixture(Path(temp)/'source',['A','B'])
            self.assertEqual(analysis['aggregates']['neutral-wording']['primary']['n'],3)
            self.assertEqual(analysis['aggregates']['neutral-wording']['metrics']['norm_ratio']['primary']['n'],3)
            self.assertEqual(analysis['aggregates']['neutral-wording']['metrics']['cosine_distance']['primary']['mean'],0)
            self.assertEqual(len([m for m in analysis['token_maps'] if m['contrast']=='neutral-wording']),3)
            behavior=analysis['behavioral']['conditions']['neutral']
            self.assertEqual(behavior['completed_units'],6);self.assertEqual(behavior['metrics']['generated_token_count']['n'],3)
            self.assertEqual(analysis['behavioral']['continuation_nll']['conditions']['neutral']['mean'],1)
            self.assertFalse(analysis['ratings']['reliability_available'])
    def test_summary_import_never_substitutes_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source,exp,analysis=measured_fixture(root/'source');exported=export_bundle(source,analysis['id'],False)
            target=Store(root/'target');result=import_bundle(target,source.root/exported['relative_path'])
            self.assertFalse(result['recomputed']);self.assertFalse(result['raw_activations_available'])
            replayed=replay_experiment(target,exp['id'],analysis['id']);self.assertEqual(replayed['verification']['status'],'summary_only')
            missing=analyze(target,exp['id']);self.assertIsNone(missing['aggregates']['remorseful-neutral']['primary']['mean'])
            self.assertTrue(any(x['reason']=='raw_activations_omitted' for x in missing['exclusions_with_reasons']))
    def test_metric_tampering_rejected_before_registration(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source,exp,analysis=measured_fixture(root/'source');exported=export_bundle(source,analysis['id'],True)
            def change(files):
                value=json.loads(files['analysis.json']);value['aggregates']['remorseful-neutral']['primary']['mean']=9;files['analysis.json']=canonical(value)
            bad=root/'tampered.zip';rewrite(source.root/exported['relative_path'],bad,change)
            target=Store(root/'target')
            with self.assertRaises(EngineError) as caught:import_bundle(target,bad)
            self.assertEqual(caught.exception.code,'REPLAY_MISMATCH')
            with target.connect() as db:self.assertEqual(db.execute('SELECT COUNT(*) FROM experiments').fetchone()[0],0)
    def test_unsafe_paths_and_checksum_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source,exp,analysis=measured_fixture(root/'source');exported=export_bundle(source,analysis['id'],False);archive=source.root/exported['relative_path']
            bad=root/'traversal.zip';rewrite(archive,bad,lambda files:files.update({'../escape':b'bad'}))
            with self.assertRaises(EngineError):verify_bundle(bad)
            with zipfile.ZipFile(archive) as z:files={n:z.read(n) for n in z.namelist()}
            files['report.md']=b'changed without checksum'
            with zipfile.ZipFile(root/'checksum.zip','w') as z:
                for name,value in files.items():z.writestr(name,value)
            with self.assertRaises(EngineError) as caught:verify_bundle(root/'checksum.zip')
            self.assertEqual(caught.exception.code,'ARTIFACT_CORRUPT')
