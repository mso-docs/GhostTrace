import json
from pathlib import Path
import shutil
import tempfile
import unittest
from ghosttrace.config import EngineError,read_dataset,load_config
from ghosttrace.engine.planner import plan
from ghosttrace.storage.db import Store

class LoaderErrors(unittest.TestCase):
    def test_missing_and_malformed_loaders_are_structured(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for loader,path in [(read_dataset,root/'missing.jsonl'),(load_config,root/'missing.yaml')]:
                with self.subTest(loader=loader):
                    with self.assertRaises(EngineError) as caught:loader(path)
                    self.assertEqual(caught.exception.code,'INVALID_CONFIG');self.assertIn('reason',caught.exception.details)
            (root/'bad.jsonl').write_text('{not json\n');(root/'bad.yaml').write_text('name: [unterminated')
            for loader,path in [(read_dataset,root/'bad.jsonl'),(load_config,root/'bad.yaml')]:
                with self.assertRaises(EngineError) as caught:loader(path)
                self.assertEqual(caught.exception.code,'INVALID_CONFIG')
    def test_malformed_template_and_receipt_are_structured(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dataset=root/'dataset';shutil.copytree('datasets/smoke-v1',dataset)
            config=load_config('configs/smoke.yaml');config['dataset']=str(dataset)
            (dataset/'templates.json').write_text('{broken')
            with self.assertRaises(EngineError) as caught:plan(Store(root/'first'),config)
            self.assertEqual(caught.exception.code,'INVALID_CONFIG')
            shutil.copyfile('datasets/synthetic-v1/templates.json',dataset/'templates.json')
            (dataset/'human-review.json').write_text('{broken')
            with self.assertRaises(EngineError) as caught:plan(Store(root/'second'),config)
            self.assertEqual(caught.exception.code,'INVALID_CONFIG')
