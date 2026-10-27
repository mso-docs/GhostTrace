"""Gate regressions use explicitly synthetic report structure, never measured proof."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('mech_eval',Path(__file__).parents[2]/'scripts/evaluate_mechanistic.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class MechanisticGateContracts(unittest.TestCase):
    def test_empty_or_unrelated_junit_counts_never_attest_real_model(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle=Path(temp)/'bundle.json';bundle.write_text(json.dumps({'schema_version':'2.0'}));junit=Path(temp)/'tests.xml'
            for xml in ['<testsuite tests="5"/>','<testsuite><testcase classname="tests.unit.fake" name="test_pass"/></testsuite>']:
                junit.write_text(xml);report=module.evaluate(bundle,junit)
                self.assertEqual(report['status'],'failed');self.assertFalse(next(c['passed'] for c in report['checks'] if c['name']=='actual_model_testcases'))
