"""The gate runner must not convert skipped/absent evidence into release success."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('validate_release',Path(__file__).parents[2]/'scripts/validate_release.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class ReleaseGateContracts(unittest.TestCase):
    def test_browser_zero_skipped_or_counter_only_reports_are_not_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'browser.json'
            for stats in [{'expected':0},{'expected':5,'skipped':1},{'expected':4,'unexpected':1},{'expected':4,'flaky':1},{'expected':9,'skipped':0,'unexpected':0,'flaky':0}]:
                path.write_text(json.dumps({'stats':stats,'errors':[]}))
                self.assertEqual(runner.browser_evidence(path)['status'],'fail')
            specs=[{'title':title,'file':'workbench.spec.ts','id':str(i),'ok':True,'tests':[{'expectedStatus':'passed','status':'expected','results':[{'status':'passed','retry':0,'errors':[]}]}]} for i,title in enumerate(sorted(runner.REQUIRED_BROWSER_TESTS))]
            report={'suites':[{'specs':specs}],'stats':{'expected':len(specs),'skipped':0,'unexpected':0,'flaky':0},'errors':[]}
            path.write_text(json.dumps(report));self.assertEqual(runner.browser_evidence(path)['status'],'pass')
            specs[0]['tests'][0]['results']=[];path.write_text(json.dumps(report));self.assertEqual(runner.browser_evidence(path)['status'],'fail')

    def test_junit_requires_exact_real_executed_model_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.xml'
            for document in ['<testsuites><testsuite tests="5" skipped="5" errors="0" failures="0"/></testsuites>','<testsuites><testsuite tests="5" skipped="0" errors="0" failures="0"/></testsuites>']:
                path.write_text(document);self.assertEqual(runner.model_test_evidence(path)['status'],'fail')
            cases=''.join(f'<testcase classname="tests.real_model.test_qwen" name="{name}"/>' for name in sorted(runner.REQUIRED_MODEL_TESTS))
            valid='<testsuites><testsuite tests="5" skipped="0" errors="0" failures="0">'+cases+'</testsuite></testsuites>'
            path.write_text(valid);self.assertEqual(runner.model_test_evidence(path)['status'],'pass')
            path.write_text(valid.replace('tests.real_model.test_qwen','tests.unit.unrelated'));self.assertEqual(runner.model_test_evidence(path)['status'],'fail')
            path.write_text(valid.replace('<testcase','<testcase',1).replace(sorted(runner.REQUIRED_MODEL_TESTS)[1],sorted(runner.REQUIRED_MODEL_TESTS)[0]))
            self.assertEqual(runner.model_test_evidence(path)['status'],'fail')

    def test_preflight_and_protocol_booleans_cannot_replace_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'fake.json'
            path.write_text(json.dumps({'preflight_passed':True,'reload_passed':True,'repeat_max_abs_error':0,'repeat_tolerance':1e-5,'generation_tokens_per_second':100,'smoke_generations':[{}, {}, {}]}))
            self.assertEqual(runner.preflight_evidence(path)['status'],'fail')
            path.write_text(json.dumps({'passed':True,'scenarios_checked':42,'results':[{'scenario_id':str(i),'alignment_verified':True,'continuation_token_count':16} for i in range(42)]}))
            self.assertEqual(runner.protocol_evidence(path)['status'],'fail')

    def test_missing_evidence_is_pending(self):
        self.assertEqual(runner.gate_evidence('missing',None,lambda p:None)['status'],'pending')

    def test_subprocess_failure_has_inspectable_log_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            result=runner.run_gate('deliberate_failure',[sys.executable,'-c','print("intentional test failure");raise SystemExit(2)'],Path(temp))
            self.assertEqual(result['status'],'fail');self.assertEqual(result['exit_code'],2)
            self.assertEqual(len(result['sha256']),64)
            self.assertIn('intentional test failure',(Path(temp)/'deliberate_failure.log').read_text())
