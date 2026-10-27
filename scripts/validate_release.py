#!/usr/bin/env python3
"""Run bounded, repeatable release gates and write machine-readable evidence.

CI mode requires automated contracts. Release mode additionally requires measured
bundle evidence, an independently reviewed packet and browser acceptance evidence.
Missing human/model gates are pending, never silently counted as passing.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import re
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def run_gate(name, command, output_dir, timeout=300):
    started = time.monotonic()
    log_path = output_dir/f'{name}.log'
    try:
        completed = subprocess.run(command, cwd=ROOT, env={**os.environ,'PYTHONPATH':str(ROOT/'src')}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        log_path.write_text(completed.stdout)
        return {'id':name,'status':'pass' if completed.returncode == 0 else 'fail','command':command,'exit_code':completed.returncode,'seconds':round(time.monotonic()-started,3),'evidence':str(log_path.relative_to(ROOT)) if log_path.is_relative_to(ROOT) else str(log_path),'sha256':hashlib.sha256(log_path.read_bytes()).hexdigest()}
    except (OSError, subprocess.TimeoutExpired) as e:
        log_path.write_text(str(e)+'\n')
        return {'id':name,'status':'fail','command':command,'error':str(e),'seconds':round(time.monotonic()-started,3),'evidence':str(log_path),'sha256':hashlib.sha256(log_path.read_bytes()).hexdigest()}


def gate_evidence(name, path, evaluator):
    if path is None:
        return {'id':name,'status':'pending','detail':'Not supplied. This gate is not assumed to pass.'}
    try:
        result = evaluator(path)
        return {'id':name,'status':result['status'],'evidence':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'evaluation':result}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError, ET.ParseError) as e:
        return {'id':name,'status':'fail','error':str(e),'evidence':str(path)}


REQUIRED_MODEL_TESTS={
 'test_real_alignment_across_conditions_and_templates',
 'test_real_capture_repeat_site_shape_and_hook_cleanup',
 'test_real_last_block_is_before_final_norm',
 'test_real_nll_uses_previous_position',
 'test_real_greedy_generation_repeat',
}
REQUIRED_BROWSER_TESTS={
 'production starts without invented measurements and local failures stay explicit',
 'keyboard reaches navigation and narrow layout keeps the main flow usable',
 'checksummed synthetic UI fixture replays offline, compares tokens and preserves zero/missing values',
 'corrupt bundle is rejected without replacing the current research',
 'unsafe archive paths are rejected',
 'bundled real smoke opens, carries provenance and remains usable offline',
 'production research views meet automated WCAG A/AA checks',
 'recorded comparison works on mobile and high contrast light meets WCAG checks',
 'a checksummed archive with the wrong declared capture site is rejected',
 'a rechecksummed inline model profile cannot escape its immutable profile hash',
}


def browser_evidence(path):
    data=json.loads(path.read_text());stats=data.get('stats',{});specs=[]
    def collect(suites):
        for suite in suites:
            specs.extend(suite.get('specs',[]));collect(suite.get('suites',[]))
    collect(data.get('suites',[]))
    ids=[spec.get('id') for spec in specs]
    titles={spec.get('title') for spec in specs if Path(spec.get('file','')).name=='workbench.spec.ts'}
    cases=[case for spec in specs for case in spec.get('tests',[])]
    valid_cases=bool(cases) and all(case.get('expectedStatus')=='passed' and case.get('status')=='expected' and len(case.get('results',[]))==1 and case['results'][0].get('status')=='passed' and case['results'][0].get('retry',0)==0 and not case['results'][0].get('errors') for case in cases)
    passed=REQUIRED_BROWSER_TESTS<=titles and len(ids)==len(set(ids)) and all(ids) and all(spec.get('ok') is True for spec in specs) and valid_cases and stats.get('expected')==len(cases) and all(stats.get(k,0)==0 for k in ('unexpected','flaky','skipped')) and not data.get('errors')
    return {'status':'pass' if passed else 'fail','stats':stats,'observed_cases':len(cases),'required_test_titles':sorted(REQUIRED_BROWSER_TESTS),'missing_test_titles':sorted(REQUIRED_BROWSER_TESTS-titles),'detail':'All required production acceptance specs must contain actual passed result records; counters alone, filtered runs, skips, retries and unrelated tests cannot pass.'}


def model_test_evidence(path):
    tree=ET.parse(path).getroot();suites=[tree] if tree.tag=='testsuite' else list(tree.iter('testsuite'))
    totals={key:sum(int(s.attrib.get(key,'0')) for s in suites) for key in ('tests','failures','errors','skipped')}
    cases=list(tree.iter('testcase'));ids=[(case.get('classname'),case.get('name')) for case in cases]
    names={case.get('name') for case in cases if case.get('classname')=='tests.real_model.test_qwen'}
    passed=len(cases)==len(REQUIRED_MODEL_TESTS) and totals['tests']==len(cases) and all(totals[k]==0 for k in ('failures','errors','skipped')) and len(ids)==len(set(ids)) and names==REQUIRED_MODEL_TESTS and all(not any(child.tag in ('failure','error','skipped') for child in case) for case in cases)
    return {'status':'pass' if passed else 'fail','counts':totals,'observed_cases':len(cases),'missing_test_names':sorted(REQUIRED_MODEL_TESTS-names),'detail':'The five exact real-model acceptance cases must execute under tests.real_model.test_qwen with unique IDs and no skipped/error/failure records. Empty/spoofed counters and unrelated test suites do not qualify.'}


def number(value,minimum=0):
    return type(value) in (int,float) and math.isfinite(value) and value>=minimum


def preflight_evidence(path):
    data=json.loads(path.read_text());generation=data.get('smoke_generations',[]);dtype=data.get('compute_dtype');maximum_tolerance=1e-5 if dtype=='float32' else 1e-3
    weight_hashes=data.get('weight_file_hashes',{})
    passed=data.get('preflight_passed') is True and data.get('reload_passed') is True and bool(re.fullmatch('[a-f0-9]{40}',str(data.get('revision','')))) and bool(re.fullmatch('[a-f0-9]{40}',str(data.get('tokenizer_revision','')))) and bool(weight_hashes) and all(re.fullmatch('[a-f0-9]{64}',str(v)) for v in weight_hashes.values()) and number(data.get('repeat_max_abs_error')) and number(data.get('repeat_tolerance')) and data['repeat_max_abs_error']<=data['repeat_tolerance']<=maximum_tolerance and number(data.get('generation_tokens_per_second'),1e-12) and number(data.get('peak_ram_bytes'),1) and len(generation)>=3
    passed=passed and all(isinstance(r.get('generated_ids'),list) and len(r['generated_ids'])==r.get('generated_token_count') and len(r['generated_ids'])>0 and isinstance(r.get('text'),str) and number(r.get('seconds'),1e-12) for r in generation)
    return {'status':'pass' if passed else 'fail','model':data.get('repo_id'),'revision':data.get('revision'),'device':data.get('device'),'dtype':dtype,'repeat_max_abs_error':data.get('repeat_max_abs_error'),'detail':'Immutable weight provenance, real nonempty timed generation records, memory, bounded repeat tolerance and reload are required; passing booleans alone do not qualify.'}


def protocol_evidence(path):
    data=json.loads(path.read_text());rows=data.get('results',[])
    passed=data.get('passed') is True and bool(re.fullmatch('[a-f0-9]{40}',str(data.get('tokenizer_revision','')))) and bool(re.fullmatch('[a-f0-9]{64}',str(data.get('dataset_hash','')))) and bool(re.fullmatch('[a-f0-9]{64}',str(data.get('templates_hash','')))) and data.get('scenarios_checked',0)>=42 and len(rows)==data.get('scenarios_checked') and len({r.get('scenario_id') for r in rows})==len(rows)
    for row in rows:
        ids=row.get('continuation_ids',[]);positions=row.get('positions',[])
        passed=passed and row.get('alignment_verified') is True and isinstance(ids,list) and len(ids)==row.get('continuation_token_count') and 16<=len(ids)<=32 and all(type(v) is int and v>=0 for v in ids) and len(positions)==6 and {(v.get('template'),v.get('condition')) for v in positions}=={(t,c) for t in ('A','B') for c in ('neutral','remorseful','unremorseful')} and all(type(v.get('prefix_token_count')) is int and v['prefix_token_count']>0 and type(v.get('total_tokens')) is int and v['prefix_token_count']+len(ids)==v['total_tokens']<=512 for v in positions)
    return {'status':'pass' if passed else 'fail','tokenizer_revision':data.get('tokenizer_revision'),'scenarios_checked':data.get('scenarios_checked'),'human_review_completed':data.get('human_review_completed',False),'detail':'All forty-two distinct scenarios/controls require actual sixteen-to-thirty-two token IDs and six condition/template position records under an immutable local tokenizer. Counters alone cannot pass. Human content review is separate.'}


def numeric_replay_evidence(path):
    data=json.loads(path.read_text())
    result=data.get('result',data)
    verification=result.get('verification',{})
    passed=result.get('recomputed') is True and result.get('raw_activations_available') is True and verification.get('status')=='matched' and verification.get('absolute_tolerance') is not None
    return {'status':'pass' if passed else 'fail','detail':'Raw-array import/recomputation must independently match recorded metrics within documented tolerance.','verification':verification}


def human_qa_evidence(path):
    data=json.loads(path.read_text())
    required={'dataset_templates','token_capture','recovery','keyboard_responsive','offline_replay','blinded_ratings','interpretation','release_inventory'}
    checks=data.get('checks',{})
    passed=data.get('schema_version')=='1.0' and data.get('reviewer',{}).get('kind')=='human' and bool(data.get('reviewer',{}).get('name')) and bool(data.get('reviewed_at')) and data.get('release_approved') is True and set(checks)==required and all(v.get('status')=='pass' and v.get('evidence') for v in checks.values())
    return {'status':'pass' if passed else 'pending','detail':'A named human must explicitly approve every documented QA gate with evidence. This checks the attestation structure, not the reviewer’s identity.'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['ci','release'],default='ci')
    p.add_argument('--output',type=Path,default=Path('artifacts/qa/release-validation.json'))
    p.add_argument('--python',default=sys.executable)
    p.add_argument('--skip-build',action='store_true',help='Development only; release mode forbids this.')
    p.add_argument('--browser',action='store_true',help='Run production Playwright acceptance tests.')
    p.add_argument('--browser-report',type=Path)
    p.add_argument('--bundle',type=Path,help='Real measured evidence ZIP; never a test fixture.')
    p.add_argument('--model-report',type=Path,help='JUnit XML from opt-in real-model tests.')
    p.add_argument('--preflight',type=Path,help='Measured preflight JSON from the selected real adapter.')
    p.add_argument('--replay-report',type=Path,help='Actual raw-array import/replay result JSON.')
    p.add_argument('--protocol-report',type=Path,help='Actual pinned-tokenizer scenario/template validation JSON.')
    p.add_argument('--agent-review',type=Path)
    p.add_argument('--human-qa',type=Path,help='Explicit named-human QA approval with evidence for each release checklist item.')
    p.add_argument('--review-packet',type=Path)
    args = p.parse_args(argv)
    if args.mode == 'release' and args.skip_build:p.error('--skip-build is not permitted for release mode')
    output = args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    log_dir=output.parent/(output.stem+'-logs');log_dir.mkdir(parents=True,exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    checks = [run_gate('unit_suite',[args.python,'-m','pytest','tests/unit','-q'],log_dir),run_gate('integration_suite',[args.python,'-m','pytest','tests/integration','-q'],log_dir),run_gate('evaluation_suite',[args.python,'-m','pytest','tests/evaluation','-q'],log_dir),run_gate('research_suite',[args.python,'-m','pytest','tests/research','tests/mechanistic','-q'],log_dir)]
    if shutil.which('uv'):checks.append(run_gate('python_lock',['uv','lock','--check','--offline','--python',args.python],log_dir))
    else:checks.append({'id':'python_lock','status':'fail','detail':'uv is required to check the frozen Python lock.'})
    checks.append(run_gate('local_stack',[args.python,'scripts/check_dev_stack.py','--output',str(log_dir/'dev-stack-result.json')],log_dir,timeout=40))
    checks.append(run_gate('help_assets',['node','scripts/validate_help_assets.mjs'],log_dir))
    checks.append(run_gate('api_contracts',[args.python,'scripts/generate_api_types.py','--check'],log_dir))
    if args.skip_build:checks.append({'id':'frontend_build','status':'pending','detail':'Build explicitly skipped in development.'})
    else:checks.append(run_gate('frontend_build',['npm','run','build'],log_dir))
    checks.append(run_gate('dependency_audit',['npm','audit',*(['--offline'] if args.mode=='ci' else []),'--audit-level=moderate'],log_dir))
    if args.browser:
        checks.append(run_gate('browser_suite',['npx','playwright','test'],log_dir,timeout=180))
        args.browser_report = Path('artifacts/qa/playwright-report.json')
    checks.append(gate_evidence('browser_evidence',args.browser_report,browser_evidence))
    checks.append(gate_evidence('real_model_tests',args.model_report,model_test_evidence))
    checks.append(gate_evidence('hardware_preflight',args.preflight,preflight_evidence))
    checks.append(gate_evidence('protocol_alignment',args.protocol_report,protocol_evidence))
    checks.append(gate_evidence('numeric_replay',args.replay_report,numeric_replay_evidence))
    checks.append(gate_evidence('measured_evidence',args.bundle,lambda path:load_script('evaluate_evidence').evaluate_bundle(path,require_measured=True)))
    if args.agent_review and args.review_packet:
        checks.append(gate_evidence('independent_agent_review',args.agent_review,lambda path:load_script('agent_review').validate(path,args.review_packet)))
    else:checks.append({'id':'independent_agent_review','status':'pending','detail':'Prepare an agent packet, obtain an independent review, and validate it. No self-awarded score.'})
    checks.append(gate_evidence('human_qa_approval',args.human_qa,human_qa_evidence))
    failed = any(c['status'] == 'fail' for c in checks)
    pending = any(c['status'] == 'pending' for c in checks)
    result = {'schema_version':'1.0','created_at':started,'completed_at':datetime.now(timezone.utc).isoformat(),'mode':args.mode,'status':'fail' if failed else 'pending' if pending else 'pass','release_ready':not failed and not pending,'checks':checks,'interpretation':'Passing automated contracts does not replace human scenario/template review, blinded ratings, accessibility review, or scientific interpretation. Pending gates block a release-ready verdict.'}
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
    return 1 if failed or args.mode == 'release' and pending else 0


if __name__ == '__main__':raise SystemExit(main())
