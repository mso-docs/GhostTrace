#!/usr/bin/env python3
"""Prepare and validate an evidence-grounded agent review packet.

This tool does not invent a review or call an undisclosed model. An independent
agent/human uses the rubric and submits structured findings backed by local hashes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
CRITERIA = {
    'scientific_honesty':'No psychological/causal claims from descriptive activations; unrun and missing results explicit.',
    'alignment_and_capture':'Matched token IDs, residual hook site, FP32 masks, and normalization are demonstrable.',
    'recovery_and_integrity':'Cancellation/resume, immutable completed units and hash failures are tested.',
    'replay_and_export':'Offline read-only bundle preserves inventory/provenance and reproducible metrics.',
    'ratings_and_holdout':'Blinding/rater limits and reviewed frozen family-held-out protocol are explicit.',
    'accessibility_and_ui':'Keyboard navigation, responsive comparison, readable charts and text/table alternatives work.',
    'security_and_limits':'Localhost/origin restrictions, import bounds/path checks and no implicit model download.',
    'release_operability':'A clean install, reproducible validation commands and remaining human QA are documented.',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output, validation=None):
    files = ['README.md','LICENSE','THIRD_PARTY_NOTICES.md','.env.example','pyproject.toml','uv.lock','requirements-cpu.lock.txt','package.json','package-lock.json','playwright.config.ts','playwright.pages.config.ts','vite.config.ts','tsconfig.json','index.html']
    for folder in ('src','web/src','web/public','tests','scripts','configs','datasets','preregistrations','docs','.github'):
        files.extend(str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc',))
    files=sorted(set(files))
    if len(files)>2000 or sum((ROOT/p).stat().st_size for p in files if (ROOT/p).is_file())>100_000_000:raise ValueError('Review source inventory exceeds 2000 files / 100 MB')
    if any(Path(p).suffix in ('.safetensors','.pt','.pth','.bin') for p in files):raise ValueError('Model weights cannot be release/review source assets')
    if validation:
        validation = Path(validation).resolve()
        try:files.append(str(validation.relative_to(ROOT)))
        except ValueError:raise ValueError('Validation report must be inside the repository')
    packet = {'schema_version':'1.0','created_at':datetime.now(timezone.utc).isoformat(),'purpose':'Independent release review. Read evidence; challenge claims; never grade undocumented checks as passing.','criteria':CRITERIA,'evidence_inventory':[{'path':p,'sha256':sha(ROOT/p)} for p in files if (ROOT/p).is_file()],'review_output_schema':{'schema_version':'1.0','reviewer':{'name':'independent reviewer','kind':'agent or human','model':'required for agent','reviewed_at':'UTC timestamp'},'packet_sha256':'SHA-256 of this packet','criteria':{k:{'status':'pass | fail | blocked','reason':'specific reasoning','evidence':[{'path':'repository-relative file','sha256':'actual SHA-256','locator':'line number, test name or JSON pointer'}]} for k in CRITERIA},'findings':[{'severity':'critical | high | medium | low','status':'open | resolved','description':'concrete finding','criterion':'criterion id','evidence':[]}],'recommendation':'ship | human_review | do_not_ship'}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet,indent=2)+'\n')
    return packet


def validate(review, packet, *, root=ROOT):
    report = json.loads(Path(review).read_text())
    errors = []
    if not isinstance(report,dict):return {'schema_version':'1.0','status':'fail','errors':['Review must be a JSON object']}
    if report.get('schema_version') != '1.0':errors.append('Unsupported review schema')
    reviewer = report.get('reviewer', {})
    if not isinstance(reviewer,dict):reviewer={};errors.append('Reviewer must be an object')
    if reviewer.get('kind') not in ('agent','human') or not reviewer.get('name') or not reviewer.get('reviewed_at'):errors.append('Named reviewer, kind and UTC review time required')
    if reviewer.get('kind') == 'agent' and not reviewer.get('model'):errors.append('Agent review must name its model')
    if report.get('packet_sha256') != sha(Path(packet)):errors.append('Review packet checksum mismatch')
    packet_value=json.loads(Path(packet).read_text())
    inventory=packet_value.get('evidence_inventory',[]) if isinstance(packet_value,dict) else []
    if not isinstance(inventory,list):inventory=[];errors.append('Packet evidence inventory must be a list')
    if not inventory:errors.append('Review packet must freeze a nonempty evidence inventory')
    for item in inventory:
        if not isinstance(item,dict):errors.append('Invalid packet evidence item');continue
        path=(Path(root)/item.get('path','')).resolve()
        if not path.is_relative_to(Path(root).resolve()) or not path.is_file() or sha(path)!=item.get('sha256'):
            errors.append('Review source/evidence snapshot changed: '+item.get('path',''))
    criteria = report.get('criteria', {})
    if not isinstance(criteria,dict):criteria={};errors.append('Criteria must be an object')
    if set(criteria) != set(CRITERIA):errors.append('Every rubric criterion must be reviewed exactly once')
    for key, value in criteria.items():
        if not isinstance(value,dict):errors.append(f'{key}: criterion must be an object');continue
        if value.get('status') not in ('pass','fail','blocked') or not value.get('reason'):errors.append(f'{key}: explicit status and reason required')
        evidence = value.get('evidence', [])
        if not isinstance(evidence,list):evidence=[];errors.append(f'{key}: evidence must be a list')
        if value.get('status') == 'pass' and not evidence:errors.append(f'{key}: passing criteria require inspectable evidence')
        for item in evidence:
            if not isinstance(item,dict):errors.append(f'{key}: evidence item must be an object');continue
            path = (Path(root)/item.get('path','')).resolve()
            if not path.is_relative_to(Path(root).resolve()) or not path.is_file():errors.append(f'{key}: unsafe/missing evidence path');continue
            if sha(path) != item.get('sha256') or not item.get('locator'):errors.append(f'{key}: evidence checksum/locator invalid')
    findings = report.get('findings', [])
    if not isinstance(findings,list):errors.append('Findings must be a list');findings=[]
    for finding in findings:
        if not isinstance(finding,dict):errors.append('Finding must be an object');continue
        if finding.get('severity') not in ('critical','high','medium','low') or finding.get('status') not in ('open','resolved') or finding.get('criterion') not in CRITERIA or not finding.get('description'):
            errors.append('Every finding requires known severity/status/criterion and a concrete description')
    has_blocker = any(not isinstance(v,dict) or v.get('status') != 'pass' for v in criteria.values()) or any(not isinstance(v,dict) or v.get('severity') in ('critical','high') and v.get('status')!='resolved' for v in findings)
    if report.get('recommendation') not in ('ship','human_review','do_not_ship'):errors.append('Explicit recommendation required')
    if has_blocker and report.get('recommendation') == 'ship':errors.append('Ship recommendation conflicts with failed/blocked criteria or high findings')
    machine_blocker=any(not isinstance(v,dict) or v.get('status')=='fail' or v.get('status')=='blocked' and k!='ratings_and_holdout' for k,v in criteria.items()) or any(not isinstance(v,dict) or v.get('severity') in ('critical','high') and v.get('status')!='resolved' for v in findings) or report.get('recommendation')=='do_not_ship'
    return {'schema_version':'1.0','status':'fail' if errors else 'pending' if machine_blocker else 'pass','review_completed':not errors,'release_approved':not errors and not has_blocker and report.get('recommendation')=='ship','requires_human_review':report.get('recommendation')=='human_review' or isinstance(criteria.get('ratings_and_holdout'),dict) and criteria['ratings_and_holdout'].get('status')=='blocked','errors':errors,'recommendation':report.get('recommendation'),'reviewer':reviewer}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    prep = sub.add_parser('prepare');prep.add_argument('--output',type=Path,default=Path('artifacts/qa/agent-review-packet.json'));prep.add_argument('--validation-report',type=Path)
    val = sub.add_parser('validate');val.add_argument('--review',type=Path,required=True);val.add_argument('--packet',type=Path,required=True)
    runner = sub.add_parser('run');runner.add_argument('--packet',type=Path,required=True);runner.add_argument('--output',type=Path,required=True);runner.add_argument('--command-json',required=True,help='Explicit JSON argv array for a trusted configured reviewer; packet on stdin, review JSON on stdout. No shell.');runner.add_argument('--timeout',type=int,default=300)
    args = p.parse_args(argv)
    try:
        if args.command == 'prepare':result=prepare(args.output,args.validation_report)
        elif args.command == 'validate':result=validate(args.review,args.packet)
        else:
            command=json.loads(args.command_json)
            if not isinstance(command,list) or not command or not all(isinstance(v,str) and v for v in command):raise ValueError('--command-json must be a nonempty string argv array')
            if not 1<=args.timeout<=3600:raise ValueError('Timeout must be 1..3600 seconds')
            raw=args.packet.read_text()
            completed=subprocess.run(command,input=raw,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,cwd=ROOT,timeout=args.timeout,shell=False)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.with_suffix('.stderr.log').write_text(completed.stderr)
            if completed.returncode:raise ValueError(f'Agent reviewer exited {completed.returncode}; see stderr log')
            json.loads(completed.stdout) # Strict JSON only, not prose/code fences.
            args.output.write_text(completed.stdout)
            result=validate(args.output,args.packet)
    except (ValueError, OSError, TypeError, subprocess.TimeoutExpired) as e:
        print(json.dumps({'status':'fail','error':str(e)}));return 1
    print(json.dumps(result,indent=2))
    return 1 if result.get('status') == 'fail' else 0


if __name__ == '__main__':raise SystemExit(main())
