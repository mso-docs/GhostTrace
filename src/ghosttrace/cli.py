from __future__ import annotations
import argparse
import json
import sys
from ghosttrace.config import EngineError,load_config
from ghosttrace.storage.db import Store

def main(argv=None):
    parser=argparse.ArgumentParser(prog='ghosttrace',description='Local-first controlled model research. Downloads are always explicit.')
    parser.add_argument('--data-dir',default=None)
    sub=parser.add_subparsers(dest='command',required=True)
    doctor=sub.add_parser('doctor');doctor.add_argument('--profile',default='qwen25-05b-cpu');doctor.add_argument('--revision',required=True);doctor.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu');doctor.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='float32')
    validate=sub.add_parser('validate');validate.add_argument('--config',required=True)
    run=sub.add_parser('run');run.add_argument('--config',required=True);run.add_argument('--idempotency-key',default=None)
    resume=sub.add_parser('resume');resume.add_argument('--job',required=True)
    analysis=sub.add_parser('analyze');analysis.add_argument('--experiment',required=True);analysis.add_argument('--split',choices=['development','holdout','all'],default='development')
    export=sub.add_parser('export');export.add_argument('--analysis',required=True);export.add_argument('--include-activations',action='store_true')
    serve=sub.add_parser('serve');serve.add_argument('--host',choices=['127.0.0.1','localhost'],default='127.0.0.1');serve.add_argument('--port',type=int,default=8000)
    worker=sub.add_parser('worker');worker.add_argument('--once',action='store_true')
    importer=sub.add_parser('import');importer.add_argument('archive');importer.add_argument('--skip-metric-verification',action='store_true')
    replay=sub.add_parser('replay');replay.add_argument('--experiment',required=True);replay.add_argument('--analysis');replay.add_argument('--skip-metric-verification',action='store_true')
    from ghosttrace.research.cli import add_research_parser
    add_research_parser(sub)
    args=parser.parse_args(argv)
    try:
        store=Store(args.data_dir)
        if args.command=='research':
            from ghosttrace.research.cli import execute_research_cli
            result=execute_research_cli(store,args)
        elif args.command=='doctor':
            from ghosttrace.models.profiles import get_profile
            from ghosttrace.engine.runner import preflight
            profile={**get_profile(args.profile),'revision':args.revision,'tokenizer_revision':args.revision,'device':args.device,'compute_dtype':args.dtype}
            result=preflight(store,profile)
        elif args.command in ('validate','run'):
            from ghosttrace.engine.planner import plan
            exp=plan(store,load_config(args.config));result=exp
            if args.command=='run':
                profile=store.get_record('profiles',exp['config']['model_profile'])
                if not profile.get('preflight_passed'):raise EngineError('MODEL_NOT_READY','Run doctor before queueing inference')
                result=store.queue(exp['id'],{},exp['estimated_units'],idempotency_key=args.idempotency_key)
                result['next_step']='Run ghosttrace worker in a separate terminal.'
        elif args.command=='resume':
            j=store.job(args.job)
            if not j['recoverable']:raise EngineError('CONFLICT','Only interrupted, failed or cancelled jobs can resume')
            result=store.queue(j['experiment_id'],j['payload'],j['total_units'],j['kind'],idempotency_key=f'resume:{j["id"]}')
        elif args.command=='analyze':
            from ghosttrace.analysis.report import analyze
            result=analyze(store,args.experiment,args.split)
        elif args.command=='export':
            from ghosttrace.analysis.report import export_bundle
            result=export_bundle(store,args.analysis,args.include_activations)
        elif args.command=='import':
            from ghosttrace.storage.replay import import_bundle
            result=import_bundle(store,args.archive,not args.skip_metric_verification)
        elif args.command=='replay':
            from ghosttrace.storage.replay import replay_experiment
            result=replay_experiment(store,args.experiment,args.analysis,not args.skip_metric_verification)
        elif args.command=='serve':
            import uvicorn
            from ghosttrace.api.app import create_app
            uvicorn.run(create_app(args.data_dir),host=args.host,port=args.port);return
        elif args.command=='worker':
            from ghosttrace.worker.main import work
            work(args.data_dir,args.once);return
        print(json.dumps(result,indent=2,allow_nan=False))
    except EngineError as e:
        print(json.dumps({'error':{'code':e.code,'message':e.message,'details':e.details}}),file=sys.stderr);return 1
    except ImportError as e:
        print(json.dumps({'error':{'code':'DEPENDENCY_MISSING','message':str(e),'details':{'setup':'Use Python 3.11, install .[model] for live inference.'}}}),file=sys.stderr);return 1
    return 0

if __name__=='__main__':sys.exit(main())
