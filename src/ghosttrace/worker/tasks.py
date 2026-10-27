"""Worker dispatch for bounded non-model evidence operations."""
from __future__ import annotations
import threading
from ghosttrace.config import EngineError,now
from ghosttrace.analysis.report import analyze,export_bundle

def execute_task(store,job):
    stop=threading.Event()
    def beat():
        while not stop.wait(5):
            store.update_job(job['id'],heartbeat_at=now())
            store.worker_heartbeat('processing_evidence')
    thread=threading.Thread(target=beat,daemon=True);thread.start()
    try:
        if store.job(job['id'])['cancel_requested']:
            store.update_job(job['id'],state='cancelled',stage='cancelled_before_operation');return
        kind=job['kind'];payload=job['payload']
        store.update_job(job['id'],stage={'analysis':'analyzing','export':'packaging','import':'verifying_import','replay':'recomputing_replay'}.get(kind,'processing'))
        if kind=='analysis':
            value=analyze(store,job['experiment_id'],payload.get('split','development'))
            result={'resource_type':'analysis','resource_id':value['id'],'analysis_id':value['id']}
        elif kind=='export':
            value=export_bundle(store,payload['analysis_id'],payload.get('include_activations',False))
            result={'resource_type':'export','resource_id':value['id'],'export_id':value['id']}
        elif kind=='import':
            from ghosttrace.storage.replay import import_bundle
            value=import_bundle(store,store.root/payload['relative_path'],verify_metrics=payload.get('verify_metrics',True))
            result={'resource_type':'import','resource_id':value['id'],'import_id':value['id'],'experiment_id':value['experiment_id'],'analysis_id':value['analysis_id'],'raw_activations_available':value['raw_activations_available'],'recomputed':value['recomputed']}
        elif kind=='replay':
            from ghosttrace.storage.replay import replay_experiment
            value=replay_experiment(store,job['experiment_id'],payload.get('analysis_id'),payload.get('verify_metrics',True))
            result={'resource_type':'replay','resource_id':value['analysis_id'],**value}
        else:raise EngineError('INVALID_CONFIG','Unknown evidence task kind')
        store.update_job(job['id'],state='completed',stage='complete',completed_units=1,result=result)
    except EngineError as e:
        store.update_job(job['id'],state='failed',stage='error',error_code=e.code,error_message=e.message)
    except Exception as e:
        store.update_job(job['id'],state='failed',stage='error',error_code='EVIDENCE_TASK_FAILED',error_message=str(e))
    finally:
        stop.set();thread.join(timeout=1)
