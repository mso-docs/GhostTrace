import threading
from ghosttrace.config import EngineError,now
from ghosttrace.research.service import run_captures,run_intervention

def execute_research(store,job):
    stop=threading.Event()
    def heartbeat():
        while not stop.wait(5):
            store.update_job(job['id'],heartbeat_at=now());store.worker_heartbeat('research_computation')
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        if store.job(job['id'])['cancel_requested']:store.update_job(job['id'],state='cancelled',stage='cancelled_before_research_operation');return
        kind=job['kind'];store.update_job(job['id'],stage=kind)
        if kind=='research_capture':result=run_captures(store,job)
        elif kind=='research_intervention':result=run_intervention(store,job)
        elif kind=='research_export':
            from ghosttrace.research.bundles import export_research
            record=export_research(store,job['payload']['study_id'],job['payload'].get('include_arrays',True));result={'resource_type':'research_export','resource_id':record['id'],'export_id':record['id'],'study_id':record['study_id']}
        elif kind=='research_import':
            from ghosttrace.research.bundles import import_research
            record=import_research(store,store.root/job['payload']['relative_path']);result={'resource_type':'research_import','resource_id':record['id'],'import_id':record['id'],'study_id':record['study_id']}
        elif kind=='research_dictionary_import':
            from ghosttrace.research.features import import_dictionary
            record=import_dictionary(store,store.root/job['payload']['relative_path']);result={'resource_type':'research_dictionary','resource_id':record['id'],'dictionary_id':record['id']}
        elif kind=='research_lens_import':
            from ghosttrace.research.lenses import import_lens
            record=import_lens(store,store.root/job['payload']['relative_path']);result={'resource_type':'research_lens','resource_id':record['id'],'lens_id':record['id']}
        elif kind=='research_lens_train':
            from ghosttrace.research.extensions import train_lens_package
            record=train_lens_package(store,store.root/job['payload']['relative_path']);result={'resource_type':'research_lens','resource_id':record['id'],'lens_id':record['id']}
        elif kind=='research_lens_apply':
            from ghosttrace.research.extensions import apply_lens_job
            record=apply_lens_job(store,job['payload']);result={'resource_type':'research_lens_result','resource_id':record['id'],'study_id':record['study_id'],'lens_result_id':record['id']}
        elif kind=='research_feature_intervention':
            from ghosttrace.research.extensions import intervene_features
            record=intervene_features(store,job['payload']);result={'resource_type':'research_feature_intervention','resource_id':record['id'],'study_id':record['study_id'],'feature_intervention_id':record['id']}
        elif kind=='research_feature_attribution':
            from ghosttrace.research.extensions import attribute_features
            record=attribute_features(store,job['payload']);result={'resource_type':'research_feature_attribution','resource_id':record['id'],'study_id':record['study_id'],'feature_attribution_id':record['id']}
        elif kind=='research_features':
            from ghosttrace.research.features import encode_features
            record=encode_features(store,job['payload']['study_id'],job['payload']['capture_id'],job['payload']['dictionary_id']);result={'resource_type':'research_features','resource_id':record['id'],'feature_id':record['id'],'study_id':record['study_id']}
        else:raise EngineError('INVALID_CONFIG','Unknown research job kind')
        if result is not None:store.update_job(job['id'],state='completed',stage='complete',completed_units=job['total_units'],result=result)
    except EngineError as error:store.update_job(job['id'],state='failed',stage='error',error_code=error.code,error_message=error.message)
    except Exception as error:store.update_job(job['id'],state='failed',stage='error',error_code='RESEARCH_JOB_FAILED',error_message=str(error))
    finally:stop.set();thread.join(timeout=1)
