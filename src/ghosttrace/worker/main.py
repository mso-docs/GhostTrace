import time
from ghosttrace.storage.db import Store
from ghosttrace.engine.runner import run_job
from ghosttrace.worker.tasks import execute_task

def work(root=None,once=False):
    store=Store(root)
    while True:
        store.worker_heartbeat('idle')
        store.interrupt_stale()
        job=store.claim()
        if job:
            if job['kind'].startswith('research_'):
                from ghosttrace.research.worker import execute_research
                execute_research(store,job)
            elif job['kind'] in ('experiment','preflight'):run_job(store,job)
            else:execute_task(store,job)
        elif once:return
        else:time.sleep(1)
        if once:return
