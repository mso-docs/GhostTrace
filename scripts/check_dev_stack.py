#!/usr/bin/env python3
"""Exercise the real one-command stack in an isolated metadata directory.

Signals only the launcher process; verifies its owned API/worker/Vite children
close cleanly. Never queues a model or touches the user's experiment metadata.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]

def listening(port):
    with socket.socket() as connection:
        connection.settimeout(.2)
        return connection.connect_ex(('127.0.0.1',port))==0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('artifacts/qa/dev-stack-validation.json'));a=p.parse_args(argv)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result={'schema_version':'1.0','created_at':datetime.now(timezone.utc).isoformat(),'status':'fail','startup_passed':False,'shutdown_passed':False,'port_conflict_passed':False,'model_loaded':None,'data_directory_isolated':True}
    occupied=[port for port in (8000,5173) if listening(port)]
    if occupied:
        result['error']=f'Ports already in use: {occupied}. Stop your existing local stack before validation; it was not signaled or modified.'
        a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return 1
    with tempfile.TemporaryDirectory(prefix='ghosttrace-stack-check-') as temp:
        log=a.output.with_suffix('.log')
        with log.open('w') as output:
            process=subprocess.Popen(['node','scripts/dev.mjs'],cwd=ROOT,env={**os.environ,'GHOSTTRACE_DATA':temp,'PYTHONPATH':str(ROOT/'src')},stdout=output,stderr=subprocess.STDOUT)
            try:
                deadline=time.monotonic()+20
                while time.monotonic()<deadline and process.poll() is None:
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health',timeout=1) as response:health=json.load(response)
                        with urllib.request.urlopen('http://127.0.0.1:5173/',timeout=1) as response:web_status=response.status
                        if health.get('worker')=='idle':
                            result.update(startup_passed=health.get('api')=='healthy' and web_status==200 and health.get('model_loaded') is False,api_status=health.get('api'),worker_status=health.get('worker'),workbench_status=web_status,model_loaded=health.get('model_loaded'));break
                    except (OSError,urllib.error.URLError,ValueError):pass
                    time.sleep(.1)
                process.terminate() # The launcher, not a process-group shortcut.
                process.wait(timeout=5)
                deadline=time.monotonic()+5
                while time.monotonic()<deadline and any(listening(port) for port in (8000,5173)):time.sleep(.1)
                result['shutdown_passed']=not any(listening(port) for port in (8000,5173))
                result['launcher_exit_code']=process.returncode
                result['status']='pass' if result['startup_passed'] and result['shutdown_passed'] and process.returncode==0 else 'fail'
                if result['status']=='pass':
                    # Own a stand-in service but do not place it in the launcher's groups.
                    sentinel=subprocess.Popen([sys.executable,'-m','http.server','5173','--bind','127.0.0.1'],cwd=temp,stdout=output,stderr=subprocess.STDOUT)
                    conflict=None
                    try:
                        deadline=time.monotonic()+5
                        while time.monotonic()<deadline and not listening(5173):time.sleep(.05)
                        if not listening(5173):raise OSError('Owned conflict sentinel did not start')
                        conflict=subprocess.Popen(['node','scripts/dev.mjs'],cwd=ROOT,env={**os.environ,'GHOSTTRACE_DATA':str(Path(temp)/'conflict'),'PYTHONPATH':str(ROOT/'src')},stdout=output,stderr=subprocess.STDOUT)
                        conflict.wait(timeout=10)
                        deadline=time.monotonic()+5
                        while time.monotonic()<deadline and listening(8000):time.sleep(.05)
                        result['conflict_launcher_exit_code']=conflict.returncode
                        result['port_conflict_passed']=conflict.returncode!=0 and not listening(8000) and listening(5173) and sentinel.poll() is None
                        if not result['port_conflict_passed']:result['status']='fail'
                    finally:
                        if conflict and conflict.poll() is None:conflict.terminate();conflict.wait(timeout=5)
                        if sentinel.poll() is None:sentinel.terminate();sentinel.wait(timeout=5)
                result['evidence_log']=str(log)
            except (OSError,subprocess.TimeoutExpired) as error:
                result['error']=str(error);result['status']='fail'
            finally:
                if process.poll() is None:process.kill();process.wait(timeout=5)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return 0 if result['status']=='pass' else 1

if __name__=='__main__':raise SystemExit(main())
