from __future__ import annotations
import json
import os
import sqlite3
import uuid
import time
from contextlib import contextmanager
from pathlib import Path
from ghosttrace.config import EngineError, canonical, digest, now

class Store:
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get('GHOSTTRACE_DATA','artifacts')).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.db_path = self.root / 'ghosttrace.sqlite3'
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS migrations(version INTEGER PRIMARY KEY,applied_at TEXT NOT NULL);
            INSERT OR IGNORE INTO migrations VALUES(1,datetime('now'));
            CREATE TABLE IF NOT EXISTS experiments(id TEXT PRIMARY KEY,config_hash TEXT UNIQUE NOT NULL,record TEXT NOT NULL,created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,experiment_id TEXT REFERENCES experiments(id),kind TEXT NOT NULL DEFAULT 'experiment',attempt INTEGER NOT NULL,state TEXT NOT NULL,stage TEXT NOT NULL,total_units INTEGER NOT NULL,completed_units INTEGER NOT NULL DEFAULT 0,failed_units INTEGER NOT NULL DEFAULT 0,heartbeat_at TEXT,error_code TEXT,error_message TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,payload TEXT NOT NULL,cancel_requested INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS units(id TEXT PRIMARY KEY,experiment_id TEXT NOT NULL REFERENCES experiments(id),record TEXT NOT NULL,status TEXT NOT NULL,artifact_path TEXT,artifact_sha256 TEXT,updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS job_units(job_id TEXT REFERENCES jobs(id),unit_id TEXT REFERENCES units(id),PRIMARY KEY(job_id,unit_id));
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,job_id TEXT REFERENCES jobs(id),type TEXT NOT NULL,payload TEXT NOT NULL,created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS analyses(id TEXT PRIMARY KEY,experiment_id TEXT REFERENCES experiments(id),record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS ratings(id TEXT PRIMARY KEY,run_unit_id TEXT REFERENCES units(id),record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS exports(id TEXT PRIMARY KEY,record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS imports(id TEXT PRIMARY KEY,record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS idempotency(scope TEXT PRIMARY KEY,body_hash TEXT NOT NULL,response TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS profiles(id TEXT PRIMARY KEY,record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS worker_state(id INTEGER PRIMARY KEY CHECK(id=1),heartbeat_at TEXT NOT NULL,state TEXT NOT NULL);
            ''')
            db.execute('BEGIN IMMEDIATE')
            columns={r['name'] for r in db.execute('PRAGMA table_info(jobs)')}
            if 'result' not in columns:db.execute('ALTER TABLE jobs ADD COLUMN result TEXT')
            db.execute('INSERT OR IGNORE INTO migrations VALUES(2,?)',(now(),))

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.db_path,timeout=5)
        try:
            db.row_factory=sqlite3.Row
            db.execute('PRAGMA foreign_keys=ON')
            # API and worker may create the same brand-new database concurrently.
            # SQLite's journal-mode transition can return BUSY before honoring the
            # connection timeout, so retry only transient lock errors within 5 s.
            deadline=time.monotonic()+5
            while True:
                remaining=max(1,int((deadline-time.monotonic())*1000))
                db.execute(f'PRAGMA busy_timeout={min(5000,remaining)}')
                try:
                    db.execute('PRAGMA journal_mode=WAL')
                    break
                except sqlite3.OperationalError as error:
                    code=getattr(error,'sqlite_errorcode',0)&255
                    transient=code in (sqlite3.SQLITE_BUSY,sqlite3.SQLITE_LOCKED) or 'locked' in str(error).lower() or 'busy' in str(error).lower()
                    if not transient or time.monotonic()>=deadline:raise
                    time.sleep(min(.025,max(0,deadline-time.monotonic())))
            db.execute('PRAGMA busy_timeout=5000')
            yield db
            db.commit()
        except Exception:
            db.rollback(); raise
        finally:
            db.close()

    def event(self,db,job_id,event_type,payload):
        db.execute('INSERT INTO events(job_id,type,payload,created_at) VALUES(?,?,?,?)',(job_id,event_type,canonical(payload).decode(),now()))

    def experiment(self,experiment_id):
        with self.connect() as db:
            row=db.execute('SELECT record FROM experiments WHERE id=?',(experiment_id,)).fetchone()
        if not row: raise EngineError('NOT_FOUND','Unknown experiment')
        return json.loads(row['record'])

    def job(self,job_id):
        with self.connect() as db:
            row=db.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone()
        if not row: raise EngineError('NOT_FOUND','Unknown job')
        result=dict(row); result['payload']=json.loads(result['payload']); result['result']=json.loads(result['result']) if result.get('result') else None; result['recoverable']=result['state'] in ('failed','interrupted','cancelled')
        return result

    def queue(self,experiment_id,payload,total,kind='experiment',idempotency_key=None):
        scope=f'{kind}:{experiment_id}/jobs/{idempotency_key}' if idempotency_key else None
        body_hash=digest(payload)
        job_id=str(uuid.uuid4())
        timestamp=now()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if scope:
                old=db.execute('SELECT * FROM idempotency WHERE scope=?',(scope,)).fetchone()
                if old:
                    if old['body_hash']!=body_hash: raise EngineError('CONFLICT','Idempotency key already used for a different request')
                    return json.loads(old['response'])
            prior=db.execute('SELECT MAX(attempt) n FROM jobs WHERE experiment_id IS ?',(experiment_id,)).fetchone()['n'] or 0
            db.execute('INSERT INTO jobs(id,experiment_id,kind,attempt,state,stage,total_units,created_at,updated_at,payload) VALUES(?,?,?,?,?,?,?,?,?,?)',(job_id,experiment_id,kind,prior+1,'queued','waiting_for_worker',total,timestamp,timestamp,canonical(payload).decode()))
            response={'id':job_id,'job_id':job_id,'kind':kind,'state':'queued','total_units':total,'schema_version':'1.0'}
            self.event(db,job_id,'queued',response)
            if scope: db.execute('INSERT INTO idempotency VALUES(?,?,?)',(scope,body_hash,canonical(response).decode()))
        return response

    def claim(self):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute("SELECT id FROM jobs WHERE state='queued' ORDER BY created_at LIMIT 1").fetchone()
            if not row:return None
            db.execute("UPDATE jobs SET state='running',stage='loading_model',heartbeat_at=?,updated_at=? WHERE id=?",(now(),now(),row['id']))
            self.event(db,row['id'],'running',{})
        return self.job(row['id'])

    def update_job(self,job_id,**values):
        allowed={'state','stage','completed_units','failed_units','heartbeat_at','error_code','error_message','cancel_requested','result'}
        if set(values)-allowed: raise ValueError('Invalid job update')
        event_values=dict(values)
        if 'result' in values:values['result']=canonical(values['result']).decode()
        values['updated_at']=now()
        with self.connect() as db:
            db.execute('UPDATE jobs SET '+','.join(f'{k}=?' for k in values)+' WHERE id=?',(*values.values(),job_id))
            self.event(db,job_id,'progress',event_values)

    def interrupt_stale(self):
        with self.connect() as db:
            rows=db.execute("SELECT id FROM jobs WHERE state='running' AND (heartbeat_at IS NULL OR julianday('now')-julianday(heartbeat_at)>60.0/86400)").fetchall()
            for row in rows:
                db.execute("UPDATE jobs SET state='interrupted',stage='worker_lost',updated_at=? WHERE id=?",(now(),row['id']))
                self.event(db,row['id'],'interrupted',{'reason':'Worker heartbeat stale for more than 60 seconds'})
        return len(rows)

    def put_record(self,table,record,experiment_id=None):
        if table not in ('analyses','exports','profiles','ratings','imports'): raise ValueError('Invalid table')
        with self.connect() as db:
            if table=='analyses':db.execute('INSERT INTO analyses VALUES(?,?,?)',(record['id'],experiment_id,canonical(record).decode()))
            elif table=='ratings':db.execute('INSERT INTO ratings VALUES(?,?,?)',(record['id'],record['run_unit_id'],canonical(record).decode()))
            else:db.execute(f'INSERT OR REPLACE INTO {table} VALUES(?,?)',(record['id'],canonical(record).decode()))

    def get_record(self,table,record_id):
        if table not in ('analyses','exports','profiles','units','imports'): raise ValueError('Invalid table')
        with self.connect() as db:row=db.execute(f'SELECT record FROM {table} WHERE id=?',(record_id,)).fetchone()
        if not row:raise EngineError('NOT_FOUND',f'Unknown {table} record')
        return json.loads(row['record'])

    def unit_rows(self,experiment_id):
        with self.connect() as db:return [dict(x) for x in db.execute('SELECT * FROM units WHERE experiment_id=? ORDER BY id',(experiment_id,)).fetchall()]

    def worker_heartbeat(self,state='idle'):
        with self.connect() as db:db.execute('INSERT INTO worker_state VALUES(1,?,?) ON CONFLICT(id) DO UPDATE SET heartbeat_at=excluded.heartbeat_at,state=excluded.state',(now(),state))

    def activation_location(self,record):
        """Resolve local storage independently of portable, immutable run provenance."""
        artifact=record.get('activation_artifact')
        if not artifact:raise EngineError('UNSUPPORTED_CAPTURE','This unit has no activation capture')
        with self.connect() as db:
            row=db.execute('SELECT artifact_path,artifact_sha256 FROM units WHERE id=?',(record['id'],)).fetchone()
        if not row or not row['artifact_path']:raise EngineError('UNSUPPORTED_CAPTURE','Raw activations were omitted from this evidence bundle')
        path=(self.root/row['artifact_path']).resolve()
        if not path.is_relative_to(self.root):raise EngineError('ARTIFACT_CORRUPT','Unsafe local activation path')
        return path,row['artifact_sha256']
