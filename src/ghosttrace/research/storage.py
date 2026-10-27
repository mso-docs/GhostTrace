from __future__ import annotations
import json
import uuid
from ghosttrace.config import EngineError,canonical,digest,now

class ResearchRepository:
    def __init__(self,store):
        self.store=store
        with store.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS research_records(id TEXT PRIMARY KEY,kind TEXT NOT NULL,study_id TEXT,record TEXT NOT NULL,created_at TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS research_artifacts(record_id TEXT PRIMARY KEY REFERENCES research_records(id),relative_path TEXT NOT NULL,sha256 TEXT NOT NULL)')
            db.execute('CREATE INDEX IF NOT EXISTS research_by_study ON research_records(study_id,kind,created_at)')
    def put(self,kind,record,study_id=None):
        value=dict(record);value.setdefault('id',str(uuid.uuid4()));value.setdefault('schema_version','2.0');value.setdefault('created_at',now());value.setdefault('record_hash',digest({k:v for k,v in value.items() if k!='record_hash'}))
        with self.store.connect() as db:
            existing=db.execute('SELECT record FROM research_records WHERE id=?',(value['id'],)).fetchone()
            if existing:
                if json.loads(existing['record'])!=value:raise EngineError('CONFLICT','Research record is immutable')
                return value
            db.execute('INSERT INTO research_records VALUES(?,?,?,?,?)',(value['id'],kind,study_id,canonical(value).decode(),value['created_at']))
            artifact=value.get('artifact')
            if artifact and (self.store.root/artifact['relative_path']).is_file():
                path=(self.store.root/artifact['relative_path']).resolve()
                if not path.is_relative_to(self.store.root):raise EngineError('ARTIFACT_CORRUPT','Unsafe research artifact path')
                db.execute('INSERT INTO research_artifacts VALUES(?,?,?)',(value['id'],str(path.relative_to(self.store.root)),artifact['sha256']))
        return value
    def get(self,record_id,kind=None):
        with self.store.connect() as db:row=db.execute('SELECT * FROM research_records WHERE id=?',(record_id,)).fetchone()
        if not row or kind and row['kind']!=kind:raise EngineError('NOT_FOUND','Unknown research record')
        return json.loads(row['record'])
    def list(self,kind,study_id=None):
        with self.store.connect() as db:
            if study_id is None:rows=db.execute('SELECT record FROM research_records WHERE kind=? ORDER BY created_at,id',(kind,)).fetchall()
            else:rows=db.execute('SELECT record FROM research_records WHERE kind=? AND study_id=? ORDER BY created_at,id',(kind,study_id)).fetchall()
        return [json.loads(row['record']) for row in rows]

    def artifact_location(self,record):
        artifact=record.get('artifact')
        if not artifact:raise EngineError('UNSUPPORTED_CAPTURE','This research record has no raw arrays')
        with self.store.connect() as db:row=db.execute('SELECT * FROM research_artifacts WHERE record_id=?',(record['id'],)).fetchone()
        if not row:raise EngineError('UNSUPPORTED_CAPTURE','Raw arrays were omitted from this research bundle')
        path=(self.store.root/row['relative_path']).resolve()
        if not path.is_relative_to(self.store.root):raise EngineError('ARTIFACT_CORRUPT','Unsafe research artifact path')
        return path,row['sha256']
