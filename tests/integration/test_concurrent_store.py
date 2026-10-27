"""Real OS processes exercise the first-launch API/worker SQLite race."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from ghosttrace.storage.db import Store

CHILD='''
import sys,time
from pathlib import Path
from ghosttrace.storage.db import Store
root=Path(sys.argv[1]);number=sys.argv[2]
(root/('ready-'+number)).touch()
while not (root/'go').exists():time.sleep(.001)
store=Store(root/'metadata')
store.queue(None,{'process':number},1,'preflight')
with store.connect() as db:
    assert db.execute('PRAGMA journal_mode').fetchone()[0]=='wal'
    assert db.execute('PRAGMA foreign_keys').fetchone()[0]==1
    assert db.execute('PRAGMA busy_timeout').fetchone()[0]==5000
'''

class ConcurrentStore(unittest.TestCase):
    def test_simultaneous_new_database_initialization_and_migrations(self):
        for repetition in range(4):
            with self.subTest(repetition=repetition),tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                children=[subprocess.Popen([sys.executable,'-c',CHILD,temp,str(index)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for index in range(4)]
                try:
                    deadline=time.monotonic()+5
                    while len(list(root.glob('ready-*')))<4 and time.monotonic()<deadline:time.sleep(.005)
                    self.assertEqual(len(list(root.glob('ready-*'))),4)
                    (root/'go').touch()
                    for child in children:
                        stdout,stderr=child.communicate(timeout=12)
                        self.assertEqual(child.returncode,0,stderr)
                    store=Store(root/'metadata')
                    with store.connect() as db:
                        self.assertEqual(db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0],4)
                        self.assertEqual(db.execute('SELECT COUNT(*) FROM migrations WHERE version=2').fetchone()[0],1)
                        self.assertIn('result',{r['name'] for r in db.execute('PRAGMA table_info(jobs)')})
                finally:
                    for child in children:
                        if child.poll() is None:child.kill();child.communicate()
