#!/usr/bin/env python3
"""Bounded read-only verification; never extracts files or executes content."""
import argparse
import hashlib
import json
from pathlib import PurePosixPath
import zipfile

parser=argparse.ArgumentParser();parser.add_argument('archive');args=parser.parse_args()
with zipfile.ZipFile(args.archive) as archive:
    entries=archive.infolist()
    if len(entries)>10000 or sum(e.file_size for e in entries)>2_000_000_000:raise SystemExit('Bundle exceeds file/size limits')
    names=[e.filename for e in entries]
    if len(names)!=len(set(names)):raise SystemExit('Duplicate ZIP entries prohibited')
    for name in names:
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name:raise SystemExit('Unsafe ZIP path')
    inventory=json.loads(archive.read('inventory.json'))
    for name,expected in inventory['file_inventory_with_checksums'].items():
        if name not in names:raise SystemExit(f'Missing file: {name}')
        actual=hashlib.sha256(archive.read(name)).hexdigest()
        if actual!=expected:raise SystemExit(f'Checksum mismatch: {name}')
    unexpected=set(names)-set(inventory['file_inventory_with_checksums'])-{'inventory.json'}
    if unexpected:raise SystemExit(f'Uninventoried entries: {sorted(unexpected)}')
print('Verified: safe paths, bounded inventory and all SHA-256 checksums match. No model loaded.')
