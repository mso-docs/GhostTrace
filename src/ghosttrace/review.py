"""Hash-bound human review packets. AI checks never substitute human attestation."""
from __future__ import annotations
import json
import re
from pathlib import Path
from ghosttrace.config import EngineError, canonical, digest, file_digest, now, read_dataset
from ghosttrace.storage.artifacts import atomic_json, atomic_bytes


def prepare(source):
    source = Path(source)
    rows = read_dataset(source / 'scenarios.jsonl')
    controls = read_dataset(source / 'factual-controls.jsonl') if (source / 'factual-controls.jsonl').exists() else []
    templates = json.loads((source / 'templates.json').read_text(encoding='utf-8'))
    entries = []
    for kind, records in [('scenario', rows), ('factual_control', controls), ('template', templates)]:
        for row in records:
            entries.append({'kind': kind, 'id': row['id'], 'content_hash': digest(row), 'content': row,
                            'decision': 'pending', 'notes': ''})
    return {'schema_version': '1.0', 'created_at': now(), 'source_hash': digest({'scenarios': rows, 'controls': controls, 'templates': templates}),
            'reviewer': '', 'attestation': '', 'entries': entries,
            'instructions': 'A human must review every scenario, control and template for clarity, synthetic content, balanced wording, and family split. Set decision to approve or needs_revision. Do not inspect holdout results. Tokenizer validation remains a separate gate.'}


def apply(source, decisions, output, reviewer, attest_human=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not attest_human or not reviewer or not reviewer.strip():
        raise EngineError('INVALID_CONFIG', 'An explicit human attestation and nonempty reviewer identity are required')
    if output.exists() or source == output:
        raise EngineError('CONFLICT', 'Review creates a new, previously nonexistent dataset directory; source files are never overwritten')
    packet = prepare(source)
    if decisions.get('source_hash') != packet['source_hash']:
        raise EngineError('CONFLICT', 'Review source hash changed; prepare and review a new packet')
    supplied = decisions.get('entries', [])
    keys = [(e.get('kind'), e.get('id')) for e in supplied]
    expected = {(e['kind'], e['id']): e for e in packet['entries']}
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise EngineError('INVALID_CONFIG', 'Review must cover every scenario, control and template exactly once')
    reviewed = {'scenario': [], 'factual_control': [], 'template': []}
    reviewed_at = now()
    for entry in supplied:
        original = expected[(entry['kind'], entry['id'])]
        if entry.get('content_hash') != original['content_hash']:
            raise EngineError('CONFLICT', f'Reviewed content hash mismatch: {entry["id"]}')
        if entry.get('decision') != 'approve':
            raise EngineError('INVALID_CONFIG', f'Unapproved item: {entry["id"]}; revise a new dataset version and repeat review')
        # Always use the original source record, never arbitrary content supplied by a review file.
        row = {**original['content'], 'review_status': 'reviewed'}
        row['sha256'] = digest({k:v for k,v in row.items() if k != 'sha256'})
        reviewed[entry['kind']].append(row)
    output.mkdir(parents=True)
    for kind, name in [('scenario', 'scenarios.jsonl'), ('factual_control', 'factual-controls.jsonl')]:
        if reviewed[kind]: atomic_bytes(output/name, b'\n'.join(canonical(r) for r in reviewed[kind])+b'\n')
    atomic_json(output/'templates.json', reviewed['template'])
    for name in ['LICENSE', 'LICENSE.txt', 'README.md']:
        if (source/name).is_file(): atomic_bytes(output/name, (source/name).read_bytes())
    review = {'schema_version':'1.0','created_at':reviewed_at,'reviewer':reviewer.strip(),
              'attestation':'The named human reviewed every listed item before confirmatory results were accessed.',
              'source_hash':packet['source_hash'],'decisions_hash':digest(decisions),
              'reviewed_dataset_hash':digest(reviewed['scenario']),
              'reviewed_templates_hash':digest(reviewed['template']),
              'reviewed_controls_hash':digest(reviewed['factual_control']),
              'items':[{'kind':e['kind'],'id':e['id'],'content_hash':e['content_hash'],'decision':e['decision'],'notes':e.get('notes','')} for e in supplied]}
    atomic_json(output/'human-review.json',review)
    manifest={'schema_version':'1.0','created_at':reviewed_at,'version':output.name,'review_status':'reviewed',
              'confirmatory_use_allowed':False,'pending_gates':['pinned tokenizer validation','successful deterministic model preflight','frozen preregistration'],
              'scenario_count':len(reviewed['scenario']),'dataset_hash':review['reviewed_dataset_hash'],
              'review_hash':digest(review),'files':{p.name:file_digest(p) for p in output.iterdir() if p.is_file()}}
    atomic_json(output/'manifest.json',manifest)
    return manifest


def render_html(packet):
    """Self-contained, offline review UI. Approval always requires individual human actions."""
    payload=json.dumps(packet,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GhostTrace · Human protocol review</title><style>
    :root{font:15px/1.7 system-ui;color:#d6e4db;background:#111a15}*{box-sizing:border-box}body{max-width:950px;margin:35px auto;padding:0 22px}h1{line-height:1.2;letter-spacing:-1px}p,small{color:#a1b6a9}button,input,select,textarea{font:inherit;color:inherit;background:#1b2d21;border:1px solid #496751;border-radius:5px;padding:9px 12px}button{cursor:pointer}button:focus-visible,input:focus-visible,textarea:focus-visible{outline:3px solid #a1edbf;outline-offset:3px}.bar{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:20px 0}.card{padding:22px;background:#1a271f;border:1px solid #3c5744;border-radius:9px;margin:18px 0}.card p{white-space:pre-wrap;color:#d2e4d8}.badge{font-size:12px;color:#92cba5}.approved{border-color:#87d5a0}.revision{border-color:#daaa7a}.card textarea{width:100%;margin:12px 0;min-height:55px}label{display:block}input{width:100%;margin:7px 0}button[aria-pressed=true]{background:#7dc897;color:#102318}.sticky{position:sticky;top:0;background:#111a15ef;padding:10px 0;backdrop-filter:blur(8px)}.warn{border-left:3px solid #d4aa77;padding:8px 18px;background:#392f2033}footer{margin:35px 0}button:disabled{opacity:.4;cursor:not-allowed}code{font-size:12px;overflow-wrap:anywhere}
    </style><h1>Review the protocol.<br>Before looking at the results.</h1><p>Review every scenario, factual control, and instruction template. Check clarity, synthetic content, family separation, consistent sentence count, neutral continuations, and whether persona wording reflects the intended instructions. These are drafts, not evidence.</p><p class="warn">This page has no network requests and does not run a model. Individual approvals are your human review; automated checks cannot replace it. A new dataset version is created only when you apply the exported decisions through the CLI.</p><div class="sticky"><div class="bar"><strong id="progress" aria-live="polite"></strong><button id="pending">Next pending item</button><button id="all">Show all</button></div></div><div id="cards"></div><footer><label>Reviewer name or stable ID<input id="reviewer" placeholder="Your name / reviewer ID" autocomplete="off"></label><label><input id="attest" type="checkbox" style="width:auto"> I personally reviewed these items before inspecting confirmatory results.</label><button id="save" disabled>Download review decisions</button><p id="status" role="status"></p><p>Apply your decisions with <code>python scripts/review_dataset.py apply --source datasets/synthetic-v1 --decisions YOUR_FILE.json --output datasets/synthetic-v1-reviewed --reviewer YOUR_ID --attest-human</code>. Items requiring revision must be revised in a new source version and reviewed again; they cannot be silently approved.</p></footer><script type="application/json" id="packet">'''+payload+'''</script><script>
    const packet=JSON.parse(document.getElementById('packet').textContent),cards=document.getElementById('cards');
    function update(){const pending=packet.entries.filter(e=>e.decision==='pending').length;document.getElementById('progress').textContent=`${packet.entries.length-pending} / ${packet.entries.length} reviewed · ${pending} pending`;document.getElementById('save').disabled=!!pending||!document.getElementById('attest').checked||!document.getElementById('reviewer').value.trim()}
    packet.entries.forEach((entry,i)=>{const card=document.createElement('section');card.className='card';card.id='item-'+i;const badge=document.createElement('div');badge.className='badge';badge.textContent=`${i+1} · ${entry.kind} · ${entry.id} · ${entry.content.split||entry.content.family||''}`;card.append(badge);const text=document.createElement('p');text.textContent=entry.content.text||entry.content.system_text;card.append(text);if(entry.content.common_continuation){const c=document.createElement('p');c.textContent='Shared continuation: '+entry.content.common_continuation;card.append(c)}const row=document.createElement('div');row.className='bar';['approve','needs_revision'].forEach(decision=>{const button=document.createElement('button');button.textContent=decision==='approve'?'Approve this item':'Needs revision';button.setAttribute('aria-pressed','false');button.onclick=()=>{entry.decision=decision;card.className='card '+(decision==='approve'?'approved':'revision');row.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));update()};row.append(button)});card.append(row);const notes=document.createElement('textarea');notes.placeholder='Review notes (optional)';notes.setAttribute('aria-label','Notes for '+entry.id);notes.oninput=()=>entry.notes=notes.value;card.append(notes);cards.append(card)});
    document.getElementById('reviewer').oninput=update;document.getElementById('attest').onchange=update;document.getElementById('pending').onclick=()=>{const i=packet.entries.findIndex(e=>e.decision==='pending');if(i>=0)document.getElementById('item-'+i).scrollIntoView({block:'center'});else document.getElementById('reviewer').focus()};document.getElementById('all').onclick=()=>window.scrollTo(0,0);document.getElementById('save').onclick=()=>{packet.reviewer=document.getElementById('reviewer').value.trim();packet.attestation='I personally reviewed every listed item before inspecting confirmatory results.';const url=URL.createObjectURL(new Blob([JSON.stringify(packet,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='ghosttrace-human-review-decisions.json';link.click();URL.revokeObjectURL(url);document.getElementById('status').textContent='Review decisions downloaded. Apply them to a new dataset version with the CLI.'};update();
    </script></html>'''
