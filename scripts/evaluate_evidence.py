#!/usr/bin/env python3
"""Independent evidence-contract evaluator. Never executes or extracts bundle content.

These are structural/scientific guardrails, not a claim that a model has emotions.
JSON output deliberately distinguishes a missing measurement from numeric zero.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import re
from pathlib import Path, PurePosixPath
import zipfile

MAX_BUNDLE_BYTES = 2_000_000_000
MAX_ENTRIES = 10_000
SCHEMA_VERSION = '1.0'
CONDITIONS = {'neutral', 'remorseful', 'unremorseful'}
SITE = 'residual_post_block_pre_final_norm'


def check(name, passed, detail, *, pending=False):
    return {'id': name, 'status': 'pending' if pending else 'pass' if passed else 'fail', 'detail': detail}


def finite_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    return True


def evaluate_records(manifest, analysis, runs, *, require_measured=False):
    """Evaluate decoded records independently of SQL and HTTP implementation."""
    checks = []
    checks.append(check('finite_json', finite_tree([manifest, analysis, runs]), 'Unavailable values must be null, never NaN or infinity.'))
    profile = manifest.get('model_profile', {})
    measured = analysis.get('measurement_status') == 'measured'
    fixture = bool(manifest.get('fixture_type') or manifest.get('illustrative') or manifest.get('is_fixture'))
    fixture |= any(r.get('fixture') or r.get('is_fixture') or r.get('model_profile', {}).get('fixture') for r in runs)
    checks.append(check('fixture_separation', not (fixture and measured), 'Illustrative or synthetic test artifacts cannot be labeled measured.'))
    checks.append(check('measurement_available', measured, 'A measured run is required for release evidence.' if require_measured else 'No measured experiment is a valid development state.', pending=not measured and not require_measured))
    if measured:
        checks.append(check('immutable_model', bool(re.fullmatch('[a-f0-9]{40}', str(profile.get('revision', '')))) and bool(re.fullmatch('[a-f0-9]{40}', str(profile.get('tokenizer_revision', '')))), 'Model and tokenizer revisions must be immutable 40-character commits.'))
        checks.append(check('provenance', all(profile.get(k) for k in ('config_hash', 'chat_template_hash', 'weight_file_hashes', 'device', 'compute_dtype', 'torch_version', 'transformers_version')), 'Measured evidence requires model, backend, weight, and template provenance.'))
        generated = [r for r in runs if r.get('mode') == 'greedy_generation' and r.get('status') == 'completed' and not r.get('control')]
        captured = [r for r in runs if r.get('mode') == 'fixed_continuation' and r.get('status') == 'completed' and not r.get('control')]
        checks.append(check('three_conditions', {r.get('condition') for r in generated} >= CONDITIONS and {r.get('condition') for r in captured} >= CONDITIONS, 'Real smoke evidence needs generation and fixed-continuation capture for all three instructions.'))
        actual_ids = {r.get('id') for r in captured}
        checks.append(check('analysis_unit_provenance', set(analysis.get('included_units', [])) <= {r.get('id') for r in runs}, 'Every included analysis unit must resolve to a retained raw record.'))
        capture_ok = bool(captured)
        for r in captured:
            art = r.get('activation_artifact', {})
            shape = art.get('shape', [])
            capture_ok &= art.get('site') == SITE and art.get('dtype') == 'float32' and art.get('axis_names') == ['layer', 'selected_token', 'hidden'] and art.get('normalization_state') == 'before_final_model_norm' and len(shape) == 3
            mask = art.get('pool_mask', [])
            capture_ok &= len(mask) == 16 and len(set(mask)) == 16 and all(isinstance(p, int) and 0 <= p < (shape[1] if len(shape) == 3 else 0) for p in mask)
        checks.append(check('capture_contract', capture_ok, 'Capture site, axes, normalization and sixteen distinct pooled positions must be explicit.'))
        alignment_ok = True
        pairs = {}
        for r in captured:
            key = (r.get('scenario_id'), r.get('template_id'))
            ids = r.get('continuation_ids')
            if not isinstance(ids, list) or len(ids) < 16:
                alignment_ok = False
            if key in pairs and pairs[key] != ids:
                alignment_ok = False
            pairs[key] = ids
        checks.append(check('matched_token_identity', alignment_ok, 'Corresponding continuation token IDs must match within each scenario/template.'))
        checks.append(check('repeat_floor', 'neutral-repeat' in analysis.get('aggregates', {}) and analysis['aggregates']['neutral-repeat'].get('primary', {}).get('n', 0) > 0, 'Measured neutral exact-repeat floor must accompany persona contrasts.'))
        if manifest.get('config', {}).get('templates') == ['A', 'B']:
            checks.append(check('wording_control', analysis.get('aggregates', {}).get('neutral-wording', {}).get('primary', {}).get('n', 0) > 0, 'A/B pilot evidence needs a measured neutral wording reference.'))
    # Empty effects remain empty, rather than becoming reassuring zero bars.
    missing_ok = True
    for a in analysis.get('aggregates', {}).values():
        primary = a.get('primary', {})
        if primary.get('n') == 0:
            missing_ok &= all(primary.get(k) is None for k in ('mean', 'lower', 'upper'))
        else:
            m, lo, hi = (primary.get(k) for k in ('mean', 'lower', 'upper'))
            missing_ok &= all(isinstance(v, (int, float)) and math.isfinite(v) for v in (m, lo, hi)) and lo <= hi
    checks.append(check('missingness_not_zero', missing_ok, 'An empty denominator must have null mean and interval; nonempty intervals must be finite and ordered.'))
    settings = analysis.get('settings', {})
    checks.append(check('scenario_bootstrap', settings.get('bootstrap_draws') == 2000 and settings.get('metric_version') == 'normalized-l2-v1', 'Canonical intervals use two thousand scenario-level draws and a versioned metric.'))
    scenarios = manifest.get('scenarios', [])
    family_splits = {}
    split_ok = True
    for row in scenarios:
        family = row.get('family_id')
        split = row.get('split')
        if family in family_splits and family_splits[family] != split:
            split_ok = False
        family_splits[family] = split
    checks.append(check('family_split_separation', split_ok, 'Scenario families cannot span development and holdout.'))
    if settings.get('split') in ('holdout', 'all'):
        checks.append(check('holdout_preregistration', bool(manifest.get('preregistration_hash')) and all(r.get('review_status') == 'reviewed' for r in scenarios), 'Holdout analysis requires frozen preregistration and reviewed scenarios.'))
    return checks


def evaluate_bundle(path, *, require_measured=False):
    checks = []
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = [e.filename for e in entries]
            safe = len(entries) <= MAX_ENTRIES and len(set(names)) == len(names) and sum(e.file_size for e in entries) <= MAX_BUNDLE_BYTES
            safe &= all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts and '\\' not in n and not re.match(r'^[A-Za-z]:', n) and not n.startswith('/') for n in names)
            # ZIP symlinks and encrypted payloads have no role in evidence import.
            safe &= all(not e.flag_bits & 1 and ((e.external_attr >> 16) & 0o170000) != 0o120000 for e in entries)
            checks.append(check('bounded_safe_archive', safe, 'Unique normalized files only, no links/encryption, at most two GB expanded and ten thousand entries.'))
            if not safe:
                return {'schema_version': SCHEMA_VERSION, 'status': 'fail', 'checks': checks}
            inventory = json.loads(archive.read('inventory.json'))
            indexed = inventory.get('file_inventory_with_checksums', {})
            hashes_ok = set(names) == set(indexed) | {'inventory.json'} and all(name in names and re.fullmatch('[a-f0-9]{64}', str(sha)) and hashlib.sha256(archive.read(name)).hexdigest() == sha for name, sha in indexed.items())
            checks.append(check('inventory_checksums', hashes_ok, 'Every archive file is inventoried and checksum verified.'))
            if not hashes_ok:
                return {'schema_version': SCHEMA_VERSION, 'status': 'fail', 'checks': checks}
            manifest = json.loads(archive.read('manifest.json'))
            analysis = json.loads(archive.read('analysis.json'))
            runs = [json.loads(archive.read(n)) for n in names if re.fullmatch(r'runs/[a-f0-9]{64}/record\.json', n)]
            checks += evaluate_records(manifest, analysis, runs, require_measured=require_measured)
            checks.append(check('raw_arrays_declared', bool(inventory.get('includes_raw_activations')) == any(n.endswith('/activations.npz') for n in names), 'Raw-array inclusion must match the export declaration.'))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, TypeError, AttributeError, RecursionError) as e:
        checks.append(check('readable_contract', False, f'{type(e).__name__}: {e}'))
    return {'schema_version': SCHEMA_VERSION, 'status': 'fail' if any(c['status'] == 'fail' for c in checks) else 'pending' if any(c['status'] == 'pending' for c in checks) else 'pass', 'checks': checks}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle', type=Path)
    p.add_argument('--require-measured', action='store_true')
    p.add_argument('--output', type=Path)
    args = p.parse_args(argv)
    report = evaluate_bundle(args.bundle, require_measured=args.require_measured)
    serialized = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end='')
    return 1 if report['status'] == 'fail' else 0


if __name__ == '__main__':
    raise SystemExit(main())
