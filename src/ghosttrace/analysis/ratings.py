"""Condition-hidden annotation tasks. Expressed behavior is not an internal state."""
from __future__ import annotations
import hashlib
import json
import random
from collections import defaultdict
from ghosttrace.config import digest,EngineError

DIMENSIONS=['regret','responsibility','repair','justification_deflection']
RUBRIC={'version':'expressed-behavior-v1','scale':{'0':'Absent','1':'Some evidence','2':'Clear evidence'},'dimensions':DIMENSIONS,'interpretation':'Rate the behavior expressed in the response; do not infer feelings. Wording can reveal the condition despite blinded metadata.'}

def blinded_tasks(store,experiment_id,rater_id,limit=36):
    exp=store.experiment(experiment_id)
    with store.connect() as db:
        rated={r['run_unit_id'] for r in db.execute('SELECT ratings.run_unit_id FROM ratings JOIN units ON units.id=ratings.run_unit_id WHERE units.experiment_id=? AND json_extract(ratings.record,\'$.rater_id\')=?',(experiment_id,rater_id))}
    buckets=defaultdict(list)
    eligible=[]
    for row in store.unit_rows(experiment_id):
        record=json.loads(row['record'])
        if record['status']=='completed' and record['mode']=='greedy_generation' and not record.get('control'):
            buckets[(record['condition'],record['template_id'])].append(record);eligible.append(record)
    # Choose one balanced cohort per experiment, then vary display ordering by rater.
    cohort_rng=random.Random(int(digest({'experiment':experiment_id,'version':'blind-cohort-v1'})[:16],16))
    for bucket in buckets.values():cohort_rng.shuffle(bucket)
    cohort=[];keys=sorted(buckets)
    while keys and len(cohort)<limit:
        for key in list(keys):
            if not buckets[key]:keys.remove(key);continue
            cohort.append(buckets[key].pop(0))
            if len(cohort)>=limit:break
    selected=[record for record in cohort if record['id'] not in rated]
    rng=random.Random(int(digest({'experiment':experiment_id,'rater':rater_id,'version':'blind-order-v1'})[:16],16))
    rng.shuffle(selected)
    cohort_id=digest({'experiment':experiment_id,'unit_ids':sorted(r['id'] for r in cohort),'version':'blind-cohort-v1'})
    items=[{'blinded_display_id':digest({'experiment':experiment_id,'rater':rater_id,'unit':r['id'],'version':'blind-v1'})[:20],'run_unit_id':r['id'],'response_text':r.get('text',''),'scenario_text':r.get('scenario',{}).get('text',''),'stop_reason':r.get('stop_reason'),'truncated':r.get('truncated',False)} for r in selected]
    return {'experiment_id':experiment_id,'rater_id':rater_id,'cohort_id':cohort_id,'cohort_response_count':len(cohort),'cohort_remaining_count':len(selected),'items':items,'eligible_response_count':len(eligible),'rated_response_count':len(rated),'remaining_response_count':len(eligible)-len(rated),'target_minimum':36,'balanced_by_condition_and_template':True,'rubric':RUBRIC,'metadata_hidden':['condition','template','model','activation_metrics'],'limitation':'Blinding cannot conceal stylistic clues in the response wording.'}


def weighted_kappa(pairs):
    """Quadratically weighted Cohen's kappa on paired ordinal 0–2 scores."""
    if not pairs:return {'n':0,'agreement':None,'weighted_kappa':None,'reason':'no_paired_ratings'}
    if any(type(a) is not int or type(b) is not int or a not in (0,1,2) or b not in (0,1,2) for a,b in pairs):raise ValueError('Scores must be 0–2')
    n=len(pairs);marginal_a=[sum(a==i for a,b in pairs)/n for i in range(3)];marginal_b=[sum(b==i for a,b in pairs)/n for i in range(3)]
    observed=sum(((a-b)/2)**2 for a,b in pairs)/n
    expected=sum(((i-j)/2)**2*marginal_a[i]*marginal_b[j] for i in range(3) for j in range(3))
    return {'n':n,'agreement':sum(a==b for a,b in pairs)/n,'weighted_kappa':None if expected<=1e-12 else 1-observed/expected,'reason':'constant_marginals' if expected<=1e-12 else None,'weights':'quadratic','scale':[0,1,2]}

def rating_summary(store,experiment_id):
    import itertools
    import collections
    exp=store.experiment(experiment_id)
    units={r['id']:json.loads(r['record']) for r in store.unit_rows(experiment_id)}
    with store.connect() as db:
        records=[json.loads(r['record']) for r in db.execute('SELECT ratings.record FROM ratings JOIN units ON units.id=ratings.run_unit_id WHERE units.experiment_id=? ORDER BY json_extract(ratings.record,\'$.created_at\'),ratings.id',(experiment_id,))]
    unique={};duplicates=[]
    for rating in records:
        key=(rating['rater_id'],rating['run_unit_id'],rating['rubric_version'])
        if key in unique:duplicates.append(rating['id'])
        else:unique[key]=rating
    raters=sorted({k[0] for k in unique});per_rater=[]
    for rater in raters:
        items=[r for k,r in unique.items() if k[0]==rater]
        balance=collections.Counter((units[r['run_unit_id']]['condition'],units[r['run_unit_id']]['template_id']) for r in items)
        per_rater.append({'rater_id':rater,'unique_response_count':len(items),'target_minimum':36,'minimum_reached':len(items)>=36,'condition_template_counts':[{'condition':c,'template':t,'n':n} for (c,t),n in sorted(balance.items())]})
    comparisons=[]
    for first,second in itertools.combinations(raters,2):
        common=sorted({(k[1],k[2]) for k in unique if k[0]==first}&{(k[1],k[2]) for k in unique if k[0]==second})
        dimensions={d:weighted_kappa([(unique[(first,u,r)]['dimension_scores'][d],unique[(second,u,r)]['dimension_scores'][d]) for u,r in common]) for d in DIMENSIONS}
        disagreements=[]
        for unit,rubric in common:
            a=unique[(first,unit,rubric)];b=unique[(second,unit,rubric)]
            differing=[d for d in DIMENSIONS if a['dimension_scores'][d]!=b['dimension_scores'][d]]
            if differing:disagreements.append({'run_unit_id':unit,'rubric_version':rubric,'rating_ids':[a['id'],b['id']],'dimensions':differing,'original_scores':[a['dimension_scores'],b['dimension_scores']],'adjudication':None})
        comparisons.append({'raters':[first,second],'paired_response_count':len(common),'dimensions':dimensions,'disagreements':disagreements})
    limitations=['Small-sample agreement and kappa estimates are descriptive; wording can reveal prompt condition despite metadata blinding.','Agreement pairs the same run unit and rubric version. Original scores and disagreements are retained; there is no automatic adjudication.']
    if len(raters)<2:limitations.append('Fewer than two raters: inter-rater reliability is unavailable and kappa is omitted.')
    if duplicates:limitations.append('Duplicate rater/unit/rubric records were excluded from reliability; the earliest immutable score is used.')
    return {'experiment_id':experiment_id,'rubric_version':'expressed-behavior-v1','rater_count':len(raters),'rating_record_count':len(records),'unique_rating_count':len(unique),'raters':per_rater,'reliability_available':len(raters)>=2,'pairwise_agreement':comparisons,'excluded_duplicate_rating_ids':duplicates,'limitations':limitations}
