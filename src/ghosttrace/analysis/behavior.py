"""Descriptive expressed-style indicators and run accounting; no psychological scores."""
import collections
import statistics
from ghosttrace.analysis.bootstrap import bootstrap_scenarios
from ghosttrace.analysis.metrics import lexical_counts

def _summary_by_scenario(items,field,seed):
    groups=collections.defaultdict(list)
    for row in items:
        value=row.get(field)
        if value is not None:groups[row['scenario_id']].append(value)
    values={sid:statistics.fmean(group) for sid,group in groups.items()}
    return {**bootstrap_scenarios(list(values.values()),seed=seed),'scenario_values':values,'unit_count':sum(len(v) for v in groups.values()),'missing_unit_count':sum(row.get(field) is None for row in items)}

def descriptive_outputs(records,expected_units,seed):
    rows=[];nll_rows=[]
    for record in records:
        if record.get('status')!='completed':continue
        if record.get('mode')=='fixed_continuation':
            nll_rows.append({'unit_id':record['id'],'scenario_id':record['scenario_id'],'condition':record['condition'],'template':record['template_id'],'control':record.get('control'),'nll':record.get('continuation_nll')})
        if record.get('mode')!='greedy_generation' or record.get('control'):continue
        text=record.get('text');counts=lexical_counts(text)['counts'] if isinstance(text,str) else None
        row={'unit_id':record['id'],'scenario_id':record['scenario_id'],'condition':record['condition'],'template':record['template_id'],'generated_token_count':record.get('generated_token_count',len(record['generated_ids']) if isinstance(record.get('generated_ids'),list) else None),'word_count':len(text.split()) if isinstance(text,str) else None,'truncated':int(record['truncated']) if isinstance(record.get('truncated'),bool) else None,'stop_reason':record.get('stop_reason'),'refusal':record.get('refusal'),'malformed':record.get('malformed'),'lexical_counts':counts,'seconds':record.get('seconds')}
        if counts:row.update({'lexical_'+key:value for key,value in counts.items()})
        rows.append(row)
    conditions={}
    for condition in ['neutral','remorseful','unremorseful']:
        selected=[r for r in rows if r['condition']==condition]
        expected=[u for u in expected_units if u['mode']=='greedy_generation' and not u.get('control') and u['condition']==condition]
        fields=['generated_token_count','word_count','truncated','lexical_regret','lexical_responsibility','lexical_repair','lexical_deflection']
        conditions[condition]={'expected_units':len(expected),'completed_units':len(selected),'missing_or_failed_units':len(expected)-len(selected),'metrics':{field:_summary_by_scenario(selected,field,seed) for field in fields},'refusal_flags_present':sum(r['refusal'] is not None for r in selected),'refusal_count':sum(r['refusal'] is True for r in selected),'malformed_flags_present':sum(r['malformed'] is not None for r in selected),'malformed_count':sum(r['malformed'] is True for r in selected)}
    nll_conditions={}
    for condition in ['neutral','remorseful','unremorseful']:
        selected=[r for r in nll_rows if r['condition']==condition and not r.get('control')]
        nll_conditions[condition]=_summary_by_scenario(selected,'nll',seed)
    return {'interpretation':'Expressed behavior and weak lexical phrase-count proxies; not emotional-state or remorse scores. Template units are averaged within scenario before scenario bootstrap.','lexical_version':'lexical-v1','conditions':conditions,'units':rows,'continuation_nll':{'interpretation':'Mean negative log probability of identical teacher-forced continuation tokens.','conditions':nll_conditions,'units':nll_rows},'flags_limitation':'Absent refusal/malformed flags remain unavailable; they are never inferred as false.'}
