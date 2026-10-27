#!/usr/bin/env python3
"""Executable evidence checks for measured V2 diagnostics; no scientific claim upgrade."""
from __future__ import annotations
import argparse,hashlib,json,math,sys
from pathlib import Path
import xml.etree.ElementTree as ET

REAL_CASES={
 'test_real_components_norms_heads_attention_and_logits_parity',
 'test_real_exact_patch_positive_control_and_self_patch',
 'test_real_selected_head_only_replacement_and_gqa',
 'test_real_attribution_and_tested_receiver_path_are_separate',
 'test_real_fixed_candidate_scoring_and_component_cleanup',
}
def evaluate(bundle,junit):
    checks=[]
    def check(name,passed,detail):checks.append({'name':name,'passed':bool(passed),'detail':detail})
    cases=list(ET.parse(junit).iter('testcase'));ids=[(c.get('classname'),c.get('name')) for c in cases]
    required={('tests.real_model.test_mechanistic',name) for name in REAL_CASES}
    check('actual_model_testcases',set(ids)==required and len(ids)==len(required) and all(not any(c.find(k)is not None for k in ['failure','error','skipped']) for c in cases),'Five exact real-model test records, no skipped, duplicate, unrelated or empty-count substitutes.')
    p=json.loads(Path(bundle).read_text(),parse_constant=lambda value:(_ for _ in ()).throw(ValueError(value)))
    def finite(value):
        if isinstance(value,float) and not math.isfinite(value):raise ValueError('Evidence contains nonfinite numeric values')
        if isinstance(value,dict):
            for child in value.values():finite(child)
        elif isinstance(value,list):
            for child in value:finite(child)
    finite(p)
    check('v2_study',p.get('schema_version')=='2.0' and p.get('study',{}).get('study_hash'),'Versioned immutable study.')
    captures=p.get('captures',[]);runs=p.get('interventions',[])
    check('measured_capture_pair',len(captures)>=2 and all(c.get('status')=='completed' and c.get('provenance') and c.get('artifact',{}).get('sha256') for c in captures),'At least two completed captures with artifact and backend provenance.')
    parity=[c.get('metrics',{}).get('logit_parity_max_abs_error') for c in captures]
    check('capture_logit_parity',bool(parity) and all(type(v)in(int,float) and math.isfinite(v) and 0<=v<=1e-5 for v in parity),parity)
    kinds={r.get('spec',{}).get('kind') for r in runs if r.get('status')=='completed'}
    check('separate_methods',{'exact','attribution','path'}<=kinds,sorted(kinds))
    exact=[r for r in runs if r.get('spec',{}).get('kind')=='exact'];paths=[r for r in runs if r.get('spec',{}).get('kind')=='path']
    check('exact_and_path_controls',bool(exact) and bool(paths) and all(r.get('checks',{}).get('self_patch_passed') is True and r.get('controls') for r in exact+paths),'Original-model self controls must pass; controls are actual recorded values.')
    check('raw_effect_arithmetic',bool(exact+paths) and all(type(r.get('raw_delta'))in(int,float) and abs(r['raw_delta']-(r['metrics']['patched']-r['metrics']['baseline']))<=1e-5 for r in exact+paths),'Raw effect equals patched minus baseline, including zero and negative effects.')
    estimated=[r for r in runs if r.get('spec',{}).get('kind')=='attribution']
    check('attribution_not_exact',bool(estimated) and all(r.get('raw_delta')is None and type(r.get('estimated_delta'))in(int,float) and type(r.get('exact_reference_delta'))in(int,float) for r in estimated),'Gradient estimate, exact reference and linearization error stay separate.')
    features=p.get('features',[])
    check('feature_claims_explicit',all(r.get('dictionary_classification')!='identity_coordinate_diagnostic' or r.get('learned_features') is False for r in features),'Identity coordinates cannot be labeled learned semantic features.')
    check('feature_fidelity_not_inferred',all(r.get('behavioral_fidelity',{}).get('status')=='not_measured' for r in features),'Projection reconstruction does not independently attest behavioral fidelity; feature intervention records carry that experiment.')
    feature_runs=p.get('feature_interventions',[]);feature_estimates=p.get('feature_attributions',[]);lenses=p.get('lenses',[]);lens_results=p.get('lens_results',[])
    check('feature_original_model_controls',bool(feature_runs) and all(r.get('status')=='completed' and r.get('metrics',{}).get('checks',{}).get('self_patch_passed') is True and {'reconstruction_only','reconstruction_plus_saved_error','norm_matched_random_delta'}<=set(r.get('metrics',{}).get('controls',{})) for r in feature_runs),'Measured coordinate edits retain original-model reconstruction/error/random controls.')
    local_feature_ids={r['id']:r for r in feature_runs}
    comparisons=[c for r in feature_estimates for c in r.get('exact_intervention_comparisons',[])]
    check('feature_estimates_exact_references',bool(feature_estimates) and bool(comparisons) and all(c.get('intervention_id') in local_feature_ids and type(c.get('estimated_delta'))in(int,float) and type(c.get('exact_delta'))in(int,float) and abs(c['exact_delta']-local_feature_ids[c['intervention_id']]['raw_delta'])<=1e-5 and abs(abs(c['estimated_delta']-c['exact_delta'])-c['linearization_error'])<=1e-5 for c in comparisons),'Feature gradient estimates reference independently recorded exact edits and numeric linearization error.')
    predictive=[lens for lens in lenses if lens.get('objective')=='predictive_kl']
    def valid_lens(lens):
        meta=lens.get('metadata',{});metrics=list(lens.get('metrics',{}).values())
        return meta.get('training_split')=='development' and meta.get('holdout_used_for_fit') is False and meta.get('excludes_persona_fixture') is True and not set(meta.get('development_group_ids',[]))&set(meta.get('holdout_group_ids',[])) and bool(metrics) and all(m.get('heldout_used_for_fit') is False and m.get('model_parameters_updated') is False and type(m.get('holdout_kl'))in(int,float) and m.get('holdout_rows',0)>0 and m.get('objective')=='predictive_kl' for m in metrics)
    check('predictive_lens_frozen_split',bool(predictive) and all(valid_lens(lens) for lens in predictive),'Actual predictive-KL translator uses development-only groups, detached teacher, held-out evaluation and unchanged model parameters; diagnostic scope remains explicit.')
    check('lens_actual_readout',bool(lens_results) and all(r.get('prediction_status')=='measured_model_readout' and r.get('readouts') for r in lens_results),'Lens output predictions have compatible-model normalization/unembedding readouts, not invented token labels.')
    graph=p.get('graph',{});edges=graph.get('edges',[])
    check('structural_not_effect',all(e.get('value',e.get('raw_delta'))is None for e in edges if e.get('evidence_type',e.get('kind'))=='structural'),'Architectural connections carry no invented numeric effect.')
    return {'schema_version':'2.0-mechanistic-evaluation','status':'passed' if all(c['passed'] for c in checks) else 'failed','checks':checks,'evidence':{str(path):hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in [bundle,junit]},'limitations':['This is a measured copying diagnostic, not a population-level mechanistic claim.','Human interpretation, scientific review and release approval remain separate gates.']}

def main():
    a=argparse.ArgumentParser();a.add_argument('--bundle',type=Path,required=True);a.add_argument('--real-model-junit',type=Path,required=True);a.add_argument('--output',type=Path,default=Path('artifacts/qa/mechanistic-evaluation.json'));args=a.parse_args()
    try:report=evaluate(args.bundle,args.real_model_junit)
    except Exception as error:report={'schema_version':'2.0-mechanistic-evaluation','status':'failed','error':str(error)}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'output':str(args.output),'failed_checks':[c['name'] for c in report.get('checks',[]) if not c['passed']]}));return 0 if report['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
