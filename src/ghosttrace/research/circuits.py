"""Typed evidence graphs and a non-executing Circuit Tracer JSON adapter.

External replacement-model attributions never become original-model interventions
or tested paths merely because a source file uses a causal-sounding label.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import re
from ghosttrace.config import EngineError,canonical,digest
from ghosttrace.research.storage import ResearchRepository

MAX_NODES=5000;MAX_EDGES=20000;MAX_JSON=5_000_000
NODE_KINDS={'embedding','resid_pre','attn_in','mlp_in','head_pre','head_contribution','mlp_post_gate','attention_pattern','input_norm','attn_out','resid_mid','post_attention_norm','mlp_out','resid_post','attention_head','final_norm','logits','component','feature','error','token','target','imported_unknown'}
EVIDENCE={'structural','activation','attribution','intervention','tested_path'}
ALIASES={'estimated_attribution':'attribution','intervention_effect':'intervention','activation_overlay':'activation'}


def _finite(value,depth=0):
    if depth>32:raise EngineError('INVALID_BUNDLE','Graph JSON nesting exceeds32 levels')
    if isinstance(value,float) and not math.isfinite(value):raise EngineError('INVALID_BUNDLE','Graph values must be finite; missing values use null')
    if isinstance(value,list):
        if len(value)>MAX_EDGES:raise EngineError('INVALID_BUNDLE','Graph nested array exceeds limits')
        for child in value:_finite(child,depth+1)
    elif isinstance(value,dict):
        if len(value)>10000:raise EngineError('INVALID_BUNDLE','Graph object exceeds limits')
        for key,child in value.items():
            if not isinstance(key,str) or len(key)>500:raise EngineError('INVALID_BUNDLE','Graph object keys must be bounded strings')
            _finite(child,depth+1)


def _id(value):
    if not isinstance(value,str) or not 1<=len(value)<=200 or any(ord(c)<32 for c in value):raise EngineError('INVALID_BUNDLE','Graph IDs must be bounded printable strings')
    return value


def validate_graph(graph,known_interventions=None,*,external=False):
    if not isinstance(graph,dict):raise EngineError('INVALID_BUNDLE','Graph must be an object')
    _finite(graph)
    if len(canonical(graph))>MAX_JSON:raise EngineError('PAYLOAD_TOO_LARGE','Graph JSON exceeds5 MB')
    nodes=graph.get('nodes',[]);edges=graph.get('edges',graph.get('links',[]))
    if not isinstance(nodes,list) or not 1<=len(nodes)<=MAX_NODES or not isinstance(edges,list) or len(edges)>MAX_EDGES:raise EngineError('INVALID_BUNDLE','Graph node/edge inventory exceeds bounded schema')
    identifiers=set();clean_nodes=[];clean_edges=[];edge_ids=set();unsupported=[]
    for index,node in enumerate(nodes):
        if not isinstance(node,dict):raise EngineError('INVALID_BUNDLE','Each graph node must be an object')
        identifier=_id(node.get('id'))
        if identifier in identifiers:raise EngineError('INVALID_BUNDLE','Duplicate graph node ID')
        identifiers.add(identifier);kind=node.get('kind','component')
        if kind not in NODE_KINDS:raise EngineError('UNSUPPORTED_CAPTURE','Unsupported graph node kind',{'kind':kind})
        label=node.get('label',identifier)
        if not isinstance(label,str) or len(label)>2000:raise EngineError('INVALID_BUNDLE','Node labels must be bounded text annotations')
        for key,limit in [('layer',127),('token_index',4095),('feature_index',100_000_000),('head',255)]:
            if node.get(key) is not None and (type(node[key])is not int or not 0<=node[key]<=limit):raise EngineError('INVALID_BUNDLE',f'Invalid graph node {key}')
        value={**node,'id':identifier,'kind':kind,'label':label}
        if external:value.update(label_status='external_unvalidated_annotation',verified_in_original_model=False)
        clean_nodes.append(value)
    for index,edge in enumerate(edges):
        if not isinstance(edge,dict):raise EngineError('INVALID_BUNDLE','Each graph edge must be an object')
        source=_id(edge.get('source'));target=_id(edge.get('target'))
        if source not in identifiers or target not in identifiers:raise EngineError('INVALID_BUNDLE','Graph edge endpoint is absent')
        claimed=edge.get('evidence_type',edge.get('kind','structural'));kind=ALIASES.get(claimed,claimed)
        if kind not in EVIDENCE:raise EngineError('UNSUPPORTED_CAPTURE','Unsupported graph evidence type',{'type':kind})
        if external and kind in ('intervention','tested_path'):
            unsupported.append({'path':f'edges/{index}/evidence_type','claimed':kind,'reason':'External causal claims are retained as unverified source annotations, not promoted to locally tested paths.'});kind='attribution'
        value=edge.get('value',edge.get('weight',edge.get('raw_delta',edge.get('effect'))))
        if value is not None and (type(value)not in (float,int) or not math.isfinite(value)):raise EngineError('INVALID_BUNDLE','Edge effect must be finite or explicitly unavailable')
        if kind in ('intervention','tested_path'):
            proof=(known_interventions or {}).get(edge.get('evidence_record_id'))
            wanted='path' if kind=='tested_path' else 'exact'
            if not proof or proof.get('status')!='completed' or proof.get('spec',{}).get('kind')!=wanted:raise EngineError('CONFLICT','Measured/path graph claims need an actual compatible completed intervention record')
            pairs={(p.get('source'),p.get('target')) for p in proof.get('graph',{}).get('edges',[])}
            if (source,target) not in pairs:raise EngineError('CONFLICT','Intervention record did not test this graph edge/path')
        identifier=edge.get('id') or digest({'source':source,'target':target,'type':kind,'index':index})
        _id(identifier)
        if identifier in edge_ids:raise EngineError('INVALID_BUNDLE','Duplicate graph edge ID')
        edge_ids.add(identifier)
        clean_edges.append({**edge,'id':identifier,'source':source,'target':target,'kind':kind,'evidence_type':kind,'value':value,'measured':bool(not external and kind in ('activation','intervention','tested_path') and value is not None),'verification_status':'external_not_recomputed' if external else 'local_record_bound' if kind in ('intervention','tested_path') else 'structural' if kind=='structural' else 'observed_or_estimated','unavailable_reason':None if value is not None else ('architectural_connection_not_effect_measurement' if kind=='structural' else edge.get('unavailable_reason','No numeric effect was supplied.'))})
    errors=sum(n['kind']=='error' for n in clean_nodes);features=sum(n['kind']=='feature' for n in clean_nodes)
    for key in ('hidden_node_count','hidden_edge_count'):
        if type(graph.get(key,0))is not int or not 0<=graph.get(key,0)<=1_000_000_000:raise EngineError('INVALID_BUNDLE','Hidden graph counts must be bounded nonnegative integers')
    if not isinstance(graph.get('unsupported_fields',[]),list):raise EngineError('INVALID_BUNDLE','unsupported_fields must be a list')
    return {'schema_version':'2.0','nodes':clean_nodes,'edges':clean_edges,'hidden_node_count':graph.get('hidden_node_count',0),'hidden_edge_count':graph.get('hidden_edge_count',0),'unsupported_fields':unsupported+graph.get('unsupported_fields',[]),'coverage':{'visible_error_nodes':errors,'visible_feature_nodes':features,'error_node_fraction':errors/len(clean_nodes),'graph_coverage_status':'unknown' if external else 'scope_of_referenced_records','note':'Node counts are display coverage, not a replacement-model fidelity score or proof of causal completeness.'},'limitations':['Structural edges, activation overlays, attribution estimates, interventions and tested paths are distinct evidence types.','External graph values are reported source results, not recomputed original-model evidence.']}


def circuit_tracer_json(document,source_metadata):
    """Adapt the official frontend JSON nodes/links shape; never load .pt files."""
    if not isinstance(document,dict) or not isinstance(source_metadata,dict):raise EngineError('INVALID_BUNDLE','Circuit Tracer import requires JSON graph and explicit source metadata')
    _finite(document);_finite(source_metadata)
    if len(canonical(document))>MAX_JSON:raise EngineError('PAYLOAD_TOO_LARGE','Circuit Tracer JSON exceeds5 MB')
    required={'model_repo_id','model_revision','dictionary_revision','license','source_revision'}
    if not required<=set(source_metadata) or not re.fullmatch(r'[a-f0-9]{40}',str(source_metadata.get('model_revision',''))) or any(not isinstance(source_metadata[k],str) or not 1<=len(source_metadata[k])<=500 for k in required):raise EngineError('INVALID_CONFIG','External graph needs explicit pinned base model, dictionary/source revisions and license')
    raw_nodes=document.get('nodes');raw_links=document.get('links')
    if not isinstance(raw_nodes,list) or not isinstance(raw_links,list) or len(raw_nodes)>MAX_NODES or len(raw_links)>MAX_EDGES:raise EngineError('INVALID_BUNDLE','Expected bounded Circuit Tracer frontend JSON nodes/links; .pt checkpoints are unsupported')
    supported_top={'metadata','qParams','nodes','links'};unsupported=[{'path':key,'reason':'Unsupported external top-level field; not interpreted.'} for key in document if key not in supported_top]
    kinds={'cross layer transcoder':'feature','mlp reconstruction error':'error','embedding':'token','logit':'target'};nodes=[]
    for index,node in enumerate(raw_nodes):
        if not isinstance(node,dict):raise EngineError('INVALID_BUNDLE','External node must be an object')
        identifier=_id(node.get('node_id'));feature_type=node.get('feature_type');kind=kinds.get(feature_type,'imported_unknown');layer=node.get('layer');position=node.get('ctx_idx')
        if kind=='imported_unknown':unsupported.append({'path':f'nodes/{index}/feature_type','reason':'Unknown source feature type is not interpreted.','value':str(feature_type)[:200]})
        item={'id':identifier,'kind':kind,'label':node.get('clerp') or identifier,'token_index':position,'activation':node.get('activation'),'external_influence':node.get('influence'),'external_feature_id':node.get('feature'),'external_feature_type':feature_type,'external_token_probability':node.get('token_prob'),'label_status':'external_unvalidated_annotation'}
        if isinstance(layer,str) and layer.isdecimal() and int(layer)<128:item['layer']=int(layer)
        if kind=='feature':
            match=re.fullmatch(r'(\d+)_(\d+)_(\d+)',identifier)
            if match:item['feature_index']=int(match[2])
            else:unsupported.append({'path':f'nodes/{index}/node_id','reason':'Raw feature index could not be recovered; source feature field is a Cantor-paired identifier, not used as a feature index.'})
        known={'node_id','feature','layer','ctx_idx','feature_type','token_prob','is_target_logit','run_idx','reverse_ctx_idx','jsNodeId','clerp','influence','activation'}
        unsupported.extend({'path':f'nodes/{index}/{key}','reason':'Unsupported source field is not interpreted.'} for key in node if key not in known)
        nodes.append(item)
    edges=[]
    for index,link in enumerate(raw_links):
        if not isinstance(link,dict):raise EngineError('INVALID_BUNDLE','External link must be an object')
        edges.append({'source':link.get('source'),'target':link.get('target'),'kind':'attribution','evidence_type':'attribution','weight':link.get('weight'),'method':'external_circuit_tracer_replacement_model_attribution','claimed_source_semantics':'Direct effect in a replacement/transcoder graph; not an original-model tested path.'})
        unsupported.extend({'path':f'links/{index}/{key}','reason':'Unsupported source link field is not interpreted.'} for key in link if key not in {'source','target','weight'})
    graph=validate_graph({'nodes':nodes,'edges':edges,'unsupported_fields':unsupported},external=True)
    graph['source_metadata']=source_metadata;graph['source_format']='circuit-tracer-frontend-json';graph['source_graph_hash']=digest(document);graph['source_metadata_hash']=digest(source_metadata)
    graph['external_metadata']=document.get('metadata',{});graph['upstream_pruning_unknown']=True
    return graph


def import_circuit(store,study_id,document,source_metadata=None):
    repo=ResearchRepository(store);study=repo.get(study_id,'study')
    if source_metadata is None:
        if not isinstance(document,dict):raise EngineError('INVALID_BUNDLE','Circuit import must be a JSON object')
        source_metadata=document.get('source_metadata');document=document.get('graph',document)
    if not isinstance(source_metadata,dict):raise EngineError('INVALID_CONFIG','Circuit import requires pinned source_metadata')
    _finite(source_metadata)
    required={'model_repo_id','model_revision','dictionary_revision','license','source_revision'}
    if not required<=set(source_metadata) or not re.fullmatch(r'[a-f0-9]{40}',str(source_metadata.get('model_revision',''))) or any(not isinstance(source_metadata[k],str) or not 1<=len(source_metadata[k])<=500 for k in required):raise EngineError('INVALID_CONFIG','External graph needs explicit pinned base model, dictionary/source revisions and license')
    if not isinstance(document,dict):raise EngineError('INVALID_BUNDLE','Circuit graph must be a JSON object')
    profile=study['model_profile']
    if source_metadata.get('model_repo_id')!=profile['repo_id'] or source_metadata.get('model_revision')!=profile['revision']:raise EngineError('CONFLICT','External circuit base model/revision differs from the frozen study')
    graph=circuit_tracer_json(document,source_metadata) if 'links' in document else validate_graph(document,external=True)
    return repo.put('circuit',{'study_id':study_id,'source':'external_import','graph':graph,'source_metadata':source_metadata,'source_hash':digest(document),'measurement_status':'external_reported_not_recomputed','limitations':['External import preserves reported estimates; it does not validate the replacement model or promote any path to locally tested causal evidence.','Opaque/pickled Circuit Tracer .pt files are never loaded; export safe frontend JSON instead.']},study_id)


def filter_graph(graph,*,minimum_absolute_value=None,node_ids=None,max_nodes=500,max_edges=2000):
    # Filtering is selection over already validated evidence, not a new scientific claim.
    # Never demote or promote a local tested path while choosing a visible subgraph.
    _finite(graph)
    if not isinstance(graph.get('nodes'),list) or not isinstance(graph.get('edges'),list) or len(graph['nodes'])>MAX_NODES or len(graph['edges'])>MAX_EDGES:raise EngineError('INVALID_CONFIG','Filter requires a bounded validated graph')
    ids=[_id(n.get('id')) for n in graph['nodes']]
    if len(ids)!=len(set(ids)) or any(e.get('source') not in ids or e.get('target') not in ids for e in graph['edges']):raise EngineError('INVALID_CONFIG','Filter graph has duplicate or missing endpoints')
    validated={**graph,'edges':[{**e,'value':e.get('value',e.get('weight'))} for e in graph['edges']]}
    if minimum_absolute_value is not None and (type(minimum_absolute_value)not in (int,float) or not math.isfinite(minimum_absolute_value) or minimum_absolute_value<0):raise EngineError('INVALID_CONFIG','Graph threshold must be finite and nonnegative')
    if type(max_nodes)is not int or not 1<=max_nodes<=MAX_NODES or type(max_edges)is not int or not 0<=max_edges<=MAX_EDGES:raise EngineError('INVALID_CONFIG','Graph display limits exceed bounds')
    chosen=set(node_ids) if node_ids is not None else None
    if chosen is not None and not chosen<={n['id'] for n in validated['nodes']}:raise EngineError('INVALID_CONFIG','Requested graph node is absent')
    nodes=[n for n in validated['nodes'] if chosen is None or n['id'] in chosen][:max_nodes];kept={n['id'] for n in nodes}
    edges=[e for e in validated['edges'] if e['source'] in kept and e['target'] in kept and (minimum_absolute_value is None or e['value'] is not None and abs(e['value'])>=minimum_absolute_value)][:max_edges]
    return {**validated,'nodes':nodes,'edges':edges,'hidden_node_count':len(validated['nodes'])-len(nodes)+int(validated.get('hidden_node_count') or 0),'hidden_edge_count':len(validated['edges'])-len(edges)+int(validated.get('hidden_edge_count') or 0),'filter':{'minimum_absolute_value':minimum_absolute_value,'node_ids':sorted(chosen) if chosen is not None else None,'max_nodes':max_nodes,'max_edges':max_edges,'selection':'Declared node filter, signed-value magnitude threshold, then original input order; hidden/unmeasured edges are not zero effects.'},'unmeasured_edges_in_source':sum(e['value'] is None for e in validated['edges'])}
