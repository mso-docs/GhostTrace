from __future__ import annotations
from pathlib import Path
from pydantic import ValidationError
from ghosttrace.config import EngineError,read_json_file

def add_research_parser(sub):
    research=sub.add_parser('research',help='Version 2 general mechanistic studies; preserves v1 protocols')
    commands=research.add_subparsers(dest='research_command',required=True)
    create=commands.add_parser('create');create.add_argument('--config',required=True)
    capture=commands.add_parser('capture');capture.add_argument('--study',required=True);capture.add_argument('--conditions',nargs='*');capture.add_argument('--idempotency-key')
    intervention=commands.add_parser('intervene');intervention.add_argument('--study',required=True);intervention.add_argument('--spec',required=True);intervention.add_argument('--idempotency-key')
    bundle=commands.add_parser('bundle');bundle.add_argument('--study',required=True);bundle.add_argument('--output')
    export=commands.add_parser('export');export.add_argument('--study',required=True);export.add_argument('--summary',action='store_true')
    imported=commands.add_parser('import');imported.add_argument('archive')
    hypothesis=commands.add_parser('hypothesis');hypothesis.add_argument('--study',required=True);hypothesis.add_argument('--spec',required=True)
    dictionary=commands.add_parser('dictionary-import');dictionary.add_argument('archive')
    features=commands.add_parser('features');features.add_argument('--study',required=True);features.add_argument('--capture',required=True);features.add_argument('--dictionary',required=True)
    for command in ('lens-import','lens-train'):
        commands.add_parser(command).add_argument('archive')
    lens=commands.add_parser('lens-apply');lens.add_argument('--study',required=True);lens.add_argument('--capture',required=True);lens.add_argument('--lens',required=True);lens.add_argument('--no-decode',action='store_true');lens.add_argument('--layers',nargs='*',type=int)
    circuit=commands.add_parser('circuit-import');circuit.add_argument('--study',required=True);circuit.add_argument('--config',required=True)
    edit=commands.add_parser('feature-intervene');edit.add_argument('--study',required=True);edit.add_argument('--capture',required=True);edit.add_argument('--dictionary',required=True);edit.add_argument('--indices',nargs='+',type=int,required=True);edit.add_argument('--operation',choices=['ablate','scale','set'],default='ablate');edit.add_argument('--value',type=float,default=0)
    attribution=commands.add_parser('feature-attribution');attribution.add_argument('--study',required=True);attribution.add_argument('--capture',required=True);attribution.add_argument('--dictionary',required=True);attribution.add_argument('--top-k',type=int,default=64)

def execute_research_cli(store,args):
    from ghosttrace.research.service import freeze_study,queue_capture,queue_intervention,get_bundle,save_hypothesis
    try:
        command=args.research_command
        if command=='create':return freeze_study(store,read_json_file(args.config,dict))
        if command=='capture':return queue_capture(store,args.study,args.conditions,args.idempotency_key)
        if command=='intervene':return queue_intervention(store,args.study,read_json_file(args.spec,dict),args.idempotency_key)
        if command=='bundle':
            value=get_bundle(store,args.study)
            if args.output:
                from ghosttrace.storage.artifacts import atomic_json
                atomic_json(Path(args.output),value);return {'study_id':args.study,'output':str(Path(args.output))}
            return value
        if command=='export':
            from ghosttrace.research.bundles import export_research
            return export_research(store,args.study,not args.summary)
        if command=='import':
            from ghosttrace.research.bundles import import_research
            return import_research(store,args.archive)
        if command=='hypothesis':return save_hypothesis(store,args.study,read_json_file(args.spec,dict))
        if command=='dictionary-import':
            from ghosttrace.research.features import import_dictionary
            return import_dictionary(store,args.archive)
        if command=='features':
            from ghosttrace.research.features import encode_features
            return encode_features(store,args.study,args.capture,args.dictionary)
        if command=='lens-import':
            from ghosttrace.research.lenses import import_lens
            return import_lens(store,args.archive)
        if command=='lens-train':
            from ghosttrace.research.extensions import train_lens_package
            return train_lens_package(store,args.archive)
        if command=='lens-apply':
            from ghosttrace.research.extensions import apply_lens_job
            from ghosttrace.research.schemas import LensApplyRequest
            spec=LensApplyRequest(capture_id=args.capture,lens_id=args.lens,decode=not args.no_decode,layers=args.layers)
            return apply_lens_job(store,{'study_id':args.study,**spec.model_dump(mode='json')})
        if command=='circuit-import':
            from ghosttrace.research.circuits import import_circuit
            document=read_json_file(args.config,dict)
            return import_circuit(store,args.study,document.get('document',document.get('graph',document)),document.get('source_metadata'))
        if command=='feature-intervene':
            from ghosttrace.research.extensions import intervene_features
            from ghosttrace.research.schemas import FeatureInterventionRequest
            spec=FeatureInterventionRequest(capture_id=args.capture,dictionary_id=args.dictionary,feature_indices=args.indices,operation=args.operation,value=args.value)
            return intervene_features(store,{'study_id':args.study,**spec.model_dump(mode='json')})
        if command=='feature-attribution':
            from ghosttrace.research.extensions import attribute_features
            from ghosttrace.research.schemas import FeatureAttributionRequest
            spec=FeatureAttributionRequest(capture_id=args.capture,dictionary_id=args.dictionary,top_k=args.top_k)
            return attribute_features(store,{'study_id':args.study,**spec.model_dump(mode='json')})
        raise EngineError('INVALID_CONFIG','Unknown research command')
    except ValidationError as error:raise EngineError('INVALID_CONFIG','Invalid research request',{'validation':str(error)[:2000]}) from error
