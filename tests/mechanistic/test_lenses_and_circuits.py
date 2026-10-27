"""Analytic synthetic tests distinguish readout calibration and graph evidence."""
import copy
import unittest
import numpy as np
from ghosttrace.config import EngineError
from ghosttrace.research.lenses import fit_affine_translator,transform_hidden,fit_affine_kl
from ghosttrace.research.circuits import validate_graph,circuit_tracer_json,filter_graph


def lens_meta():return {'schema_version':'2.0-lens','name':'TEST ONLY affine','model_repo_id':'test/model','model_revision':'a'*40,'lens_revision':'fixture-v1','license':'CC0-1.0','hidden_size':2,'layers':[0],'input_site':'resid_post','normalization_state':'raw_component','transform':'affine','objective':'ridge_hidden_alignment','training_dataset_sha256':'b'*64,'training_split':'development','holdout_used_for_fit':False,'training_sample_count':4,'development_group_ids':['dev-a','dev-b'],'holdout_group_ids':['held'],'excludes_persona_fixture':True,'training_description':'Synthetic algebra fixture only, not trained model evidence.'}


class AffineLensContracts(unittest.TestCase):
    def test_ridge_identity_and_affine_translation(self):
        x=np.array([[1,0],[0,1],[2,1],[1,2],[4,3]],dtype=np.float32);mask=np.array([True,True,True,True,False]);groups=['dev-a','dev-a','dev-b','dev-b','held']
        identity=fit_affine_translator(x,x,mask,groups,.01);np.testing.assert_allclose(identity['W'],np.eye(2),atol=1e-6);np.testing.assert_allclose(identity['b'],0,atol=1e-6)
        y=x+np.array([2,-1],dtype=np.float32);fitted=fit_affine_translator(x,y,mask,groups,.01)
        arrays={'W':fitted['W'][None,:],'b':fitted['b'][None,:]};np.testing.assert_allclose(transform_hidden(x,lens_meta(),arrays,0),y,atol=1e-5)
        self.assertFalse(fitted['metrics']['heldout_used_for_fit']);self.assertFalse(fitted['metrics']['canonical_predictive_kl'])

    def test_heldout_targets_do_not_change_fitted_weights(self):
        x=np.arange(20,dtype=np.float32).reshape(10,2);mask=np.array([True]*8+[False]*2);groups=[f'dev-{i}' for i in range(8)]+['held-a','held-b']
        y=x+1;first=fit_affine_translator(x,y,mask,groups)
        changed=y.copy();changed[~mask]*=1000;second=fit_affine_translator(x,changed,mask,groups)
        np.testing.assert_array_equal(first['W'],second['W']);np.testing.assert_array_equal(first['b'],second['b']);self.assertNotEqual(first['metrics']['holdout_mse'],second['metrics']['holdout_mse'])
        groups[-1]=groups[0]
        with self.assertRaises(EngineError):fit_affine_translator(x,y,mask,groups)

    def test_predictive_kl_affine_uses_only_development_and_no_model_grad(self):
        try:import torch
        except ImportError:self.skipTest('Optional model runtime is required for predictive-KL gradient validation')
        torch.set_num_threads(2)
        decoder=torch.nn.Linear(2,3,bias=False);decoder.weight.data.copy_(torch.tensor([[1.,0.],[0.,1.],[-1.,-1.]]));decoder.weight.grad=torch.ones_like(decoder.weight)*7
        original_weight=decoder.weight.detach().clone();original_grad=decoder.weight.grad.clone()
        x=np.array([[1,0],[0,1],[1,1],[-1,1],[2,0],[0,2],[2,1],[-2,1]],dtype=np.float32);y=x+np.array([.5,-.25],dtype=np.float32);mask=np.array([True]*6+[False]*2);groups=[f'dev-{i}' for i in range(6)]+['held-a','held-b']
        first=fit_affine_kl(x,y,mask,groups,decoder,steps=30,batch_size=3,learning_rate=.05)
        altered=y.copy();altered[~mask]*=-10;second=fit_affine_kl(x,altered,mask,groups,decoder,steps=30,batch_size=3,learning_rate=.05)
        np.testing.assert_array_equal(first['W'],second['W']);np.testing.assert_array_equal(first['b'],second['b']);torch.testing.assert_close(decoder.weight,original_weight);torch.testing.assert_close(decoder.weight.grad,original_grad)
        self.assertEqual(first['transform'],'residual_affine_delta');self.assertTrue(first['metrics']['canonical_predictive_kl']);self.assertFalse(first['metrics']['model_parameters_updated']);self.assertLess(first['metrics']['development_kl'],first['metrics']['development_identity_kl'])


class CircuitEvidenceContracts(unittest.TestCase):
    def graph(self):return {'schema_version':'2.0','nodes':[{'id':'a','kind':'component'},{'id':'b','kind':'target'}],'edges':[{'source':'a','target':'b','kind':'structural'}]}
    def source(self):return {'model_repo_id':'test/model','model_revision':'a'*40,'dictionary_revision':'dictionary-v1','license':'MIT','source_revision':'pinned-source-v1'}
    def test_structural_unmeasured_is_not_numeric_zero(self):
        graph=validate_graph(self.graph());self.assertIsNone(graph['edges'][0]['value']);self.assertFalse(graph['edges'][0]['measured'])
        filtered=filter_graph(graph,minimum_absolute_value=.1);self.assertEqual(filtered['hidden_edge_count'],1);self.assertEqual(filtered['unmeasured_edges_in_source'],1)

    def test_tested_path_requires_actual_matching_proof(self):
        graph=self.graph();graph['edges'][0].update(kind='tested_path',value=0.,evidence_record_id='proof')
        with self.assertRaises(EngineError):validate_graph(graph)
        proof={'proof':{'status':'completed','spec':{'kind':'path'},'graph':{'edges':[{'source':'a','target':'b'}]}}};validated=validate_graph(graph,proof)
        self.assertEqual(validated['edges'][0]['value'],0);self.assertTrue(validated['edges'][0]['measured']);self.assertEqual(filter_graph(validated)['edges'][0]['kind'],'tested_path')
        imported=validate_graph(graph,external=True);self.assertEqual(imported['edges'][0]['kind'],'attribution');self.assertFalse(imported['edges'][0]['measured']);self.assertTrue(imported['unsupported_fields'])

    def test_external_frontend_json_preserves_errors_and_no_causal_promotion(self):
        document={'metadata':{'scan':'external'},'nodes':[{'node_id':'0_7_0','feature':99,'layer':'0','ctx_idx':0,'feature_type':'cross layer transcoder','activation':1.,'clerp':'External tentative label','unknown_annotation':'not interpreted'},{'node_id':'0_-1_0','feature':-1,'layer':'0','ctx_idx':0,'feature_type':'mlp reconstruction error'}],'links':[{'source':'0_7_0','target':'0_-1_0','weight':-.5}],'unsupported_global':True}
        graph=circuit_tracer_json(document,self.source());self.assertEqual(graph['nodes'][0]['feature_index'],7);self.assertEqual(graph['nodes'][1]['kind'],'error');self.assertEqual(graph['edges'][0]['value'],-.5);self.assertEqual(graph['edges'][0]['evidence_type'],'attribution');self.assertFalse(graph['edges'][0]['measured']);self.assertEqual(graph['coverage']['visible_error_nodes'],1);self.assertTrue(graph['unsupported_fields'])
        self.assertEqual(graph['nodes'][0]['label_status'],'external_unvalidated_annotation')
        with self.assertRaises(EngineError):circuit_tracer_json('unsafe.pt',self.source())

    def test_duplicate_missing_endpoint_nonfinite_and_unknown_evidence_rejected(self):
        for change in ('duplicate','endpoint','nan','evidence'):
            graph=self.graph()
            if change=='duplicate':graph['nodes'].append(copy.deepcopy(graph['nodes'][0]))
            if change=='endpoint':graph['edges'][0]['target']='missing'
            if change=='nan':graph['edges'][0]['value']=float('nan')
            if change=='evidence':graph['edges'][0]['kind']='proved_moral_center'
            with self.subTest(change=change),self.assertRaises(EngineError):validate_graph(graph)

class LensArtifactWorkflow(unittest.TestCase):
    def test_supplied_fit_apply_subset_keeps_prior_artifact_immutable(self):
        import tempfile
        from pathlib import Path
        from ghosttrace.storage.db import Store
        from ghosttrace.research.storage import ResearchRepository
        from ghosttrace.research.arrays import write_arrays,read_arrays
        from ghosttrace.research.lenses import fit_lens_from_arrays,apply_lens
        with tempfile.TemporaryDirectory() as temp:
            store=Store(Path(temp)/'data');repo=ResearchRepository(store)
            x=np.arange(80,dtype=np.float32).reshape(40,2)/40;y=x+np.array([.2,-.1],dtype=np.float32);mask=np.array([True]*32+[False]*8);groups=[f'dev-{i}' for i in range(32)]+[f'held-{i}' for i in range(8)]
            meta=lens_meta();meta.update(layers=[0,1],training_scope='small_corpus_diagnostic',training_sample_count=32,development_group_ids=groups[:32],holdout_group_ids=groups[32:])
            lens=fit_lens_from_arrays(store,meta,{0:x,1:x},y,mask,groups)
            study=repo.put('study',{'model_profile':{'repo_id':'test/model','revision':'a'*40,'hidden_size':2,'layers':2}})
            artifact=write_arrays(store.root/'capture.npz',{'L0':x[:2],'L1':x[:2]});artifact['relative_path']='capture.npz'
            capture=repo.put('capture',{'study_id':study['id'],'status':'completed','artifact':artifact,'components':[{'site':'resid_post','layer':layer,'array_key':f'L{layer}','normalization_state':'raw_component','axis_names':['token','hidden']} for layer in [0,1]]},study['id'])
            all_layers=apply_lens(store,study['id'],capture['id'],lens['id']);first_path,first_sha=repo.artifact_location(all_layers)
            subset=apply_lens(store,study['id'],capture['id'],lens['id'],layers=[1]);self.assertEqual(subset['applied_layers'],[1]);self.assertEqual(subset['prediction_status'],'not_available');self.assertNotEqual(subset['artifact']['relative_path'],all_layers['artifact']['relative_path'])
            self.assertEqual(set(read_arrays(first_path,first_sha)),{'L0.translated','L1.translated'})
            with self.assertRaises(EngineError):apply_lens(store,study['id'],capture['id'],lens['id'],layers=[2])

    def test_generic_external_circuit_requires_full_pinned_provenance(self):
        import tempfile
        from pathlib import Path
        from ghosttrace.storage.db import Store
        from ghosttrace.research.storage import ResearchRepository
        from ghosttrace.research.circuits import import_circuit
        with tempfile.TemporaryDirectory() as temp:
            store=Store(Path(temp)/'data');repo=ResearchRepository(store);study=repo.put('study',{'model_profile':{'repo_id':'test/model','revision':'a'*40}})
            graph={'nodes':[{'id':'x','kind':'feature'}],'edges':[]}
            with self.assertRaises(EngineError):import_circuit(store,study['id'],graph,{'model_repo_id':'test/model','model_revision':'a'*40})
            source=CircuitEvidenceContracts().source();record=import_circuit(store,study['id'],graph,source);self.assertEqual(record['measurement_status'],'external_reported_not_recomputed')

    def test_invalid_training_provenance_rejects_before_fitting(self):
        from unittest.mock import patch
        from ghosttrace.research.lenses import fit_lens_from_arrays
        x=np.zeros((40,2),dtype=np.float32);mask=np.array([True]*32+[False]*8);groups=[f'dev-{i}' for i in range(32)]+[f'held-{i}' for i in range(8)]
        meta=lens_meta();meta.update(training_scope='small_corpus_diagnostic',training_sample_count=32,development_group_ids=['incorrect-provenance'],holdout_group_ids=groups[32:])
        with patch('ghosttrace.research.lenses.fit_affine_translator') as fit:
            with self.assertRaises(EngineError):fit_lens_from_arrays(None,meta,{0:x},x,mask,groups)
            fit.assert_not_called()
