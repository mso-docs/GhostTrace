"""Dictionary tests use declared synthetic tensors, never learned feature evidence."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from ghosttrace.config import EngineError
from ghosttrace.storage.db import Store
from ghosttrace.research.features import encode_dictionary,reconstruction_metrics,feature_delta,import_dictionary,validate_feature_pair,encode_features
from ghosttrace.research.arrays import write_arrays
from ghosttrace.research.storage import ResearchRepository


def metadata(**overrides):
    return {'schema_version':'2.0-dictionary','name':'TEST ONLY identity coordinates','kind':'sae','model_repo_id':'Qwen/Qwen2.5-0.5B-Instruct','model_revision':'a'*40,'dictionary_revision':'test-only-v1','license':'CC0-1.0','input_site':'resid_post','output_site':'resid_post','layer':1,'input_width':2,'output_width':2,'feature_count':2,'dtype':'float32','preprocessing':'none','encoder_activation':'identity','normalization_state':'raw_component','fidelity_evidence':'Synthetic test identity, not a learned SAE or semantic feature dictionary.',**overrides}

def weights():return {'W_enc':np.eye(2,dtype=np.float32),'b_enc':np.zeros(2,dtype=np.float32),'W_dec':np.eye(2,dtype=np.float32),'b_dec':np.zeros(2,dtype=np.float32)}

def package(path,meta,arrays=None,unsafe=False):
    stream=io.BytesIO();np.savez_compressed(stream,**(weights() if arrays is None else arrays));raw=stream.getvalue()
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('dictionary.json',json.dumps({'metadata':meta,'weights_sha256':hashlib.sha256(raw).hexdigest()}));archive.writestr('weights.npz',raw)
        if unsafe:archive.writestr('../bad.py','print("never executed")')


class FeatureDictionaryContracts(unittest.TestCase):
    def test_identity_reconstruction_and_preserved_error_delta(self):
        x=np.array([[1,-2],[3,4]],dtype=np.float32);z,reconstructed=encode_dictionary(x,metadata(),weights())
        np.testing.assert_array_equal(reconstructed,x);metrics=reconstruction_metrics(x,reconstructed,z)
        self.assertEqual(metrics['mse'],0);self.assertEqual(metrics['relative_l2'],0);self.assertEqual(metrics['l0_mean'],2)
        delta=feature_delta(z,weights(),0,0);np.testing.assert_array_equal(delta,np.array([[-1,0],[-3,0]],dtype=np.float32))

    def test_relu_and_bias_centering_are_explicit(self):
        x=np.array([[-1,2]],dtype=np.float32);arrays=weights();arrays['b_dec']=np.array([1,1],dtype=np.float32)
        z,reconstructed=encode_dictionary(x,metadata(encoder_activation='relu',preprocessing='center_decoder_bias'),arrays)
        np.testing.assert_array_equal(z,[[0,1]]);np.testing.assert_array_equal(reconstructed,[[1,2]])
        with self.assertRaises(EngineError):encode_dictionary(x,metadata(encoder_activation='unknown'),arrays)

    def test_jump_relu_and_topk_architectures_are_explicit(self):
        x=np.array([[.5,2],[3,-1]],dtype=np.float32);jump=weights();jump['threshold']=np.array([1,1],dtype=np.float32)
        z,_=encode_dictionary(x,metadata(encoder_activation='jumprelu'),jump);np.testing.assert_array_equal(z,[[0,2],[3,0]])
        z,_=encode_dictionary(np.array([[2,2],[1,3]],dtype=np.float32),metadata(encoder_activation='topk',top_k=1),weights());np.testing.assert_array_equal(z,[[2,0],[0,3]])
        with self.assertRaises(EngineError):encode_dictionary(x,metadata(encoder_activation='topk'),weights())
        jump['threshold'][0]=-1
        with self.assertRaises(EngineError):encode_dictionary(x,metadata(encoder_activation='jumprelu'),jump)

    def test_empty_variance_is_missing_not_perfect_score(self):
        z=np.zeros((1,2),dtype=np.float32);result=reconstruction_metrics(z,z,z)
        self.assertEqual(result['mse'],0);self.assertIsNone(result['relative_l2']);self.assertIsNone(result['explained_variance'])

    def test_import_scope_and_real_projection_compatibility(self):
        with tempfile.TemporaryDirectory() as temp:
            store=Store(Path(temp)/'data');repo=ResearchRepository(store);archive=Path(temp)/'dictionary.zip';package(archive,metadata());dictionary=import_dictionary(store,archive)
            self.assertEqual(dictionary['dictionary_classification'],'identity_coordinate_diagnostic');self.assertFalse(dictionary['learned_features'])
            study=repo.put('study',{'model_profile':{'repo_id':metadata()['model_repo_id'],'revision':'a'*40,'layers':3,'hidden_size':2}})
            component={'site':'resid_post','layer':1,'array_key':'L1.resid_post','shape':[2,2],'axis_names':['token','hidden'],'normalization_state':'raw_component','token_indices':[4,5]}
            artifact=write_arrays(store.root/'captures'/'test.npz',{'L1.resid_post':np.array([[1,2],[3,4]],dtype=np.float32)});artifact['relative_path']=str(Path(artifact['relative_path']).relative_to(store.root))
            capture=repo.put('capture',{'study_id':study['id'],'status':'completed','components':[component],'artifact':artifact,'selected_positions':[4,5]},study['id'])
            result=encode_features(store,study['id'],capture['id'],dictionary['id'])
            self.assertEqual(result['reconstruction']['mse'],0);self.assertEqual(result['behavioral_fidelity']['status'],'not_measured');self.assertEqual(result['features'][0]['test_status'],'literal_coordinate_not_learned_feature')
            other=repo.put('study',{'model_profile':{'repo_id':'different','revision':'b'*40,'layers':3,'hidden_size':2}})
            with self.assertRaises(EngineError):validate_feature_pair(store,other['id'],capture['id'],dictionary['id'])

    def test_transcoder_separate_normalization_states(self):
        with tempfile.TemporaryDirectory() as temp:
            store=Store(Path(temp)/'data');repo=ResearchRepository(store);archive=Path(temp)/'dictionary.zip'
            meta=metadata(kind='transcoder',input_site='mlp_in',output_site='mlp_out',input_normalization_state='after_post_attention_rmsnorm',output_normalization_state='raw_component')
            package(archive,meta);dictionary=import_dictionary(store,archive)
            study=repo.put('study',{'model_profile':{'repo_id':meta['model_repo_id'],'revision':'a'*40,'layers':3,'hidden_size':2}})
            components=[{'site':site,'layer':1,'array_key':site,'shape':[1,2],'axis_names':['token','hidden'],'normalization_state':state,'token_indices':[5]} for site,state in [('mlp_in','after_post_attention_rmsnorm'),('mlp_out','raw_component')]]
            capture=repo.put('capture',{'study_id':study['id'],'status':'completed','components':components,'selected_positions':[5]},study['id'])
            result=validate_feature_pair(store,study['id'],capture['id'],dictionary['id']);self.assertTrue(result['passed']);self.assertEqual(result['input_normalization_state'],'after_post_attention_rmsnorm')

    def test_unsafe_package_and_weight_shape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bad.zip';store=Store(Path(temp)/'data');package(path,metadata(),unsafe=True)
            with self.assertRaises(EngineError):import_dictionary(store,path)
            bad=weights();bad['W_enc']=np.eye(3,dtype=np.float32);package(path,metadata(),bad)
            with self.assertRaises(EngineError):import_dictionary(store,path)
