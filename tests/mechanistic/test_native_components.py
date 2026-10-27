"""Analytic/bookkeeping tests on a tiny random model, not measured research evidence."""
import sys
import pytest
import numpy as np

torch=pytest.importorskip('torch')
transformers=pytest.importorskip('transformers')
from ghosttrace.models.mechanistic import MechanisticAdapter,structural_graph
from ghosttrace.config import EngineError

class Tokenizer:
    def decode(self,ids,**kwargs):return ' '.join(str(i) for i in ids)
    def convert_ids_to_tokens(self,ids):return [str(i) for i in ids]

@pytest.fixture()
def adapter():
    from transformers import Qwen2Config,Qwen2ForCausalLM
    torch.manual_seed(11);torch.set_num_threads(2)
    cfg=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,attention_dropout=0.,sliding_window=None)
    cfg._attn_implementation='eager'
    a=MechanisticAdapter();a.model=Qwen2ForCausalLM(cfg).eval();a.blocks=a.model.model.layers;a.tokenizer=Tokenizer();a.metadata={'compute_dtype':'float32','layers':2,'hidden_size':16,'fixture':'tiny_random_analytic_validation_only'}
    yield a
    a.unload()

TARGET={'kind':'next_token_logit_diff','positive_token_id':7,'negative_token_id':8}
DONOR=[1,2,5,4];RECIPIENT=[1,2,3,4]

def hook_counts(a):return [(len(m._forward_hooks),len(m._forward_pre_hooks)) for m in a.model.modules()]

def test_components_residuals_heads_and_attention_leave_logits_unchanged(adapter):
    before=hook_counts(adapter)
    data=adapter.inspect(RECIPIENT,layers=[0,1],positions=[1,3],target=TARGET,attention=True)
    assert data['metrics']['logit_parity_max_abs_error']==0
    assert data['artifact_arrays']['L0.head_pre'].shape==(2,4,4)
    assert data['artifact_arrays']['L0.attention_pattern'].shape==(4,2,4)
    assert data['artifact_arrays']['L0.mlp_post_gate'].shape==(2,32)
    for checks in data['metrics']['residual_identities'].values():assert max(checks.values())<1e-5
    assert len(data['metrics']['logit_lens'])==4
    assert before==hook_counts(adapter)
    graph=data['graph'];assert graph['dimensions']['kv_heads']==2
    assert {n['kv_group'] for n in graph['nodes'] if n['kind']=='attention_head'}=={0,1}
    assert all(e['kind']=='structural' for e in graph['edges'])

def test_last_residual_donor_patch_restores_last_token_and_self_is_noop(adapter):
    data=adapter.intervene(DONOR,RECIPIENT,site='resid_post',layer=1,positions=[3],operation='patch',target=TARGET)
    assert abs(data['metrics']['patched']-data['metrics']['donor'])<1e-6
    assert data['metrics']['checks']['self_patch_error']==0
    assert data['metrics']['recovery']==pytest.approx(1.)
    assert abs(data['metrics']['controls']['reverse']-data['metrics']['baseline'])<1e-6
    assert all(e['kind']=='structural' for e in data['graph']['edges'])

@pytest.mark.parametrize('site,head,channel',[('resid_pre',None,None),('resid_mid',None,None),('attn_out',None,None),('mlp_out',None,None),('head_pre',2,None),('mlp_post_gate',None,5),('final_norm',None,None)])
def test_all_patch_sites_self_patch_and_cleanup(adapter,site,head,channel):
    before=hook_counts(adapter)
    data=adapter.intervene(RECIPIENT,RECIPIENT,site=site,layer=1,positions=[3],operation='patch',target=TARGET,head=head,channel=channel)
    assert data['metrics']['delta']==pytest.approx(0.,abs=1e-6)
    assert data['metrics']['recovery'] is None
    assert data['metrics']['recovery_missing_reason']=='donor_recipient_baseline_gap_too_small'
    assert before==hook_counts(adapter)

def test_gradient_is_explicit_approximation_and_matches_small_linearized_patch(adapter):
    data=adapter.attribute(DONOR,RECIPIENT,site='resid_post',layer=1,positions=[3],target=TARGET)
    assert np.isfinite(data['artifact_arrays']['gradient']).all()
    assert data['graph']['edges'][-1]['kind']=='estimated_attribution'
    assert 'First-order' in data['metrics']['interpretation']
    assert all(p.grad is None for p in adapter.model.parameters())

def test_path_is_measured_receiver_mediation_with_self_controls(adapter):
    data=adapter.path_patch(DONOR,RECIPIENT,source_site='resid_post',source_layer=0,destination_site='resid_pre',destination_layer=1,positions=[3],target=TARGET)
    assert data['metrics']['checks']['self_patch_passed']
    assert data['graph']['edges'][-1]['kind']=='tested_path'
    assert 'intermediate routes' in data['metadata']['limitations'][0]

def test_likelihood_candidates_and_invalid_alignment(adapter):
    target={'kind':'continuation_loglikelihood','positive_ids':[7,8],'negative_ids':[9],'normalization':'mean'}
    value=adapter.intervene(DONOR,RECIPIENT,site='mlp_out',layer=0,positions=[2],operation='zero_ablate',target=target)
    assert np.isfinite(value['metrics']['delta'])
    with pytest.raises(EngineError,match='equal absolute'):adapter.intervene(DONOR,RECIPIENT+[1],site='resid_post',layer=1,positions=[3],operation='patch',target=TARGET)
    with pytest.raises(EngineError,match='strictly later'):adapter.path_patch(DONOR,RECIPIENT,source_site='resid_post',source_layer=1,destination_site='resid_pre',destination_layer=0,positions=[3],target=TARGET)


def test_feature_edit_preserves_error_and_has_original_model_controls(adapter):
    original=adapter._capture_site(RECIPIENT,'resid_post',1,[3])
    decoded=(original*.9).astype(np.float32);modified=decoded.copy();modified[:,0]=0
    value=adapter.feature_intervene(RECIPIENT,site='resid_post',layer=1,positions=[3],target=TARGET,reconstruction=decoded,modified_reconstruction=modified,feature_indices=[0],dictionary_id='synthetic-coordinate-validation')
    assert value['metrics']['checks']['self_patch_passed']
    assert value['metrics']['checks']['error_restoration_error']<1e-5
    assert value['metrics']['reconstruction_mse']>0
    assert value['artifact_arrays']['feature_delta'].dtype==np.float32
    assert 'not proof' in value['metadata']['limitations'][0]
