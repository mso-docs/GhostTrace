"""Opt-in actual native component/intervention tests. No downloads or mock traces."""
import gc
import os
import weakref
import numpy as np
import pytest
from ghosttrace.models.mechanistic import MechanisticAdapter
from ghosttrace.models.profiles import get_profile

pytestmark=pytest.mark.skipif(os.environ.get('GHOSTTRACE_REAL_MODEL')!='1',reason='Explicit real-model opt-in and cached immutable weights required')

@pytest.fixture(scope='module')
def real_case():
    revision=os.environ.get('GHOSTTRACE_MODEL_REVISION')
    if not revision:pytest.fail('Set GHOSTTRACE_MODEL_REVISION; tests never download or resolve main')
    adapter=MechanisticAdapter();adapter.load({**get_profile('qwen25-05b-cpu'),'revision':revision,'tokenizer_revision':revision,'device':'cpu','compute_dtype':'float32','threads':2});adapter.seed(73)
    prompts={}
    for name,color in [('donor','blue'),('recipient','red')]:
        prompts[name]=adapter.tokenize(adapter.render([{'role':'system','content':'You are a precise copying assistant. Answer with only the code color word, no punctuation.'},{'role':'user','content':f'The code color is {color}. Repeat the code color:'}]))
    blue=adapter.tokenize('blue');red=adapter.tokenize('red');assert len(blue)==len(red)==1
    target={'kind':'next_token_logit_diff','positive_token_id':blue[0],'negative_token_id':red[0]}
    assert len(prompts['donor'])==len(prompts['recipient'])
    yield adapter,prompts,target
    model_ref=weakref.ref(adapter.model);blocks_ref=weakref.ref(adapter.blocks);adapter.unload();gc.collect();assert model_ref() is None and blocks_ref() is None


def test_real_components_norms_heads_attention_and_logits_parity(real_case):
    adapter,prompts,target=real_case;positions=list(range(len(prompts['recipient'])-3,len(prompts['recipient'])))
    data=adapter.inspect(prompts['recipient'],layers=[0,11,23],positions=positions,target=target,attention=True)
    assert data['metrics']['logit_parity_max_abs_error']<=1e-5
    for layer in [0,11,23]:
        assert data['artifact_arrays'][f'L{layer}.head_pre'].shape==(3,14,64)
        assert data['artifact_arrays'][f'L{layer}.attention_pattern'].shape==(14,3,len(prompts['recipient']))
        assert data['artifact_arrays'][f'L{layer}.mlp_post_gate'].shape==(3,4864)
        assert max(data['metrics']['residual_identities'][str(layer)].values())<2e-5
    assert len(data['metrics']['logit_lens'])==9
    assert {c['site'] for c in data['components']}=={'resid_pre','attn_in','attn_out','resid_mid','mlp_in','mlp_out','resid_post','final_norm','head_pre','head_contribution','mlp_post_gate','attention_pattern'}
    last=data['artifact_arrays']['L23.resid_post'][-1:]
    readout=adapter.decode_residual(last,target=target)
    assert abs(readout[0]['target_logit_difference']-data['metrics']['target_score'])<1e-5


def test_real_exact_patch_positive_control_and_self_patch(real_case):
    adapter,prompts,target=real_case;last=len(prompts['recipient'])-1
    data=adapter.intervene(prompts['donor'],prompts['recipient'],site='resid_post',layer=23,positions=[last],operation='patch',target=target)
    assert data['metrics']['checks']['self_patch_passed']
    assert abs(data['metrics']['patched']-data['metrics']['donor'])<1e-5
    assert data['metrics']['recovery']==pytest.approx(1.,abs=1e-5)
    assert data['metrics']['repeat_floor']<=1e-5
    assert 'not evidence of emotion' in data['metrics']['interpretation']


def test_real_selected_head_only_replacement_and_gqa(real_case):
    adapter,prompts,target=real_case;positions=[len(prompts['recipient'])-1]
    original=adapter._capture_site(prompts['recipient'],'head_pre',11,positions)
    replacement=np.zeros((1,1,64),dtype=np.float32)
    with adapter._patch('head_pre',11,positions,replacement,head=3):modified=adapter._capture_site(prompts['recipient'],'head_pre',11,positions)
    assert np.max(np.abs(modified[:,3,:]))==0
    assert np.array_equal(original[:,:3,:],modified[:,:3,:]) and np.array_equal(original[:,4:,:],modified[:,4:,:])
    assert np.array_equal(original,adapter._capture_site(prompts['recipient'],'head_pre',11,positions))


def test_real_attribution_and_tested_receiver_path_are_separate(real_case):
    adapter,prompts,target=real_case;positions=[len(prompts['recipient'])-1]
    estimate=adapter.attribute(prompts['donor'],prompts['recipient'],site='resid_post',layer=11,positions=positions,target=target)
    assert np.isfinite(estimate['artifact_arrays']['attribution']).all()
    assert any(e['kind']=='estimated_attribution' and not e['measured'] for e in estimate['graph']['edges'])
    path=adapter.path_patch(prompts['donor'],prompts['recipient'],source_site='resid_post',source_layer=11,destination_site='mlp_in',destination_layer=23,positions=positions,target=target)
    assert path['metrics']['checks']['self_patch_passed']
    assert any(e['kind']=='tested_path' for e in path['graph']['edges'])
    assert all(p.grad is None for p in adapter.model.parameters())


def test_real_fixed_candidate_scoring_and_component_cleanup(real_case):
    adapter,prompts,target=real_case
    fixed={'kind':'continuation_loglikelihood','positive_ids':adapter.tokenize('blue.'),'negative_ids':adapter.tokenize('red.'),'normalization':'mean'}
    before=[(len(m._forward_hooks),len(m._forward_pre_hooks)) for m in adapter.model.modules()]
    data=adapter.intervene(prompts['donor'],prompts['recipient'],site='mlp_in',layer=11,positions=[len(prompts['recipient'])-1],operation='mean_ablate',target=fixed)
    assert np.isfinite(data['metrics']['patched'])
    assert data['metrics']['checks']['self_patch_error']<=1e-5
    assert before==[(len(m._forward_hooks),len(m._forward_pre_hooks)) for m in adapter.model.modules()]
