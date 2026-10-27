"""Opt-in real-model acceptance. No downloads; immutable local weights required."""
import os
from unittest.mock import patch
import numpy as np
import pytest
from ghosttrace.config import SITE
from ghosttrace.models.qwen import QwenAdapter
from ghosttrace.models.profiles import get_profile
from ghosttrace.engine.render import messages,align_continuation

pytestmark = pytest.mark.skipif(os.environ.get('GHOSTTRACE_REAL_MODEL')!='1', reason='Opt-in: set GHOSTTRACE_REAL_MODEL=1 with locally downloaded immutable weights')

@pytest.fixture(scope='module')
def adapter():
    revision=os.environ.get('GHOSTTRACE_MODEL_REVISION')
    if not revision:pytest.fail('GHOSTTRACE_MODEL_REVISION is required; real tests never resolve main or download')
    import torch
    original_threads=torch.get_num_threads()
    a=QwenAdapter()
    try:
        # Simulate an API/worker process whose host default differs from doctor.
        torch.set_num_threads(3)
        a.load({**get_profile('qwen25-05b-cpu'),'revision':revision,'tokenizer_revision':revision,'device':'cpu','compute_dtype':'float32','threads':2})
        assert torch.get_num_threads()==2
        assert a.describe()['threads']==2
        a.seed(73)
        yield a
    finally:
        import weakref,gc
        model_reference=weakref.ref(a.model) if a.model is not None else None
        blocks_reference=weakref.ref(a.blocks) if a.blocks is not None else None
        try:
            a.unload();gc.collect()
            assert a.model is None and a.blocks is None and a.tokenizer is None
            assert model_reference is None or model_reference() is None
            assert blocks_reference is None or blocks_reference() is None
        finally:torch.set_num_threads(original_threads)

@pytest.fixture(scope='module')
def aligned(adapter):
    scenario={'text':'You damaged a borrowed notebook. Respond in two sentences.'}
    continuation='I will describe the sequence of events and explain the available next steps so we can decide how to proceed.'
    expected=None;results=[]
    for template in ('A','B'):
        for condition in ('neutral','remorseful','unremorseful'):
            rendered=adapter.render(messages(scenario,condition,template))
            aligned=align_continuation(adapter.tokenizer,rendered,continuation,expected)
            expected=aligned['continuation_ids'];results.append(aligned)
    return results

def test_real_alignment_across_conditions_and_templates(aligned):
    assert len(aligned[0]['continuation_ids'])>=16
    assert all(a['continuation_ids']==aligned[0]['continuation_ids'] for a in aligned)
    assert len({a['continuation_offset'] for a in aligned})>1  # recorded position confound, not hidden

def test_real_capture_repeat_site_shape_and_hook_cleanup(adapter,aligned):
    item=aligned[0];positions=item['token_indices'];before=[len(b._forward_hooks) for b in adapter.blocks]
    a=adapter.capture(item['token_ids'],SITE,positions);b=adapter.capture(item['token_ids'],SITE,positions)
    assert a.shape==(adapter.describe()['layers'],17,adapter.describe()['hidden_size'])
    assert a.dtype==np.float32 and np.isfinite(a).all()
    assert float(np.max(np.abs(a-b)))<=1e-5
    assert np.max(np.abs(a-a))==0
    assert before==[len(b._forward_hooks) for b in adapter.blocks]
    with patch.object(adapter.blocks[1],'forward',side_effect=RuntimeError('intentional-hook-cleanup-test')):
        with pytest.raises(RuntimeError,match='intentional'):adapter.capture(item['token_ids'],SITE,positions)
    assert before==[len(b._forward_hooks) for b in adapter.blocks]

def test_real_last_block_is_before_final_norm(adapter,aligned):
    import torch
    item=aligned[0];positions=item['token_indices'];normalized=[]
    handle=adapter.model.model.norm.register_forward_hook(lambda m,i,o:normalized.append(o[0,positions,:].detach().float().cpu()))
    try:raw=adapter.capture(item['token_ids'],SITE,positions)
    finally:handle.remove()
    with torch.inference_mode():expected=adapter.model.model.norm(torch.tensor(raw[-1])).float().cpu()
    assert torch.allclose(expected,normalized[0],atol=1e-5,rtol=1e-5)
    assert not np.allclose(raw[-1],normalized[0].numpy())

def test_real_nll_uses_previous_position(adapter,aligned):
    import torch
    item=aligned[0];offset=item['continuation_offset'];prefix=item['token_ids'][:offset];continuation=item['continuation_ids'][:16]
    actual=adapter.score_continuation(prefix,continuation)
    with torch.inference_mode():
        logits=adapter.model(input_ids=adapter._input(prefix+continuation),use_cache=False).logits[0].float()
        expected=torch.stack([-logits[offset+i-1].log_softmax(-1)[token] for i,token in enumerate(continuation)]).mean().item()
    assert abs(actual-expected)<1e-6

def test_real_greedy_generation_repeat(adapter,aligned):
    item=aligned[0];ids=item['token_ids'][:item['continuation_offset']]
    a=adapter.generate(ids,{'max_new_tokens':8});b=adapter.generate(ids,{'max_new_tokens':8})
    assert a['generated_ids']==b['generated_ids']
    assert a['decoding']['do_sample'] is False and a['decoding']['temperature'] is None
