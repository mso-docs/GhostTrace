import json
import tempfile
from pathlib import Path
import pytest
from ghosttrace.config import EngineError
from ghosttrace.review import prepare, apply


def approved_packet():
    packet=prepare('datasets/synthetic-v1')
    for e in packet['entries']:e['decision']='approve'
    return packet


def test_review_is_hash_bound_complete_and_explicitly_human():
    packet=approved_packet()
    with tempfile.TemporaryDirectory() as temp:
        output=Path(temp)/'reviewed'
        with pytest.raises(EngineError,match='attestation'):apply('datasets/synthetic-v1',packet,output,'rater')
        packet['entries'][0]['content_hash']='0'*64
        with pytest.raises(EngineError,match='mismatch'):apply('datasets/synthetic-v1',packet,output,'rater',True)
        assert not output.exists()


def test_review_creates_new_version_and_never_approves_partial_packet():
    with tempfile.TemporaryDirectory() as temp:
        packet=approved_packet();packet['entries'][0]['decision']='pending'
        with pytest.raises(EngineError,match='Unapproved'):apply('datasets/synthetic-v1',packet,Path(temp)/'rejected','rater',True)
        packet=approved_packet();output=Path(temp)/'reviewed'
        manifest=apply('datasets/synthetic-v1',packet,output,'test-human',True)
        assert manifest['confirmatory_use_allowed'] is False
        assert all(json.loads(line)['review_status']=='reviewed' for line in (output/'scenarios.jsonl').read_text().splitlines())
        assert json.loads(Path('datasets/synthetic-v1/scenarios.jsonl').read_text().splitlines()[0])['review_status']=='pending_human_review'
        with pytest.raises(EngineError,match='nonexistent'):apply('datasets/synthetic-v1',packet,output,'test-human',True)


def test_review_packet_html_is_offline_and_does_not_embed_executable_content():
    from ghosttrace.review import render_html
    packet=prepare('datasets/synthetic-v1');packet['entries'][0]['content']['text']='</script><script>untrusted()</script>'
    html=render_html(packet)
    assert '</script><script>untrusted()' not in html
    assert 'https://' not in html and 'fetch(' not in html
    assert 'Approve this item' in html and 'I personally reviewed' in html


def test_configuration_rejects_silent_unsupported_and_coerced_settings():
    import copy
    from ghosttrace.config import load_config,validate_config
    config=load_config('configs/smoke.yaml')
    cases=[('templates',['A','A']),('modes',['replay_generated']),('order_seed',True),('split_policy','random-token-split')]
    for key,value in cases:
        changed=copy.deepcopy(config);changed[key]=value
        with pytest.raises(EngineError):validate_config(changed)
    for section,key,value in [('limits','batch_size',True),('limits','max_input_tokens','512'),('analysis','primary_layer',0),('capture','layers',[1]),('capture','anchors',[]),('generation','stop','arbitrary')]:
        changed=copy.deepcopy(config);changed[section][key]=value
        with pytest.raises(EngineError):validate_config(changed)
