PROFILES = [
    {'id':'qwen25-15b-local-validated','repo_id':'Qwen/Qwen2.5-1.5B-Instruct','license':'Apache-2.0','layers':28,'hidden_size':1536,'estimated_weight_bytes':3_000_000_000},
    {'id':'qwen25-05b-cpu','repo_id':'Qwen/Qwen2.5-0.5B-Instruct','license':'Apache-2.0','layers':24,'hidden_size':896,'estimated_weight_bytes':2_000_000_000},
]

def get_profile(profile_id):
    from ghosttrace.config import EngineError
    for profile in PROFILES:
        if profile['id'] == profile_id:
            return dict(profile)
    raise EngineError('INVALID_CONFIG', 'Unknown model profile')
