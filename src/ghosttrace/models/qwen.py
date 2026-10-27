"""Direct Transformers adapter. Imports are lazy; never downloads during replay."""
from __future__ import annotations
import gc
import hashlib
import json
import platform
import importlib.metadata
import random
import time
from pathlib import Path
from ghosttrace.config import EngineError, file_digest, digest
from ghosttrace.models.profiles import get_profile

class QwenAdapter:
    def __init__(self):
        self.model = self.tokenizer = self.blocks = None
        self.metadata = None

    def load(self, profile):
        try:
            import torch
            import numpy as np
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import transformers
        except ImportError as e:
            raise EngineError('MODEL_NOT_READY','Install GhostTrace model dependencies under Python 3.11 before loading a model',{'dependency':str(e)})
        revision = profile.get('revision','')
        if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
            raise EngineError('INVALID_CONFIG','Model revision must be a resolved 40-character immutable commit')
        tokenizer_revision=profile.get('tokenizer_revision',revision)
        if len(tokenizer_revision)!=40 or any(c not in '0123456789abcdef' for c in tokenizer_revision):raise EngineError('INVALID_CONFIG','Tokenizer revision must be immutable')
        device = profile.get('device','cpu')
        dtype_name = profile.get('compute_dtype','float32')
        if dtype_name not in ('float32','float16','bfloat16'):
            raise EngineError('INVALID_CONFIG','Unsupported compute dtype')
        dtype = getattr(torch,dtype_name)
        frozen_threads=profile.get('threads')
        if 'threads' in profile and (type(frozen_threads) is not int or not 1<=frozen_threads<=1024):
            raise EngineError('INVALID_CONFIG','Validated threads must be an integer in 1..1024')
        try:
            # A validated profile is an explicit execution setting, not a request
            # to inherit a new worker process's possibly different CPU default.
            if frozen_threads is not None:torch.set_num_threads(frozen_threads)
            self.tokenizer = AutoTokenizer.from_pretrained(profile['repo_id'], revision=profile.get('tokenizer_revision',revision), trust_remote_code=False, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(profile['repo_id'],revision=revision,torch_dtype=dtype,trust_remote_code=False,use_safetensors=True,local_files_only=True,attn_implementation='eager').to(device)
            self.model.eval()
            self.blocks = self.model.model.layers
            cfg = self.model.config
            if len(self.blocks) != cfg.num_hidden_layers or cfg.hidden_size < 1:
                raise EngineError('UNSUPPORTED_CAPTURE','Qwen block graph does not match resolved configuration')
            if not all(hasattr(x,'register_forward_hook') for x in self.blocks):
                raise EngineError('UNSUPPORTED_CAPTURE','Missing residual block hooks')
            from huggingface_hub import snapshot_download
            snapshot=Path(snapshot_download(profile['repo_id'],revision=revision,local_files_only=True))
            weight_files=sorted(snapshot.glob('*.safetensors'))
            if not weight_files:raise EngineError('MODEL_NOT_READY','Safetensors weight inventory is unavailable')
            weight_hashes={p.name:file_digest(p) for p in weight_files}
            tokenizer_hashes={p.name:file_digest(p) for p in snapshot.glob('*') if p.is_file() and (p.name.startswith('tokenizer') or p.name in ('vocab.json','merges.txt'))}
            from ghosttrace.provenance import code_provenance
            self.metadata = {**profile,'code_provenance':code_provenance(),'use_sliding_window':bool(cfg.use_sliding_window),'upstream_warnings':['Transformers 4.51.3 eager constructor emits a sliding-window warning even with use_sliding_window=false; inspected attention path keeps sliding_window=None.'] if cfg.sliding_window and not cfg.use_sliding_window else [],'package_versions':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'python_version':platform.python_version(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'interop_threads':torch.get_num_interop_threads(),'weight_file_hashes':weight_hashes,'tokenizer_file_hashes':tokenizer_hashes,'layers':len(self.blocks),'hidden_size':cfg.hidden_size,'query_heads':int(cfg.num_attention_heads),'kv_heads':int(cfg.num_key_value_heads),'head_dim':int(getattr(cfg,'head_dim',cfg.hidden_size//cfg.num_attention_heads)),'intermediate_size':int(cfg.intermediate_size),'config_hash':digest(cfg.to_dict()),'chat_template':self.tokenizer.chat_template,'chat_template_hash':digest(self.tokenizer.chat_template),'storage_dtype':'float32','quantization':None,'attention_backend':'eager','torch_version':torch.__version__,'transformers_version':transformers.__version__,'numpy_version':np.__version__,'os':platform.platform(),'device':device,'compute_dtype':dtype_name,'threads':torch.get_num_threads(),'capabilities':['generation','residual_capture','continuation_scoring']}
        except EngineError:
            self.unload()
            raise
        except Exception as e:
            self.unload()
            raise EngineError('MODEL_NOT_READY',str(e)) from e
        return self.describe()

    def render(self,messages):
        return self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

    def tokenize(self,text):
        return self.tokenizer.encode(text,add_special_tokens=False)

    def describe(self):
        return self.metadata

    def _input(self, token_ids):
        import torch
        if len(token_ids) > self.model.config.max_position_embeddings:
            raise EngineError('CONTEXT_LIMIT','Input exceeds model context window')
        return torch.tensor([token_ids],dtype=torch.long,device=next(self.model.parameters()).device)

    def seed(self,seed=73):
        import torch
        import numpy as np
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
            if self.metadata is not None:self.metadata['deterministic_algorithms']=True
        except RuntimeError as e:
            raise EngineError('INVALID_CONFIG','Deterministic algorithms unavailable',{'reason':str(e)})

    def generate(self,token_ids,settings,seed=73):
        import torch
        self.seed(seed)
        start = time.monotonic()
        from transformers import GenerationConfig
        # Do not inherit remote sampling defaults (temperature/top-p/top-k) into canonical greedy decoding.
        config = GenerationConfig.from_model_config(self.model.config)
        config.do_sample = False
        config.max_new_tokens = settings.get('max_new_tokens',96)
        config.pad_token_id = self.tokenizer.eos_token_id
        config.use_cache = True
        inputs = self._input(token_ids)
        with torch.inference_mode():
            result = self.model.generate(inputs,attention_mask=torch.ones_like(inputs),generation_config=config)
        generated = result[0,len(token_ids):].tolist()
        eos_ids = config.eos_token_id if isinstance(config.eos_token_id,list) else [config.eos_token_id]
        eos = bool(generated) and generated[-1] in eos_ids
        return {'generated_ids':generated,'text':self.tokenizer.decode(generated,skip_special_tokens=True),'stop_reason':'eos' if eos else 'max_new_tokens','truncated':not bool(eos),'generated_token_count':len(generated),'word_count':len(self.tokenizer.decode(generated,skip_special_tokens=True).split()),'decoding':{'do_sample':False,'max_new_tokens':config.max_new_tokens,'eos_token_id':config.eos_token_id,'temperature':None,'top_p':None},'seconds':time.monotonic()-start}

    def capture(self,token_ids,sites,positions):
        import torch
        import numpy as np
        from ghosttrace.config import SITE
        if sites != SITE:
            raise EngineError('UNSUPPORTED_CAPTURE','Unsupported capture site')
        if not positions or min(positions)<0 or max(positions)>=len(token_ids):
            raise EngineError('INVALID_CONFIG','Token positions out of bounds')
        slices, handles = {}, []
        def hook_for(index):
            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output,tuple) else output
                if hidden.ndim != 3 or hidden.shape[-1] != self.model.config.hidden_size:
                    raise EngineError('UNSUPPORTED_CAPTURE','Unexpected block output tensor shape')
                slices[index] = hidden[0,positions,:].detach().float().cpu().numpy().copy()
            return hook
        try:
            for i, block in enumerate(self.blocks):
                handles.append(block.register_forward_hook(hook_for(i)))
            with torch.inference_mode():
                self.model(input_ids=self._input(token_ids),use_cache=False,output_attentions=False,output_hidden_states=False)
        finally:
            for handle in handles:
                handle.remove()
        if len(slices) != len(self.blocks):
            raise EngineError('UNSUPPORTED_CAPTURE','Not all blocks were captured')
        return np.stack([slices[i] for i in range(len(self.blocks))])

    def score_continuation(self,prefix,continuation):
        import torch
        if not prefix or not continuation:raise EngineError('INVALID_CONFIG','Continuation scoring requires nonempty prefix and continuation')
        full = prefix + continuation
        with torch.inference_mode():
            logits = self.model(input_ids=self._input(full),use_cache=False).logits[0]
            logprobs = logits[len(prefix)-1:len(full)-1].float().log_softmax(-1)
            labels = torch.tensor(continuation,device=logprobs.device)
            return float(-logprobs.gather(1,labels[:,None]).mean().cpu())

    def unload(self):
        self.model = self.tokenizer = self.blocks = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except ImportError:
            pass
