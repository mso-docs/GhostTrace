"""Bounded native Qwen mechanistic probes; v1 capture semantics are unchanged.

Structural graphs describe the architecture. Only exact path interventions produce
edges labeled tested_path; gradient scores remain explicitly first-order estimates.
"""
from __future__ import annotations
from contextlib import contextmanager
import math
import time
from ghosttrace.config import EngineError, digest
from ghosttrace.models.qwen import QwenAdapter

SITES = ('resid_pre','attn_in','mlp_in','attn_out','resid_mid','mlp_out','resid_post','final_norm','head_pre','head_contribution','mlp_post_gate','attention_pattern')
PATCH_SITES = ('resid_pre','attn_in','mlp_in','attn_out','resid_mid','mlp_out','resid_post','final_norm','head_pre','mlp_post_gate')
MAX_CAPTURE_BYTES = 64_000_000


def structural_graph(config):
    """Literal sequential decoder DAG with explicit GQA relationships."""
    layers=int(config.num_hidden_layers);heads=int(config.num_attention_heads)
    kv=int(config.num_key_value_heads);width=int(config.hidden_size)
    nodes=[{'id':'embedding','kind':'embedding','label':'Token embedding','evidence_kind':'structural','measured':False}]
    edges=[]
    def edge(a,b,label='computation'):edges.append({'id':f'{a}->{b}','source':a,'target':b,'kind':'structural','label':label,'measured':False})
    previous='embedding'
    for layer in range(layers):
        for site in ('resid_pre','input_norm','attn_out','resid_mid','post_attention_norm','mlp_out','resid_post'):
            identifier=f'L{layer}.{site}'
            nodes.append({'id':identifier,'kind':site,'site':('attn_in' if site=='input_norm' else 'mlp_in' if site=='post_attention_norm' else site),'layer':layer,'label':site.replace('_',' '),'evidence_kind':'structural','measured':False,'hidden_size':width})
        edge(previous,f'L{layer}.resid_pre')
        for a,b,label in [('resid_pre','input_norm','normalization'),('input_norm','attn_out','attention'),('resid_pre','resid_mid','residual skip'),('attn_out','resid_mid','addition'),('resid_mid','post_attention_norm','normalization'),('post_attention_norm','mlp_out','gated MLP'),('resid_mid','resid_post','residual skip'),('mlp_out','resid_post','addition')]:edge(f'L{layer}.{a}',f'L{layer}.{b}',label)
        for head in range(heads):
            identifier=f'L{layer}.head{head}'
            nodes.append({'id':identifier,'kind':'attention_head','site':'head_pre','layer':layer,'head':head,'kv_group':head//(heads//kv),'label':f'H{head} · KV group {head//(heads//kv)}','evidence_kind':'structural','measured':False,'parent':f'L{layer}.attn_out'})
            edge(f'L{layer}.input_norm',identifier,'query head');edge(identifier,f'L{layer}.attn_out','output projection')
        previous=f'L{layer}.resid_post'
    nodes.extend([{'id':'final_norm','kind':'final_norm','site':'final_norm','layer':layers-1,'label':'Final RMSNorm','evidence_kind':'structural','measured':False},{'id':'logits','kind':'logits','label':'Unembedding / logits','evidence_kind':'structural','measured':False}])
    edge(previous,'final_norm','normalization');edge('final_norm','logits','unembedding')
    return {'schema_version':'2.0','nodes':nodes,'edges':edges,'dimensions':{'layers':layers,'hidden_size':width,'query_heads':heads,'kv_heads':kv,'head_dim':int(getattr(config,'head_dim',width//heads)),'intermediate_size':int(config.intermediate_size)},'limitations':['Structural edges are not discovered causal dependencies.','Head output projections mix coordinates; raw channels are not semantic features.'],'hidden_node_count':0,'hidden_edge_count':0}


class MechanisticAdapter(QwenAdapter):
    def _ready(self):
        if self.model is None or self.blocks is None:raise EngineError('MODEL_NOT_READY','Load a validated local model before mechanistic inspection')
        cfg=self.model.config
        for block in self.blocks:
            if not all(hasattr(block,k) for k in ('self_attn','mlp','post_attention_layernorm','input_layernorm')) or not hasattr(block.self_attn,'o_proj') or not hasattr(block.mlp,'down_proj'):
                raise EngineError('UNSUPPORTED_CAPTURE','This adapter requires the validated sequential Qwen module graph')
        return cfg

    def _selection(self,ids,layers,positions,attention=False):
        cfg=self._ready()
        if not isinstance(ids,list) or not ids or len(ids)>512 or any(type(v)is not int or v<0 or v>=cfg.vocab_size for v in ids):raise EngineError('CONTEXT_LIMIT','Research inputs require 1–512 valid unpadded token IDs')
        if not layers or len(layers)>len(self.blocks) or len(set(layers))!=len(layers) or any(type(v)is not int or not 0<=v<len(self.blocks) for v in layers):raise EngineError('INVALID_CONFIG','Select distinct valid zero-based blocks')
        if not positions or len(positions)>32 or len(set(positions))!=len(positions) or any(type(v)is not int or not 0<=v<len(ids) for v in positions):raise EngineError('INVALID_CONFIG','Select 1–32 distinct absolute token positions')
        heads=cfg.num_attention_heads;dim=cfg.hidden_size;hd=getattr(cfg,'head_dim',dim//heads)
        estimated=len(layers)*len(positions)*4*(7*dim+heads*hd+heads*dim+cfg.intermediate_size+(heads*len(ids) if attention else 0))+len(positions)*dim*4
        if estimated>MAX_CAPTURE_BYTES:raise EngineError('DISK_QUOTA','Selected component capture exceeds 64 MB; select fewer layers or tokens',{'estimated_bytes':estimated})
        return cfg,estimated

    @staticmethod
    def _hidden(output):return output[0] if isinstance(output,tuple) else output
    @staticmethod
    def _replace(output,value):return (value,*output[1:]) if isinstance(output,tuple) else value
    @staticmethod
    def _input_hidden(args,kwargs):return args[0] if args else kwargs['hidden_states']

    def _target(self,target):
        cfg=self._ready();kind=target.get('kind')
        if kind=='next_token_logit_diff':
            for key in ('positive_token_id','negative_token_id'):
                v=target.get(key)
                if type(v)is not int or not 0<=v<cfg.vocab_size:raise EngineError('INVALID_CONFIG','Target token IDs must be in the pinned vocabulary')
            if target['positive_token_id']==target['negative_token_id']:raise EngineError('INVALID_CONFIG','A logit-difference target requires distinct tokens')
        elif kind=='continuation_loglikelihood':
            for key in ('positive_ids','negative_ids'):
                v=target.get(key)
                if not isinstance(v,list) or not 1<=len(v)<=64 or any(type(i)is not int or not 0<=i<cfg.vocab_size for i in v):raise EngineError('INVALID_CONFIG','Candidate continuations require 1–64 valid token IDs')
            if target.get('normalization','sum') not in ('sum','mean'):raise EngineError('INVALID_CONFIG','Likelihood normalization must be sum or mean')
            if target['positive_ids']==target['negative_ids']:raise EngineError('INVALID_CONFIG','Candidate continuations must differ')
        else:raise EngineError('INVALID_CONFIG','Unsupported target metric')
        return target

    def _forward(self,ids,attention=False):
        import torch
        x=self._input(ids)
        return self.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False,output_attentions=attention,output_hidden_states=False)

    def _score(self,ids,target,*,grad=False):
        import torch
        target=self._target(target)
        if target['kind']=='next_token_logit_diff':
            logits=self._forward(ids).logits[0,-1].float()
            value=logits[target['positive_token_id']]-logits[target['negative_token_id']]
        else:
            scores=[]
            for key in ('positive_ids','negative_ids'):
                continuation=target[key]
                if len(ids)+len(continuation)>min(self.model.config.max_position_embeddings,576):raise EngineError('CONTEXT_LIMIT','Target continuation exceeds the research context budget')
                logits=self._forward(ids+continuation).logits[0,len(ids)-1:len(ids)+len(continuation)-1].float()
                labels=torch.tensor(continuation,device=logits.device)
                chosen=logits.log_softmax(-1).gather(1,labels[:,None]).squeeze(1)
                scores.append(chosen.mean() if target.get('normalization','sum')=='mean' else chosen.sum())
            value=scores[0]-scores[1]
        if grad:return value
        result=float(value.detach().cpu())
        if not math.isfinite(result):raise EngineError('INVALID_METRIC','Target score is nonfinite')
        return result

    def inspect(self,token_ids,*,layers,positions,target,attention=False,top_k=5):
        import numpy as np
        import torch
        cfg,estimated=self._selection(token_ids,layers,positions,attention);self._target(target)
        if type(top_k)is not int or not 1<=top_k<=32:raise EngineError('INVALID_CONFIG','Inspect top-k must be 1–32')
        arrays={};handles=[];checks={};lens=[];selected=set(layers);started=time.monotonic()
        def save(site,layer,tensor):arrays[f'L{layer}.{site}']=tensor.detach().float().cpu().numpy().copy()
        def pre(layer,site):
            def callback(module,args,kwargs):save(site,layer,self._input_hidden(args,kwargs)[0,positions,:])
            return callback
        def output(layer,site):
            def callback(module,args,out):save(site,layer,self._hidden(out)[0,positions,:])
            return callback
        def head_pre(layer):
            def callback(module,args):save('head_pre',layer,args[0][0,positions,:].reshape(len(positions),cfg.num_attention_heads,-1))
            return callback
        def mlp_pre(layer):
            def callback(module,args):save('mlp_post_gate',layer,args[0][0,positions,:])
            return callback
        def attention_out(layer):
            def callback(module,args,out):
                save('attn_out',layer,self._hidden(out)[0,positions,:])
                if attention:
                    weights=out[1] if isinstance(out,tuple) and len(out)>1 else None
                    if weights is None:raise EngineError('UNSUPPORTED_CAPTURE','Pinned attention implementation did not expose attention patterns')
                    save('attention_pattern',layer,weights[0,:,positions,:])
                    # Attention weights are auxiliary outputs; dropping their return
                    # after copying selected queries avoids retaining all N² layers.
                    return (out[0],None,*out[2:])
            return callback
        try:
            for layer,block in enumerate(self.blocks):
                if layer not in selected:continue
                handles.extend([block.register_forward_pre_hook(pre(layer,'resid_pre'),with_kwargs=True),block.register_forward_hook(output(layer,'resid_post')),block.input_layernorm.register_forward_hook(output(layer,'attn_in')),block.post_attention_layernorm.register_forward_hook(output(layer,'mlp_in')),block.self_attn.register_forward_hook(attention_out(layer)),block.post_attention_layernorm.register_forward_pre_hook(pre(layer,'resid_mid'),with_kwargs=True),block.mlp.register_forward_hook(output(layer,'mlp_out')),block.self_attn.o_proj.register_forward_pre_hook(head_pre(layer)),block.mlp.down_proj.register_forward_pre_hook(mlp_pre(layer))])
            handles.append(self.model.model.norm.register_forward_hook(output(len(self.blocks)-1,'final_norm')))
            with torch.inference_mode():captured=self._forward(token_ids,attention).logits[0,-1].float().cpu()
        finally:
            for handle in handles:handle.remove()
        with torch.inference_mode():
            baseline=self._forward(token_ids).logits[0,-1].float().cpu()
            score=self._score(token_ids,target)
            for layer in layers:
                raw=torch.as_tensor(arrays[f'L{layer}.resid_post'],device=next(self.model.parameters()).device,dtype=next(self.model.parameters()).dtype)
                readout=self.model.lm_head(self.model.model.norm(raw)).float()
                # Vocabulary-wide distributions never leave the worker.
                for i,pos in enumerate(positions):
                    logits=readout[i];logprobs=logits.log_softmax(-1);top=logprobs.topk(min(top_k,len(logprobs)))
                    row={'layer':layer,'token_position':pos,'normalization':'model_final_rmsnorm_then_unembedding','exploratory':True,'top_tokens':[{'token_id':int(t),'token':self.tokenizer.decode([int(t)]),'log_probability':float(v),'probability':float(v.exp())} for t,v in zip(top.indices,top.values)],'entropy':float(-(logprobs.exp()*logprobs).sum())}
                    if target['kind']=='next_token_logit_diff':
                        p=target['positive_token_id'];n=target['negative_token_id']
                        row.update(target_logit=float(logits[p]),target_log_probability=float(logprobs[p]),target_rank=int((logits>logits[p]).sum())+1,negative_logit=float(logits[n]),target_logit_difference=float(logits[p]-logits[n]))
                    lens.append(row)
        parity=float((captured-baseline).abs().max());tolerance=1e-5 if self.metadata['compute_dtype']=='float32' else 5e-3
        if parity>tolerance:raise EngineError('CAPTURE_PARITY_FAILED','Instrumentation changed logits beyond the measured tolerance',{'max_abs_error':parity,'tolerance':tolerance})
        for layer in layers:
            pre_=arrays[f'L{layer}.resid_pre'];mid=arrays[f'L{layer}.resid_mid'];post=arrays[f'L{layer}.resid_post']
            a=arrays[f'L{layer}.attn_out'];m=arrays[f'L{layer}.mlp_out']
            first=float(np.max(np.abs(mid-(pre_+a))));second=float(np.max(np.abs(post-(mid+m))))
            checks[str(layer)]={'attention_addition_max_abs_error':first,'mlp_addition_max_abs_error':second}
            if max(first,second)>tolerance:raise EngineError('CAPTURE_PARITY_FAILED','Residual addition bookkeeping failed',checks[str(layer)])
            weight=self.blocks[layer].self_attn.o_proj.weight.detach().float().cpu().numpy().reshape(cfg.hidden_size,cfg.num_attention_heads,-1)
            contributions=np.einsum('thd,ohd->tho',arrays[f'L{layer}.head_pre'],weight,optimize=True).astype(np.float32)
            arrays[f'L{layer}.head_contribution']=contributions
            bias=self.blocks[layer].self_attn.o_proj.bias
            reconstructed=contributions.sum(axis=1)+(bias.detach().float().cpu().numpy() if bias is not None else 0)
            error=float(np.max(np.abs(reconstructed-a)));checks[str(layer)]['head_reconstruction_max_abs_error']=error
            if error>max(tolerance,2e-5):raise EngineError('CAPTURE_PARITY_FAILED','Head output projection reconstruction failed',{'max_abs_error':error})
        components=[]
        for key,array in arrays.items():
            layer_string,site=key.split('.',1);layer=int(layer_string[1:])
            axes=['token','head','head_dim'] if site=='head_pre' else ['token','head','hidden'] if site=='head_contribution' else ['head','query_token','key_token'] if site=='attention_pattern' else ['token','channel'] if site=='mlp_post_gate' else ['token','hidden']
            components.append({'id':key,'array_key':key,'site':site,'layer':layer,'shape':list(array.shape),'axis_names':axes,'token_indices':positions,'dtype':'float32','normalization_state':'after_final_rmsnorm' if site=='final_norm' else 'after_input_rmsnorm' if site=='attn_in' else 'after_post_attention_rmsnorm' if site=='mlp_in' else 'raw_component','omitted_values':'No channel values omitted in stored selected slices; unselected layers/tokens were not captured.','summary':{'min':float(array.min()),'max':float(array.max()),'mean':float(array.mean()),'std':float(array.std())}})
        graph=structural_graph(cfg)
        available={f'L{c["layer"]}.{c["site"]}':c for c in components}
        for node in graph['nodes']:
            key=f'L{node.get("layer")}.{node.get("site")}'
            if key in available:node.update(measured=True,activation_artifact_key=key,evidence_kind='observed_activation')
        return {'metadata':{**self.describe(),'schema_version':'2.0','capture_version':'native-qwen-components-v2','top_k':top_k,'input_ids':token_ids,'visible_tokens':self.tokenizer.convert_ids_to_tokens(token_ids),'positions':positions,'selected_layers':layers,'estimated_capture_bytes':estimated,'actual_capture_bytes':sum(v.nbytes for v in arrays.values()),'capture_wall_seconds':time.monotonic()-started,'attention_return_policy':'Selected query patterns copied to CPU; auxiliary full-pattern outputs dropped before layer aggregation.' if attention else 'not_requested'},'components':components,'metrics':{'target':target,'target_score':score,'logit_lens':lens,'logit_parity_max_abs_error':parity,'tolerance':tolerance,'residual_identities':checks,'interpretation':'Intermediate unembedding readouts are exploratory, not a transcript of reasoning.'},'graph':graph,'artifact_arrays':arrays}

    def _site_module(self,site,layer):
        self._ready()
        if site not in PATCH_SITES or type(layer)is not int or not 0<=layer<len(self.blocks):raise EngineError('UNSUPPORTED_CAPTURE','Select a supported component site and zero-based block')
        block=self.blocks[layer]
        if site=='resid_pre':return block,True
        if site=='resid_mid':return block.post_attention_layernorm,True
        if site=='attn_in':return block.self_attn,True
        if site=='mlp_in':return block.mlp,True
        if site=='attn_out':return block.self_attn,False
        if site=='mlp_out':return block.mlp,False
        if site=='resid_post':return block,False
        if site=='head_pre':return block.self_attn.o_proj,True
        if site=='mlp_post_gate':return block.mlp.down_proj,True
        if layer!=len(self.blocks)-1:raise EngineError('INVALID_CONFIG','final_norm is a single site after the last block')
        return self.model.model.norm,False

    def _check_subcomponent(self,site,head,channel):
        cfg=self.model.config
        if head is not None and (site!='head_pre' or type(head)is not int or not 0<=head<cfg.num_attention_heads):raise EngineError('INVALID_CONFIG','A head selector requires head_pre and a valid query-head index')
        width=cfg.intermediate_size if site=='mlp_post_gate' else (getattr(cfg,'head_dim',cfg.hidden_size//cfg.num_attention_heads) if site=='head_pre' else cfg.hidden_size)
        if channel is not None and (type(channel)is not int or not 0<=channel<width):raise EngineError('INVALID_CONFIG','Channel index is outside the component width')

    def _view(self,hidden,site):
        if site=='head_pre':return hidden.reshape(*hidden.shape[:-1],self.model.config.num_attention_heads,-1)
        return hidden

    def _slice(self,hidden,site,positions,head=None,channel=None):
        value=self._view(hidden,site)[0,positions]
        if site=='head_pre' and head is not None:value=value[:,head:head+1,:]
        if channel is not None:value=value[...,channel:channel+1]
        return value

    def _assign(self,hidden,site,positions,replacement,head=None,channel=None):
        import torch
        clone=hidden.clone();view=self._view(clone,site)
        replacement=torch.as_tensor(replacement,device=clone.device,dtype=clone.dtype)
        for i,pos in enumerate(positions):
            if site=='head_pre' and head is not None:
                if channel is None:view[0,pos,head,:]=replacement[i,0,:]
                else:view[0,pos,head,channel]=replacement[i,0,0]
            elif channel is not None:view[0,pos,...,channel]=replacement[i,...,0]
            else:view[0,pos]=replacement[i]
        return clone

    @contextmanager
    def _patch(self,site,layer,positions,replacement,head=None,channel=None):
        module,is_pre=self._site_module(site,layer);self._check_subcomponent(site,head,channel)
        if is_pre:
            def callback(module,args,kwargs):
                value=self._assign(self._input_hidden(args,kwargs),site,positions,replacement,head,channel)
                if args:return (value,*args[1:]),kwargs
                return args,{**kwargs,'hidden_states':value}
            handle=module.register_forward_pre_hook(callback,with_kwargs=True)
        else:
            def callback(module,args,output):return self._replace(output,self._assign(self._hidden(output),site,positions,replacement,head,channel))
            handle=module.register_forward_hook(callback)
        try:yield
        finally:handle.remove()

    def _capture_site(self,ids,site,layer,positions,head=None,channel=None):
        import torch
        self._check_subcomponent(site,head,channel);module,is_pre=self._site_module(site,layer);values=[]
        def collect(hidden):values.append(self._slice(hidden,site,positions,head,channel).detach().float().cpu().numpy().copy())
        if is_pre:
            def callback(m,args,kwargs):collect(self._input_hidden(args,kwargs))
            handle=module.register_forward_pre_hook(callback,with_kwargs=True)
        else:
            def callback(m,args,output):collect(self._hidden(output))
            handle=module.register_forward_hook(callback)
        try:
            with torch.inference_mode():self._forward(ids)
        finally:handle.remove()
        if len(values)!=1:raise EngineError('UNSUPPORTED_CAPTURE','A selected component did not execute exactly once')
        return values[0]

    @staticmethod
    def _recovery(value,baseline,donor,floor):
        gap=donor-baseline
        if abs(gap)<=max(1e-6,10*floor):return None,'donor_recipient_baseline_gap_too_small'
        return (value-baseline)/gap,None

    def _node(self,site,layer,head=None,channel=None):
        if site=='final_norm':return 'final_norm'
        if site=='attn_in':return f'L{layer}.input_norm'
        if site=='mlp_in':return f'L{layer}.post_attention_norm'
        if site=='head_pre' and head is not None:return f'L{layer}.head{head}'
        if site=='mlp_post_gate' and channel is not None:return f'L{layer}.mlp_channel{channel}'
        return f'L{layer}.{site}'

    def _intervention_graph(self,site,layer,head,channel,delta,kind='measured_intervention',destination=None):
        graph=structural_graph(self.model.config);identifier=self._node(site,layer,head,channel)
        found=next((n for n in graph['nodes'] if n['id']==identifier),None)
        if found is None:
            found={'id':identifier,'site':site,'layer':layer,'head':head,'channel':channel,'kind':'selected_component','label':identifier,'evidence_kind':kind};graph['nodes'].append(found)
        found.update(intervention_effect=delta,evidence_kind=kind,measured=True)
        if destination:
            dst=self._node(destination['site'],destination['layer'],destination.get('head'),destination.get('channel'))
            if not any(n['id']==dst for n in graph['nodes']):graph['nodes'].append({'id':dst,'label':dst,'kind':'selected_component','evidence_kind':'structural'})
            graph['edges'].append({'id':f'path:{identifier}->{dst}','source':identifier,'target':dst,'kind':'tested_path','measured':True,'effect':delta,'label':'Two-stage source-to-receiver intervention','limitations':['This tests mediation through the receiver; intermediate routes are not resolved into a unique direct edge.']})
        elif kind=='estimated_attribution':graph['edges'].append({'id':f'estimate:{identifier}->logits','source':identifier,'target':'logits','kind':'estimated_attribution','measured':False,'effect':delta,'label':'First-order estimate; validate with exact patching'})
        return graph

    def intervene(self,donor_token_ids,recipient_token_ids,*,site,layer,positions,operation,target,head=None,channel=None,seed=73,same_condition_token_ids=None):
        import torch
        import numpy as np
        self._selection(donor_token_ids,[layer],positions);self._selection(recipient_token_ids,[layer],positions);self._target(target);self._check_subcomponent(site,head,channel)
        if len(donor_token_ids)!=len(recipient_token_ids):raise EngineError('TOKEN_ALIGNMENT_FAILED','Exact interventions require equal absolute sequence lengths; provide a position-matched study')
        if operation not in ('patch','zero_ablate','mean_ablate','norm_matched_random'):raise EngineError('INVALID_CONFIG','Unsupported replacement rule')
        started=time.monotonic();donor=self._capture_site(donor_token_ids,site,layer,positions,head,channel);recipient=self._capture_site(recipient_token_ids,site,layer,positions,head,channel)
        rng=np.random.default_rng(seed);random=rng.normal(size=donor.shape).astype(np.float32)
        random*=np.linalg.norm(donor,axis=-1,keepdims=True)/(np.linalg.norm(random,axis=-1,keepdims=True)+1e-8)
        replacements={'patch':donor,'zero_ablate':np.zeros_like(recipient),'mean_ablate':np.broadcast_to(donor.mean(axis=0,keepdims=True),donor.shape).copy(),'norm_matched_random':random}
        with torch.inference_mode():
            baseline=self._score(recipient_token_ids,target);donor_score=self._score(donor_token_ids,target);repeat=self._score(recipient_token_ids,target)
            def score_patch(ids,value):
                with self._patch(site,layer,positions,value,head,channel):return self._score(ids,target)
            self_patch=score_patch(recipient_token_ids,recipient);patched=score_patch(recipient_token_ids,replacements[operation]);random_score=score_patch(recipient_token_ids,random);reverse=score_patch(donor_token_ids,recipient)
            if same_condition_token_ids is not None:
                self._selection(same_condition_token_ids,[layer],positions)
                if len(same_condition_token_ids)!=len(recipient_token_ids):raise EngineError('TOKEN_ALIGNMENT_FAILED','Same-condition control must be position matched')
                same=self._capture_site(same_condition_token_ids,site,layer,positions,head,channel)
                same_score=score_patch(recipient_token_ids,same);same_scope='independently_supplied_same_condition_donor'
            else:same_score=self_patch;same_scope='recipient_exact_repeat_donor; no independent same-condition prompt supplied'
        floor=abs(repeat-baseline);tolerance=1e-5 if self.metadata['compute_dtype']=='float32' else 5e-3
        if abs(self_patch-baseline)>max(tolerance,10*floor):raise EngineError('INTERVENTION_CONTROL_FAILED','Self patch changed the target beyond the numerical floor',{'baseline':baseline,'self_patch':self_patch,'tolerance':tolerance})
        recovery,reason=self._recovery(patched,baseline,donor_score,floor)
        metrics={'target':target,'baseline':baseline,'donor':donor_score,'patched':patched,'delta':patched-baseline,'recovery':recovery,'recovery_missing_reason':reason,'repeat_floor':floor,'controls':{'untouched':baseline,'self_patch':self_patch,'same_condition':same_score,'same_condition_scope':same_scope,'reverse':reverse,'norm_matched_random':random_score},'checks':{'self_patch_error':abs(self_patch-baseline),'self_patch_passed':True,'numerical_tolerance':tolerance},'mean_baseline_scope':'mean of donor selected token positions; not a corpus mean' if operation=='mean_ablate' else None,'interpretation':'A measured effect of this intervention on this target; not evidence of emotion, a unique mechanism, or task-general necessity.'}
        metadata={'schema_version':'2.0','method':'exact_activation_intervention','site':site,'layer':layer,'positions':positions,'head':head,'channel':channel,'operation':operation,'donor_input_hash':digest(donor_token_ids),'recipient_input_hash':digest(recipient_token_ids),'replacement_sha256':digest(replacements[operation].tolist()),'seed':seed,'wall_seconds':time.monotonic()-started,'alignment_requirement':'equal absolute positions; semantic correspondence must be declared by study','dtype':self.metadata['compute_dtype']}
        return {'metadata':metadata,'metrics':metrics,'baseline_metric':baseline,'patched_metric':patched,'graph':self._intervention_graph(site,layer,head,channel,metrics['delta']),'artifact_arrays':{}}

    def attribute(self,donor_token_ids,recipient_token_ids,*,site,layer,positions,target,head=None,channel=None):
        """First-order delta×gradient, with the exact node patch as a reference."""
        import torch
        import numpy as np
        self._selection(donor_token_ids,[layer],positions);self._selection(recipient_token_ids,[layer],positions);self._target(target);self._check_subcomponent(site,head,channel)
        if len(donor_token_ids)!=len(recipient_token_ids):raise EngineError('TOKEN_ALIGNMENT_FAILED','Attribution pairs require equal absolute positions')
        donor=self._capture_site(donor_token_ids,site,layer,positions,head,channel);recipient=self._capture_site(recipient_token_ids,site,layer,positions,head,channel)
        module,is_pre=self._site_module(site,layer);leaves=[]
        def make_leaf(hidden):
            leaf=hidden.detach().requires_grad_(True);leaves.append(leaf);return leaf
        if is_pre:
            def callback(m,args,kwargs):
                leaf=make_leaf(self._input_hidden(args,kwargs))
                return ((leaf,*args[1:]),kwargs) if args else (args,{**kwargs,'hidden_states':leaf})
            handle=module.register_forward_pre_hook(callback,with_kwargs=True)
        else:
            def callback(m,args,output):return self._replace(output,make_leaf(self._hidden(output)))
            handle=module.register_forward_hook(callback)
        try:
            with torch.enable_grad():
                score=self._score(recipient_token_ids,target,grad=True)
                gradients=torch.autograd.grad(score,leaves,retain_graph=False,create_graph=False)
                selected=[self._slice(g,site,positions,head,channel).detach().float().cpu().numpy() for g in gradients]
                grad=np.stack(selected).sum(axis=0);contributions=(donor-recipient)*grad;estimate=float(contributions.sum())
                baseline=float(score.detach().cpu())
        finally:
            handle.remove();leaves.clear()
        with torch.inference_mode():
            with self._patch(site,layer,positions,donor,head,channel):exact=self._score(recipient_token_ids,target)
        return {'metadata':{'schema_version':'2.0','method':'first_order_attribution_patching','site':site,'layer':layer,'positions':positions,'head':head,'channel':channel,'approximation':'delta_activation dot gradient at recipient, not a causal edge','donor_input_hash':digest(donor_token_ids),'recipient_input_hash':digest(recipient_token_ids)},'metrics':{'target':target,'baseline':baseline,'estimated_delta':estimate,'exact_patch_delta':exact-baseline,'linearization_error':estimate-(exact-baseline),'interpretation':'First-order estimate; the exact intervention reference is reported separately.'},'graph':self._intervention_graph(site,layer,head,channel,estimate,'estimated_attribution'),'artifact_arrays':{'gradient':grad.astype(np.float32),'attribution':contributions.astype(np.float32)}}

    def path_patch(self,donor_token_ids,recipient_token_ids,*,source_site,source_layer,destination_site,destination_layer,positions,target,head=None,channel=None,destination_head=None,destination_channel=None):
        """Two-stage mediation through a specified later receiver, not a direct edge claim."""
        import torch
        self._selection(donor_token_ids,[source_layer,destination_layer],positions);self._selection(recipient_token_ids,[source_layer,destination_layer],positions);self._target(target)
        if source_layer>=destination_layer:raise EngineError('INVALID_CONFIG','Path receiver must be in a strictly later block')
        if len(donor_token_ids)!=len(recipient_token_ids):raise EngineError('TOKEN_ALIGNMENT_FAILED','Path patching requires position-matched prompts')
        donor=self._capture_site(donor_token_ids,source_site,source_layer,positions,head,channel)
        source_self=self._capture_site(recipient_token_ids,source_site,source_layer,positions,head,channel)
        baseline_receiver=self._capture_site(recipient_token_ids,destination_site,destination_layer,positions,destination_head,destination_channel)
        with self._patch(source_site,source_layer,positions,donor,head,channel):receiver=self._capture_site(recipient_token_ids,destination_site,destination_layer,positions,destination_head,destination_channel)
        with self._patch(source_site,source_layer,positions,source_self,head,channel):self_receiver=self._capture_site(recipient_token_ids,destination_site,destination_layer,positions,destination_head,destination_channel)
        with torch.inference_mode():
            baseline=self._score(recipient_token_ids,target);donor_score=self._score(donor_token_ids,target)
            with self._patch(destination_site,destination_layer,positions,receiver,destination_head,destination_channel):patched=self._score(recipient_token_ids,target)
            with self._patch(destination_site,destination_layer,positions,self_receiver,destination_head,destination_channel):self_score=self._score(recipient_token_ids,target)
            with self._patch(destination_site,destination_layer,positions,baseline_receiver,destination_head,destination_channel):receiver_self_score=self._score(recipient_token_ids,target)
        tolerance=1e-5 if self.metadata['compute_dtype']=='float32' else 5e-3
        if max(abs(self_score-baseline),abs(receiver_self_score-baseline))>tolerance:raise EngineError('INTERVENTION_CONTROL_FAILED','Path self-control failed')
        recovery,reason=self._recovery(patched,baseline,donor_score,abs(self_score-baseline))
        destination={'site':destination_site,'layer':destination_layer,'head':destination_head,'channel':destination_channel}
        return {'metadata':{'schema_version':'2.0','method':'two_stage_path_patch','source':{'site':source_site,'layer':source_layer,'head':head,'channel':channel},'destination':destination,'positions':positions,'other_readers':'recipient baseline in final pass; only the specified receiver is replaced','limitations':['The receiver activation includes intermediate routes from the source. This is a mediation experiment, not proof of one direct algebraic edge.']},'metrics':{'target':target,'baseline':baseline,'donor':donor_score,'patched':patched,'delta':patched-baseline,'recovery':recovery,'recovery_missing_reason':reason,'controls':{'untouched':baseline,'source_self_patch':self_score,'receiver_self_patch':receiver_self_score},'checks':{'self_patch_passed':True,'numerical_tolerance':tolerance}},'graph':self._intervention_graph(source_site,source_layer,head,channel,patched-baseline,destination=destination),'artifact_arrays':{}}

    def residual_logits_torch(self,vectors,normalized=False):
        """Differentiable readout for a separately trained affine lens.

        This does not run a transformer forward or infer a continuation from a
        hidden vector. It only applies the declared final norm and unembedding.
        """
        import torch
        self._ready();parameter=next(self.model.parameters())
        if isinstance(vectors,torch.Tensor):value=vectors.to(device=parameter.device,dtype=parameter.dtype)
        else:value=torch.as_tensor(vectors,device=parameter.device,dtype=parameter.dtype)
        if value.ndim!=2 or not 1<=value.shape[0]<=64 or value.shape[1]!=self.model.config.hidden_size:raise EngineError('INVALID_CONFIG','Residual readout requires 1–64 vectors with the exact model width')
        return self.model.lm_head(value if normalized else self.model.model.norm(value)).float()

    def decode_residual(self,vectors,*,target=None,topk=5,normalized=False):
        import torch
        if type(topk)is not int or not 1<=topk<=32:raise EngineError('INVALID_CONFIG','Readout top-k must be 1–32')
        if target is not None:self._target(target)
        with torch.inference_mode():
            logits=self.residual_logits_torch(vectors,normalized=normalized);logprobs=logits.log_softmax(-1);rows=[]
            for values,lp in zip(logits,logprobs):
                top=lp.topk(min(topk,len(lp)))
                row={'normalization':'already_final_normalized' if normalized else 'model_final_rmsnorm_then_unembedding','top_tokens':[{'token_id':int(t),'token':self.tokenizer.decode([int(t)]),'log_probability':float(v),'probability':float(v.exp())} for t,v in zip(top.indices,top.values)],'entropy':float(-(lp.exp()*lp).sum()),'interpretation':'Vocabulary readout of this vector; not a transcript of model reasoning.'}
                if target is not None:
                    positive=target['positive_token_id'] if target['kind']=='next_token_logit_diff' else target['positive_ids'][0]
                    negative=target['negative_token_id'] if target['kind']=='next_token_logit_diff' else target['negative_ids'][0]
                    row.update(target_token_id=positive,target_logit=float(values[positive]),target_log_probability=float(lp[positive]),target_probability=float(lp[positive].exp()),target_rank=int((values>values[positive]).sum())+1,target_logit_difference=float(values[positive]-values[negative]),target_scope='single next-token readout' if target['kind']=='next_token_logit_diff' else 'first candidate token only; not a full continuation likelihood')
                rows.append(row)
        return rows

    def feature_intervene(self,token_ids,*,site,layer,positions,target,reconstruction,modified_reconstruction,feature_indices=None,dictionary_id=None,seed=73):
        import numpy as np
        import torch
        self._selection(token_ids,[layer],positions);self._target(target)
        original=self._capture_site(token_ids,site,layer,positions)
        if original.ndim!=2:raise EngineError('UNSUPPORTED_CAPTURE','Feature decoder interventions require a token-by-hidden component site')
        decoded=np.asarray(reconstruction);modified=np.asarray(modified_reconstruction)
        if decoded.shape!=original.shape or modified.shape!=original.shape or decoded.dtype!=np.float32 or modified.dtype!=np.float32 or not np.isfinite(decoded).all() or not np.isfinite(modified).all():raise EngineError('INVALID_CONFIG','Decoded feature vectors must be finite float32 with exact selected component shape')
        delta=modified-decoded;error=original-decoded;replacement=(original+delta).astype(np.float32)
        random=np.random.default_rng(seed).normal(size=delta.shape).astype(np.float32)
        random*=np.linalg.norm(delta,axis=-1,keepdims=True)/(np.linalg.norm(random,axis=-1,keepdims=True)+1e-8)
        with torch.inference_mode():
            baseline=self._score(token_ids,target)
            def patched(value):
                with self._patch(site,layer,positions,value):return self._score(token_ids,target)
            self_score=patched(original);reconstructed=patched(decoded);restored=patched(decoded+error);edited=patched(replacement);random_score=patched(original+random)
        tolerance=1e-5 if self.metadata['compute_dtype']=='float32' else 5e-3
        if max(abs(self_score-baseline),abs(restored-baseline))>tolerance:raise EngineError('INTERVENTION_CONTROL_FAILED','Feature reconstruction-plus-error self control failed')
        return {'metadata':{'schema_version':'2.0','method':'decoded_feature_edit_in_original_model','site':site,'layer':layer,'positions':positions,'feature_indices':feature_indices or [],'dictionary_id':dictionary_id,'error_policy':'Keep original-minus-decoded error fixed; apply only modified_decode-minus-decode.','seed':seed,'limitations':['A dictionary coordinate edit is a specified vector intervention, not proof that a human feature label describes the model mechanism.']},'metrics':{'target':target,'baseline':baseline,'patched':edited,'delta':edited-baseline,'behavioral_reconstruction_delta':reconstructed-baseline,'reconstruction_mse':float(np.mean(error.astype(np.float64)**2)),'reconstruction_error_norm':float(np.linalg.norm(error)),'feature_delta_norm':float(np.linalg.norm(delta)),'controls':{'untouched':baseline,'self_patch':self_score,'reconstruction_only':reconstructed,'reconstruction_plus_saved_error':restored,'norm_matched_random_delta':random_score},'checks':{'self_patch_passed':True,'self_patch_error':abs(self_score-baseline),'error_restoration_error':abs(restored-baseline),'numerical_tolerance':tolerance},'interpretation':'Measured decoder-coordinate intervention with reconstruction error preserved. Semantic feature claims require further validation.'},'graph':self._intervention_graph(site,layer,None,None,edited-baseline),'artifact_arrays':{'feature_delta':delta.astype(np.float32),'reconstruction_error':error.astype(np.float32)}}
