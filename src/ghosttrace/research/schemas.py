from __future__ import annotations
from typing import Literal,Annotated
from pydantic import BaseModel,ConfigDict,Field,model_validator

SITE_NAMES=('resid_pre','attn_in','attn_out','resid_mid','mlp_in','mlp_out','resid_post','final_norm','head_pre','head_contribution','mlp_post_gate','attention_pattern')
Site=Literal['resid_pre','attn_in','attn_out','resid_mid','mlp_in','mlp_out','resid_post','final_norm','head_pre','head_contribution','mlp_post_gate','attention_pattern']
StrictIndex=Annotated[int,Field(strict=True,ge=0,le=1_000_000)]
Position=Annotated[int,Field(strict=True,ge=-512,le=511)]
class Contract(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True,allow_inf_nan=False)
class Message(Contract):
    role:Literal['system','user','assistant']
    content:str=Field(min_length=1,max_length=8000)
class Condition(Contract):
    id:str=Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$')
    name:str=Field(min_length=1,max_length=120)
    group_id:str|None=Field(default=None,pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$')
    messages:list[Message]=Field(min_length=1,max_length=20)
class Target(Contract):
    kind:Literal['next_token_logit_diff','continuation_loglikelihood']
    positive:str|None=Field(default=None,min_length=1,max_length=2000)
    negative:str|None=Field(default=None,min_length=1,max_length=2000)
    positive_token_id:StrictIndex|None=None
    negative_token_id:StrictIndex|None=None
    positive_ids:list[StrictIndex]|None=Field(default=None,min_length=1,max_length=64)
    negative_ids:list[StrictIndex]|None=Field(default=None,min_length=1,max_length=64)
    normalization:Literal['sum','mean']='sum'
    @model_validator(mode='after')
    def complete(self):
        text=self.positive is not None or self.negative is not None
        single=self.positive_token_id is not None or self.negative_token_id is not None
        sequence=self.positive_ids is not None or self.negative_ids is not None
        if sum((text,single,sequence))!=1:raise ValueError('Provide exactly one complete text pair or pinned token-ID pair')
        if text and (self.positive is None or self.negative is None):raise ValueError('Both positive and negative text are required')
        if single and (self.kind!='next_token_logit_diff' or self.positive_token_id is None or self.negative_token_id is None):raise ValueError('Single token IDs require next_token_logit_diff and both IDs')
        if sequence and (self.kind!='continuation_loglikelihood' or self.positive_ids is None or self.negative_ids is None):raise ValueError('Candidate token ID lists require continuation_loglikelihood')
        if text and self.positive==self.negative or single and self.positive_token_id==self.negative_token_id or sequence and self.positive_ids==self.negative_ids:raise ValueError('The positive and negative targets must differ')
        return self
class CaptureProfile(Contract):
    layers:list[Annotated[int,Field(strict=True,ge=0,le=127)]]=Field(default_factory=list,max_length=32)
    positions:list[Position]=Field(default_factory=lambda:[-1],min_length=1,max_length=32)
    sites:list[Site]=Field(default_factory=lambda:['resid_pre','attn_in','attn_out','resid_mid','mlp_in','mlp_out','resid_post','final_norm','head_pre','head_contribution','mlp_post_gate'],min_length=1,max_length=12)
    attention:bool=False
    top_k:int=Field(default=8,ge=1,le=32)
    max_artifact_bytes:int=Field(default=200_000_000,ge=1000,le=2_000_000_000)
    @model_validator(mode='after')
    def unique(self):
        for key in ('layers','positions','sites'):
            value=getattr(self,key)
            if len(set(value))!=len(value):raise ValueError(f'{key} cannot contain duplicates')
        if 'attention_pattern' in self.sites and not self.attention:raise ValueError('Attention pattern capture requires explicit attention=true')
        return self
class StudyCreate(Contract):
    schema_version:Literal['2.0']='2.0'
    name:str=Field(min_length=1,max_length=200)
    question:str=Field(min_length=1,max_length=4000)
    hypothesis:str=Field(min_length=1,max_length=4000)
    model_profile_id:str=Field(min_length=1,max_length=200)
    conditions:list[Condition]=Field(min_length=1,max_length=12)
    target:Target
    capture:CaptureProfile=Field(default_factory=CaptureProfile)
    max_input_tokens:int=Field(default=512,ge=16,le=512)
    order_seed:int=Field(default=73,ge=0,le=2**32-1)
    @model_validator(mode='after')
    def unique(self):
        ids=[c.id for c in self.conditions]
        if len(ids)!=len(set(ids)):raise ValueError('Condition IDs must be unique')
        if sum(len(m.content) for c in self.conditions for m in c.messages)>100_000:raise ValueError('Study messages exceed 100,000 characters')
        return self
class CaptureRequest(Contract):
    condition_ids:list[str]|None=Field(default=None,min_length=1,max_length=12)
class Alignment(Contract):
    kind:Literal['exact_tokens','declared_semantic']='exact_tokens'
    note:str|None=Field(default=None,max_length=4000)
    @model_validator(mode='after')
    def declared(self):
        if self.kind=='declared_semantic' and (self.note is None or len(self.note.strip())<10):raise ValueError('A declared semantic correspondence needs an explicit explanation')
        return self
class InterventionRequest(Contract):
    kind:Literal['exact','attribution','path']='exact'
    donor_condition_id:str=Field(min_length=1,max_length=64)
    recipient_condition_id:str=Field(min_length=1,max_length=64)
    same_condition_condition_id:str|None=Field(default=None,min_length=1,max_length=64)
    site:Site='resid_post'
    layer:int=Field(ge=0,le=127)
    positions:list[Position]=Field(default_factory=lambda:[-1],min_length=1,max_length=32)
    operation:Literal['patch','zero_ablate','mean_ablate','norm_matched_random']='patch'
    head:int|None=Field(default=None,ge=0,le=255)
    channel:int|None=Field(default=None,ge=0,le=65535)
    alignment:Alignment=Field(default_factory=Alignment)
    destination_site:Site|None=None
    destination_layer:int|None=Field(default=None,ge=0,le=127)
    destination_head:int|None=Field(default=None,ge=0,le=255)
    destination_channel:int|None=Field(default=None,ge=0,le=65535)
    @model_validator(mode='after')
    def path(self):
        if len(set(self.positions))!=len(self.positions):raise ValueError('Positions cannot contain duplicates')
        if self.kind=='path' and (self.destination_site is None or self.destination_layer is None):raise ValueError('A path intervention requires a destination site and layer')
        if self.kind=='path' and self.destination_layer is not None and self.destination_layer<=self.layer:raise ValueError('A path receiver must be in a strictly later block')
        if self.kind!='exact' and self.operation!='patch':raise ValueError('Attribution and path studies currently compare patch operations only')
        return self
class HypothesisCreate(Contract):
    question:str=Field(min_length=1,max_length=4000)
    prior_hypothesis:str=Field(min_length=1,max_length=4000)
    protocol:str=Field(min_length=1,max_length=8000)
    expected_effect:str=Field(min_length=1,max_length=4000)
    actual_result:str=Field(default='',max_length=8000)
    alternative_explanations:list[str]=Field(default_factory=list,max_length=30)
    control_outcomes:str=Field(default='',max_length=8000)
    status:Literal['supported','refuted','unresolved']='unresolved'
    source_capture_ids:list[str]=Field(default_factory=list,max_length=100)
    source_intervention_ids:list[str]=Field(default_factory=list,max_length=100)
class ResearchExportRequest(Contract):
    include_arrays:bool=True
class DictionaryMetadata(Contract):
    schema_version:Literal['2.0-dictionary']='2.0-dictionary'
    name:str=Field(min_length=1,max_length=200)
    kind:Literal['sae','transcoder']
    model_repo_id:str=Field(min_length=1,max_length=200)
    model_revision:str=Field(pattern=r'^[a-f0-9]{40}$')
    dictionary_revision:str=Field(min_length=1,max_length=200)
    license:str=Field(min_length=1,max_length=200)
    input_site:Site
    output_site:Site
    layer:int=Field(ge=0,le=127)
    input_width:int=Field(ge=1,le=65536)
    output_width:int=Field(ge=1,le=65536)
    feature_count:int=Field(ge=1,le=65536)
    dtype:Literal['float32']='float32'
    preprocessing:Literal['none','center_decoder_bias']='center_decoder_bias'
    encoder_activation:Literal['relu','identity','jumprelu','topk']='relu'
    top_k:int|None=Field(default=None,ge=1,le=65536)
    normalization_state:str=Field(min_length=1,max_length=200)
    input_normalization_state:str|None=Field(default=None,min_length=1,max_length=200)
    output_normalization_state:str|None=Field(default=None,min_length=1,max_length=200)
    fidelity_evidence:str=Field(min_length=10,max_length=8000)
    measured_reconstruction_error:float|None=Field(default=None,ge=0)
    measured_behavioral_fidelity:float|None=None
class FeatureRequest(Contract):
    capture_id:str
    dictionary_id:str
class FeatureAnnotation(Contract):
    feature_index:int=Field(ge=0,le=65535)
    label:str=Field(min_length=1,max_length=200)
    interpretation:str=Field(min_length=1,max_length=8000)
    evidence_capture_ids:list[str]=Field(default_factory=list,max_length=100)
    test_results:str=Field(min_length=1,max_length=8000)


class LensApplyRequest(Contract):
    capture_id:str
    lens_id:str
    decode:bool=True
    layers:list[Annotated[int,Field(strict=True,ge=0,le=127)]]|None=Field(default=None,min_length=1,max_length=32)
class CircuitImportRequest(Contract):
    document:dict
    source_metadata:dict|None=None
class FeatureInterventionRequest(Contract):
    capture_id:str
    dictionary_id:str
    feature_indices:list[Annotated[int,Field(strict=True,ge=0,le=65535)]]=Field(min_length=1,max_length=32)
    operation:Literal['ablate','scale','set']='ablate'
    value:float=Field(default=0,ge=-1000,le=1000)
    @model_validator(mode='after')
    def unique(self):
        if len(set(self.feature_indices))!=len(self.feature_indices):raise ValueError('Feature indices cannot repeat')
        return self

class FeatureAttributionRequest(Contract):
    capture_id:str
    dictionary_id:str
    top_k:int=Field(default=64,ge=1,le=256)
