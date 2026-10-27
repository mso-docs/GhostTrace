"""OpenAPI-visible, strict experiment DTOs; engine validation remains authoritative."""
from __future__ import annotations
from typing import Literal,Annotated
from pydantic import BaseModel,ConfigDict,Field,model_validator
from ghosttrace.config import validate_config

class Contract(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True,allow_inf_nan=False)

class CaptureConfigSchema(Contract):
    site:Literal['residual_post_block_pre_final_norm']='residual_post_block_pre_final_norm'
    layers:Literal['all']='all'
    anchors:list[Literal['last_scenario_token','last_prompt_token']]=Field(default_factory=lambda:['last_scenario_token','last_prompt_token'],min_length=2,max_length=2)
    continuation_tokens:Literal[16]=16
    storage_dtype:Literal['float32']='float32'
    use_cache:Literal[False]=False

class GenerationConfigSchema(Contract):
    do_sample:Literal[False]=False
    max_new_tokens:int=Field(default=96,ge=1,le=96)
    stop:Literal['model_eos']='model_eos'

class LimitsConfigSchema(Contract):
    max_input_tokens:int=Field(default=512,ge=32,le=4096)
    max_total_tokens:int=Field(default=640,ge=33,le=8192)
    batch_size:Literal[1]=1
    max_artifact_bytes:int=Field(default=2_000_000_000,ge=1,le=20_000_000_000)

class AnalysisConfigSchema(Contract):
    primary_layer:Literal['middle_block']='middle_block'
    primary_metric:Literal['normalized_l2']='normalized_l2'
    bootstrap_draws:Literal[2000]=2000
    bootstrap_seed:int=Field(default=20261027,ge=0,le=2**32-1)

class ExperimentConfigSchema(Contract):
    schema_version:Literal['1.0']
    name:str=Field(min_length=1,max_length=500)
    model_profile:str=Field(min_length=1,max_length=500)
    dataset:str=Field(min_length=1,max_length=500)
    split_policy:Literal['family-heldout-v1']='family-heldout-v1'
    conditions:list[Literal['neutral','remorseful','unremorseful']]=Field(min_length=3,max_length=3)
    templates:list[Literal['A','B']]=Field(min_length=1,max_length=2)
    modes:list[Literal['fixed_continuation','greedy_generation']]=Field(min_length=1,max_length=2)
    capture:CaptureConfigSchema=Field(default_factory=CaptureConfigSchema)
    generation:GenerationConfigSchema=Field(default_factory=GenerationConfigSchema)
    limits:LimitsConfigSchema=Field(default_factory=LimitsConfigSchema)
    analysis:AnalysisConfigSchema=Field(default_factory=AnalysisConfigSchema)
    order_seed:int=Field(default=73,ge=0,le=2**32-1)
    preregistration:str|None=Field(default=None,min_length=1,max_length=1000)

    @model_validator(mode='before')
    @classmethod
    def engine_contract(cls,value):
        # Reject coercions (e.g. 0 as false or bool as int) before Pydantic hydration.
        validate_config(value)
        return value
