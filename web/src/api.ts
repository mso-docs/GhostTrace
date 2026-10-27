import type {PreflightRequest,RatingRequest,ReplayRequest,AnalysisRequest,ExportRequest,ExperimentConfigSchema,CaptureConfigSchema,GenerationConfigSchema,LimitsConfigSchema,AnalysisConfigSchema} from './generated/api-schema';
/** Same-origin contracts shared with the local GhostTrace API. */
export interface Envelope {schema_version:string;request_id?:string}
export interface Page<T> extends Envelope {items:T[];next_cursor:string|null}
export interface Health extends Envelope {version:string;api:string;worker:string;offline:boolean;replay:boolean;model_loaded:boolean;mutation_token?:string}
export interface ModelProfile {id:string;repo_id:string;license:string;layers:number;hidden_size:number;revision?:string|null;device?:string;compute_dtype?:string;preflight_passed?:boolean;download_state?:string;capabilities?:string[];generation_tokens_per_second?:number;repeat_max_abs_error?:number;repeat_tolerance?:number;peak_ram_bytes?:number;peak_vram_bytes?:number;load_and_test_seconds?:number}
export type Device='cpu'|'cuda'|'mps'; export type ComputeDtype='float32'|'float16'|'bfloat16';
export interface JobResult {resource_type:string;resource_id?:string;analysis_id?:string;export_id?:string;import_id?:string;experiment_id?:string;raw_activations_available?:boolean;recomputed?:boolean;verification?:unknown}
export interface Job extends Envelope {id:string;job_id?:string;experiment_id?:string|null;kind?:string;state:string;stage?:string;total_units:number;completed_units?:number;failed_units?:number;attempt?:number;recoverable?:boolean;error_code?:string|null;error_message?:string|null;heartbeat_at?:string|null;created_at?:string;updated_at?:string;cancel_requested?:number;result?:JobResult|null}
export interface JobEvent {seq:number;type:string;created_at:string;payload:Record<string,unknown>}
export interface ExperimentConfig extends ExperimentConfigSchema {capture:Required<CaptureConfigSchema>;generation:Required<GenerationConfigSchema>;limits:Required<LimitsConfigSchema>;analysis:Required<AnalysisConfigSchema>;preregistration:string}
export interface Scenario {id:string;family_id:string;split:string;text:string;common_continuation:string;review_status:string}
export interface PromptTemplate {id:string;family:string;condition:string;system_text:string;sha256?:string}
export interface Experiment extends Envelope {id:string;config_hash:string;created_at:string;estimated_units:number;estimated_bytes:number;warnings:string[];config:ExperimentConfig;measurement_status:string;model_profile:ModelProfile;scenarios:Scenario[];control_scenarios?:Scenario[];prompt_templates?:PromptTemplate[];preregistration_hash?:string;dataset_hash?:string}
export interface ActivationArtifact {sha256:string;site:string;shape:number[];token_indices:number[];semantic_position_labels:string[];pool_mask:number[];layer_indices:number[];dtype:string;normalization_state:string}
export interface RunUnit {id:string;experiment_id?:string;scenario_id:string;condition:string;template_id:string;mode:string;status:string;control?:string|null;split?:string;family_id?:string;generated_text?:string;output_text?:string;text?:string;stop_reason?:string;truncated?:boolean;generated_token_count?:number;seconds?:number;error?:{code:string;message:string};scenario?:Scenario;activation_artifact?:ActivationArtifact;continuation_ids?:number[];visible_tokens?:string[];token_ids?:number[];token_indices?:number[];absolute_positions?:number[];continuation_offset?:number;rendered_messages?:{role:string;content:string}[];model_profile?:ModelProfile;backend_fingerprint?:string}
export interface Estimate {layer:number;mean:number|null;lower:number|null;upper:number|null;n:number}
export interface TokenMap {scenario_id:string;template:string;contrast:string;unit_ids:string[];pool_mask:number[];absolute_position_difference:number[];values:(number|null)[][]}
export interface RatingSummary extends Envelope {rater_count:number;unique_rating_count:number;rating_record_count:number;raters:{rater_id:string;unique_response_count:number;minimum_reached:boolean;condition_template_counts:{condition:string;template:string;n:number}[]}[];reliability_available:boolean;pairwise_agreement:{raters:string[];paired_response_count:number;dimensions:Record<string,{n:number;agreement:number|null;weighted_kappa:number|null;reason:string|null}>;disagreements:unknown[]}[];limitations:string[]}
export interface BehaviorSummary {interpretation:string;lexical_version:string;conditions:Record<string,{expected_units:number;completed_units:number;missing_or_failed_units:number;metrics:Record<string,Estimate>;refusal_flags_present:number;refusal_count:number;malformed_flags_present:number;malformed_count:number}>;continuation_nll:{interpretation:string;conditions:Record<string,Estimate>};flags_limitation:string}
export interface Analysis extends Envelope {id:string;experiment_id:string;created_at?:string;measurement_status:string;settings:{split:string;primary_layer:number};included_units:string[];exclusions_with_reasons:unknown[];aggregates:Record<string,{primary:Estimate;layers:Estimate[];metrics?:Record<string,{primary:Estimate;layers:Estimate[];interpretation:string}>;family_sensitivity?:Record<string,{mean:number;n:number}>}>;token_maps:TokenMap[];limitations:string[];expected_unit_count?:number;observed_unit_count?:number;ratings?:RatingSummary;behavioral?:BehaviorSummary;reliability?:Record<string,unknown>;performance?:Record<string,unknown>}
export interface EvidenceExport extends Envelope {id:string;status:string;includes_raw_activations:boolean;manifest_hash:string;file_inventory_with_checksums?:Record<string,string>}
export interface ImportRecord extends Envelope {id:string;experiment_id:string;analysis_id:string;raw_activations_available:boolean;recomputed:boolean;verification:unknown}
export interface RatingTask {run_unit_id:string;blinded_display_id:string;response_text:string;scenario_text?:string;stop_reason?:string;truncated?:boolean;already_rated?:boolean}
export interface RatingScores {regret:number;responsibility:number;repair:number;justification_deflection:number}
export interface RatingSubmission {run_unit_id:string;rubric_version:'expressed-behavior-v1';blinded_display_id:string;rater_id:string;dimension_scores:RatingScores;refusal:boolean;malformed:boolean}
export class ApiError extends Error {constructor(public code:string,message:string,public status:number,public requestId?:string){super(message)}}
let mutationToken='';
async function request<T>(path:string,method='GET',body?:unknown,headers:Record<string,string>={},binary=false):Promise<T>{
 let response:Response;
 try {response=await fetch('/api/v1'+path,{method,headers:{...(body!==undefined?{'Content-Type':binary?'application/zip':'application/json'}:{}),...(mutationToken?{'X-GhostTrace-Token':mutationToken}:{}),...headers},...(body!==undefined?{body:binary?body as BodyInit:JSON.stringify(body)}:{})})}catch{throw new ApiError('CONNECTION_FAILED','Local API unavailable. Start ghosttrace serve and ghosttrace worker in separate terminals.',0)}
 const raw=await response.text();let value:unknown;try{value=JSON.parse(raw)}catch{throw new ApiError('INVALID_RESPONSE','The local API did not return JSON. Check the local API server or Vite /api proxy.',response.status)}
 if(!response.ok){const error=value as {error?:{code?:string;message?:string};detail?:unknown;request_id?:string};throw new ApiError(error.error?.code||'VALIDATION_ERROR',error.error?.message||(error.detail?JSON.stringify(error.detail):'Request failed'),response.status,error.request_id)}
 return value as T;
}
export const engineApi={
 health:async()=>{const h=await request<Health>('/health');mutationToken=h.mutation_token||'';return h},
 models:()=>request<Page<ModelProfile>>('/models'),
 experiments:()=>request<Page<Experiment>>('/experiments'),
 experiment:(id:string)=>request<Experiment>(`/experiments/${encodeURIComponent(id)}`),
 preflight:(profile_id:string,revision:string,device:Device,compute_dtype:ComputeDtype)=>request<Job>('/preflights','POST',{profile_id,revision,device,compute_dtype} satisfies PreflightRequest),
 freeze:(config:ExperimentConfig)=>request<Experiment>('/experiments','POST',config),
 launch:(id:string,key:string)=>request<Job>(`/experiments/${encodeURIComponent(id)}/jobs`,'POST',{}, {'Idempotency-Key':key}),
 job:(id:string)=>request<Job>(`/jobs/${encodeURIComponent(id)}`),
 cancel:(id:string)=>request<Job>(`/jobs/${encodeURIComponent(id)}/cancel`,'POST'),
 resume:(id:string)=>request<Job>(`/jobs/${encodeURIComponent(id)}/resume`,'POST'),
 events:(id:string,after=0)=>request<Envelope&{items:JobEvent[];after:number}>(`/jobs/${encodeURIComponent(id)}/events?after=${after}`),
 runs:(id:string,cursor?:string)=>request<Page<RunUnit>>(`/experiments/${encodeURIComponent(id)}/runs?limit=100${cursor?'&cursor='+encodeURIComponent(cursor):''}`),
 allRuns:async(id:string)=>{let cursor:string|undefined;const items:RunUnit[]=[];for(let page=0;page<100;page++){const result=await engineApi.runs(id,cursor);items.push(...result.items);if(!result.next_cursor)return items;cursor=result.next_cursor}throw new ApiError('PAGINATION_LIMIT','Run inventory exceeded the bounded workbench limit.',413)},
 analyze:(id:string,split:'development'|'holdout'|'all')=>request<Job>(`/experiments/${encodeURIComponent(id)}/analyses`,'POST',{split} satisfies AnalysisRequest),
 analysis:(id:string)=>request<Analysis>(`/analyses/${encodeURIComponent(id)}`),
 export:(analysis_id:string,include_activations:boolean)=>request<Job>('/exports','POST',{analysis_id,include_activations} satisfies ExportRequest),
 exportMetadata:(id:string)=>request<EvidenceExport>(`/exports/${encodeURIComponent(id)}`),
 importBundle:(file:File)=>request<Job>('/imports','POST',file,{},true),
 importRecord:(id:string)=>request<ImportRecord>(`/imports/${encodeURIComponent(id)}`),
 replay:(id:string,analysis_id?:string)=>request<Job>(`/experiments/${encodeURIComponent(id)}/replay`,'POST',{verify_metrics:true,...(analysis_id?{analysis_id}:{})} satisfies ReplayRequest),
 ratingSummary:(id:string)=>request<RatingSummary>(`/experiments/${encodeURIComponent(id)}/rating-summary`),
 ratingTasks:(id:string,rater:string)=>request<Envelope&{items:RatingTask[];eligible_response_count:number;rated_response_count:number;remaining_response_count:number;target_minimum:number;balanced_by_condition_and_template:boolean;[key:string]:unknown}>(`/experiments/${encodeURIComponent(id)}/rating-tasks?rater_id=${encodeURIComponent(rater)}`),
 ratings:(id:string)=>request<Envelope&{items:unknown[];[key:string]:unknown}>(`/experiments/${encodeURIComponent(id)}/ratings`),
 rate:(body:RatingSubmission)=>request<Envelope&{id:string}>('/ratings','POST',{...body,dimension_scores:{...body.dimension_scores}} satisfies RatingRequest),
 downloadUrl:(id:string)=>'/api/v1/exports/'+encodeURIComponent(id)+'/download'
};
export function makeConfig(profile:string,pilot:boolean):ExperimentConfig{return {schema_version:'1.0',name:pilot?'ghosttrace-balanced-pilot':'ghosttrace-smoke',model_profile:profile,dataset:pilot?'datasets/synthetic-v1':'datasets/smoke-v1',split_policy:'family-heldout-v1',conditions:['neutral','remorseful','unremorseful'],templates:pilot?['A','B']:['A'],modes:['fixed_continuation','greedy_generation'],capture:{site:'residual_post_block_pre_final_norm',layers:'all',anchors:['last_scenario_token','last_prompt_token'],continuation_tokens:16,storage_dtype:'float32',use_cache:false},generation:{do_sample:false,max_new_tokens:96,stop:'model_eos'},limits:{max_input_tokens:512,max_total_tokens:640,batch_size:1,max_artifact_bytes:2000000000},analysis:{primary_layer:'middle_block',primary_metric:'normalized_l2',bootstrap_draws:2000,bootstrap_seed:20261027},order_seed:73,preregistration:'preregistrations/pilot-v1.md'}}
