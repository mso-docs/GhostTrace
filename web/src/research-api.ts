import {engineApi,type Job} from './api';
import {validateResearchBundle,type ResearchBundle} from './research-types';
let token='';
async function request<T>(path:string,method='GET',body?:unknown,binary=false):Promise<T>{const response=await fetch('/api/v2/research'+path,{method,headers:{...(body!==undefined?{'Content-Type':binary?'application/zip':'application/json'}:{}),...(token?{'X-GhostTrace-Token':token}:{})},...(body!==undefined?{body:binary?body as BodyInit:JSON.stringify(body)}:{})});let result:unknown;try{result=await response.json()}catch{throw Error('Research API returned a non-JSON response. Start the local v2 API.')}if(!response.ok){const r=result as {error?:{code:string;message:string};detail?:unknown};throw Error(r.error?`${r.error.code}: ${r.error.message}`:JSON.stringify(r.detail||result))}return result as T}
export const researchApi={
 connect:async()=>{const health=await engineApi.health();token=health.mutation_token||'';return health},
 studies:()=>request<{items:Record<string,unknown>[]}>('/studies'),
 create:(config:Record<string,unknown>)=>request<Record<string,unknown>&{id:string}>('/studies','POST',config),
 bundle:async(id:string):Promise<ResearchBundle>=>validateResearchBundle(await request(`/studies/${encodeURIComponent(id)}/bundle`)),
 tile:(id:string,array:string,start:number,count:number)=>request<Record<string,unknown>>(`/captures/${encodeURIComponent(id)}/tiles?array=${encodeURIComponent(array)}&channel_start=${start}&channel_count=${count}`),
 capture:(id:string)=>request<Job>(`/studies/${encodeURIComponent(id)}/captures`,'POST',{}),
 intervene:(id:string,spec:Record<string,unknown>)=>request<Job>(`/studies/${encodeURIComponent(id)}/interventions`,'POST',spec),
 uploadArtifact:(kind:'dictionaries'|'lenses'|'lens-training'|'research-bundle',file:File)=>request<Job>(kind==='dictionaries'?'/dictionaries/imports':kind==='lenses'?'/lenses/imports':kind==='lens-training'?'/lenses/train':'/imports','POST',file,true),
 projectFeatures:(id:string,capture_id:string,dictionary_id:string)=>request<Job>(`/studies/${encodeURIComponent(id)}/features`,'POST',{capture_id,dictionary_id}),
 applyLens:(id:string,capture_id:string,lens_id:string)=>request<Job>(`/studies/${encodeURIComponent(id)}/lenses/apply`,'POST',{capture_id,lens_id,decode:true}),
 featureAttribution:(id:string,capture_id:string,dictionary_id:string)=>request<Job>(`/studies/${encodeURIComponent(id)}/feature-attributions`,'POST',{capture_id,dictionary_id,top_k:64}),
 featureIntervention:(id:string,body:unknown)=>request<Job>(`/studies/${encodeURIComponent(id)}/feature-interventions`,'POST',body),
 annotateFeature:(id:string,dictionary:string,body:unknown)=>request<unknown>(`/studies/${encodeURIComponent(id)}/dictionaries/${encodeURIComponent(dictionary)}/annotations`,'POST',body),
 exportStudy:(id:string,include_arrays=true)=>request<Job>(`/studies/${encodeURIComponent(id)}/exports`,'POST',{include_arrays}),
 downloadURL:(id:string)=>'/api/v2/research/exports/'+encodeURIComponent(id)+'/download',
 importArtifact:(id:string,kind:string,body:unknown)=>request<unknown>(`/studies/${encodeURIComponent(id)}/${encodeURIComponent(kind)}`,'POST',body)
};
export async function loadResearchURL(path:string){const response=await fetch(path);if(!response.ok)throw Error('No recorded research case is bundled at this path. Run and export a real study locally first.');const text=await response.text();if(text.length>50_000_000)throw Error('Research bundle exceeds the 50 MB browser limit.');return validateResearchBundle(JSON.parse(text))}
