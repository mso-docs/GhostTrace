import {unzipSync,strFromU8} from 'fflate';
import type {Analysis,Experiment,RunUnit} from './api';
export interface ResearchData {experiment:Experiment;runs:RunUnit[];analysis:Analysis|null;source:'local'|'recorded';verification?:string;rawAvailable?:boolean;fileName?:string;testFixture?:boolean;report?:string;archiveFile?:File;legacyBinding?:boolean}
const MAX_ARCHIVE=100_000_000,MAX_EXPANDED=200_000_000;
function fail(message:string):never{throw Error('Evidence bundle: '+message)}
function validPath(name:string){return name.length<512&&!name.startsWith('/')&&!name.includes('\\')&&!name.split('/').some(p=>p==='..'||p==='.')&&!/^[a-zA-Z]:/.test(name)&&!name.includes('\0')}
async function sha(bytes:Uint8Array){const input=new Uint8Array(bytes).buffer;return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',input))).map(x=>x.toString(16).padStart(2,'0')).join('')}
function record(value:unknown):value is Record<string,unknown>{return !!value&&typeof value==='object'&&!Array.isArray(value)}
function validateFinite(value:unknown,depth=0):void{if(depth>60)fail('JSON nesting exceeds limits.');if(typeof value==='number'&&!Number.isFinite(value))fail('Nonfinite number.');if(Array.isArray(value)){if(value.length>100000)fail('Array exceeds limits.');for(const item of value)validateFinite(item,depth+1)}else if(record(value)){for(const item of Object.values(value))validateFinite(item,depth+1)}}
function canonicalJSON(value:unknown):string{
 if(Array.isArray(value))return '['+value.map(canonicalJSON).join(',')+']';
 if(record(value))return '{'+Object.keys(value).sort().map(key=>JSON.stringify(key)+':'+canonicalJSON(value[key])).join(',')+'}';
 const serialized=JSON.stringify(value);if(serialized===undefined)fail('Unsupported value in canonical JSON.');return serialized;
}
async function jsonHash(value:unknown){return sha(new TextEncoder().encode(canonicalJSON(value)))}
function sameJSON(a:unknown,b:unknown){return canonicalJSON(a)===canonicalJSON(b)}
/** Read a checksummed evidence ZIP entirely in-browser, with no model or API. */
export async function openReplay(file:File):Promise<ResearchData>{
 if(file.size>MAX_ARCHIVE)fail('Browser replay is limited to 100 MB compressed. Import larger raw bundles using the local engine.');
 let expanded=0,count=0;const paths=new Set<string>();
 const bytes=new Uint8Array(await file.arrayBuffer());
 const files=unzipSync(bytes,{filter:item=>{if(++count>10000)fail('Too many archive entries.');if(!validPath(item.name))fail('Unsafe archive path.');if(paths.has(item.name))fail('Duplicate archive path.');paths.add(item.name);expanded+=item.originalSize;if(expanded>MAX_EXPANDED||item.originalSize>MAX_EXPANDED)fail('Expanded archive exceeds 200 MB browser limit.');return true}});
 function json(name:string):unknown{const content=files[name];if(!content)fail('Missing '+name);if(content.length>10_000_000)fail(name+' exceeds the 10 MB JSON limit.');const result:unknown=JSON.parse(strFromU8(content));validateFinite(result);return result}
 const inventory=json('inventory.json');if(!record(inventory)||!record(inventory.file_inventory_with_checksums))fail('Missing checksum inventory.');
 const checksums=inventory.file_inventory_with_checksums;
 for(const [name,expected] of Object.entries(checksums)){if(!validPath(name)||typeof expected!=='string'||!/^[a-f0-9]{64}$/.test(expected))fail('Invalid inventory entry.');if(!files[name])fail('Missing inventoried file '+name);if(await sha(files[name])!==expected)fail('SHA-256 mismatch for '+name)}
 for(const name of Object.keys(files)){if(!name.endsWith('/')&&name!=='inventory.json'&&!(name in checksums))fail('Uninventoried file '+name)}
 const e=json('manifest.json'),a=json('analysis.json');
 if(!record(e)||e.schema_version!=='1.0'||typeof e.id!=='string'||typeof e.config_hash!=='string'||!Array.isArray(e.scenarios)||!record(e.model_profile)||!record(e.config))fail('Invalid experiment manifest.');
 if(!record(a)||a.schema_version!=='1.0'||a.experiment_id!==e.id||!record(a.aggregates)||!Array.isArray(a.included_units)||!Array.isArray(a.token_maps)||!Array.isArray(a.exclusions_with_reasons))fail('Analysis does not match experiment.');
 if(inventory.manifest_hash!==e.config_hash)fail('Inventory manifest hash does not match.');
 if(typeof e.created_at!=='string'||!Number.isFinite(Date.parse(e.created_at))||typeof e.model_profile.repo_id!=='string'||typeof e.config.name!=='string'||!Array.isArray(e.config.templates)||!e.config.templates.length||e.config.templates.some(t=>!['A','B'].includes(String(t))))fail('Malformed profile or config.');
 if(e.scenarios.length>200||e.scenarios.some(s=>!record(s)||typeof s.id!=='string'||typeof s.family_id!=='string'||typeof s.text!=='string'||s.text.length>2000||!['development','holdout'].includes(String(s.split))))fail('Malformed scenario records.');
 if(e.prompt_templates!==undefined&&(!Array.isArray(e.prompt_templates)||e.prompt_templates.some(t=>!record(t)||typeof t.system_text!=='string'||typeof t.family!=='string'||typeof t.condition!=='string')))fail('Malformed prompt templates.');
 if(!record(a.settings)||!['development','holdout','all'].includes(String(a.settings.split))||typeof a.settings.primary_layer!=='number'||!Array.isArray(a.limitations)||a.limitations.some(v=>typeof v!=='string'))fail('Malformed analysis settings.');
 function estimate(v:unknown){return record(v)&&typeof v.layer==='number'&&Number.isInteger(v.layer)&&typeof v.n==='number'&&Number.isInteger(v.n)&&v.n>=0&&['mean','lower','upper'].every(k=>v[k]===null||typeof v[k]==='number')}
 if(Object.keys(a.aggregates).length>32)fail('Too many contrast aggregates.');
 for(const aggregate of Object.values(a.aggregates)){if(!record(aggregate)||!estimate(aggregate.primary)||!Array.isArray(aggregate.layers)||aggregate.layers.length>128||aggregate.layers.some(v=>!estimate(v)))fail('Malformed aggregate estimates.');}

 // Rebind display metadata to the immutable experiment, not merely to a replaceable ZIP inventory.
 const hashPattern=/^[a-f0-9]{64}$/;
 for(const field of ['config_hash','dataset_hash','model_profile_hash'])if(typeof e[field]!=='string'||!hashPattern.test(e[field] as string))fail('Missing immutable '+field+'.');
 if(!record(e.template_hashes))fail('Missing frozen template-family hashes.');
 const fullPromptBinding=typeof e.prompt_templates_hash==='string';
 if(fullPromptBinding&&!hashPattern.test(e.prompt_templates_hash as string))fail('Invalid full prompt-template hash.');
 if(!files['model-profile.json'])fail('Legacy bundle lacks canonical model-profile.json. Use the local engine import for legacy verification; browser profile binding is unavailable.');
 // Compare original canonical bytes to the recorded hash. Never reserialize model float metadata for hashing.
 if(checksums['model-profile.json']!==e.model_profile_hash||!sameJSON(json('model-profile.json'),e.model_profile))fail('Canonical model-profile artifact does not match the frozen inline model profile.');
 const promptArtifact=json('prompts.json');
 if(fullPromptBinding){if(checksums['prompts.json']!==e.prompt_templates_hash||!Array.isArray(e.prompt_templates)||!sameJSON(promptArtifact,e.prompt_templates))fail('Canonical prompt artifact differs from frozen prompt templates.');}
 else if(e.prompt_templates!==undefined&&!sameJSON(promptArtifact,e.prompt_templates))fail('Legacy prompt artifact differs from inline templates.');
 for(const family of e.config.templates){const selected=Array.isArray(promptArtifact)?promptArtifact.filter(t=>record(t)&&t.family===family):record(promptArtifact)?promptArtifact[String(family)]:undefined;if(selected===undefined||await jsonHash(selected)!==e.template_hashes[String(family)])fail('Frozen template-family hash mismatch.');}
 if(await jsonHash(e.scenarios)!==e.dataset_hash)fail('Inline scenario data differs from the frozen dataset hash.');
 const controls=Array.isArray(e.control_scenarios)?e.control_scenarios:[];
 if((controls.length?await jsonHash(controls):null)!==e.factual_dataset_hash)fail('Factual-control dataset hash mismatch.');
 if(e.preregistration_hash!==null){if(typeof e.preregistration_hash!=='string'||checksums['preregistration.md']!==e.preregistration_hash)fail('Preregistration file does not match its frozen hash.');}
 const base:Record<string,unknown>={};
 for(const field of ['schema_version','config','dataset_hash','factual_dataset_hash','model_profile_hash','template_hashes','preregistration_hash']){if(!(field in e))fail('Frozen manifest base is incomplete: '+field);base[field]=e[field]}
 for(const field of ['prompt_templates_hash','human_review_hash','preregistration_receipt_hash'])if(field in e)base[field]=e[field];
 if(await jsonHash(base)!==e.config_hash)fail('Frozen manifest base config_hash mismatch.');
 for(const [field,path,inline] of [['human_review_hash','human-review.json','human_review'],['preregistration_receipt_hash','preregistration-receipt.json','preregistration_receipt']])if(field in e){if(checksums[path]!==e[field]||!sameJSON(json(path),e[inline]))fail('Human review or preregistration receipt binding mismatch.');}
 const legacyBinding=!fullPromptBinding;
 const runs:RunUnit[]=[];
 for(const name of Object.keys(files).filter(n=>/^runs\/[a-f0-9]{64}\/record\.json$/.test(n))){const r=json(name);if(!record(r)||typeof r.id!=='string'||r.experiment_id!==e.id||!['completed','failed','running'].includes(String(r.status)))fail('Invalid run record.');if(name!==`runs/${r.id}/record.json`)fail('Run ID/path mismatch.');if(typeof r.scenario_id!=='string'||typeof r.condition!=='string'||typeof r.template_id!=='string'||!['greedy_generation','fixed_continuation','replay_generated'].includes(String(r.mode)))fail('Malformed run metadata.');if(typeof r.text==='string'&&r.text.length>65536)fail('Generated text exceeds limits.');if(r.visible_tokens!==undefined&&(!Array.isArray(r.visible_tokens)||r.visible_tokens.length>4096||r.visible_tokens.some(t=>typeof t!=='string')))fail('Invalid visible tokens.');if(r.continuation_ids!==undefined&&(!Array.isArray(r.continuation_ids)||r.continuation_ids.length>4096||r.continuation_ids.some(t=>typeof t!=='number'||!Number.isInteger(t)||t<0)))fail('Invalid continuation token IDs.');if(record(r.activation_artifact)){const artifact=r.activation_artifact;
 if(artifact.site!=='residual_post_block_pre_final_norm'||artifact.normalization_state!=='before_final_model_norm'||artifact.dtype!=='float32')fail('Unsupported activation capture site, normalization state or dtype.');
 if(!Array.isArray(artifact.axis_names)||artifact.axis_names.join(',')!=='layer,selected_token,hidden')fail('Invalid activation axis names.');
 const shape=artifact.shape,pool=artifact.pool_mask,profile=e.model_profile;
 if(!Array.isArray(shape)||shape.length!==3||shape.some(v=>typeof v!=='number'||!Number.isInteger(v)||v<1)||shape[0]!==profile.layers||shape[2]!==profile.hidden_size||shape[0]>128||shape[1]>4096||typeof profile.hidden_size!=='number'||!Number.isInteger(profile.hidden_size)||profile.hidden_size<1||profile.hidden_size>16384||shape[0]*shape[1]*shape[2]*4>2_000_000_000)fail('Activation shape does not match the bounded model profile.');
 if(!Array.isArray(pool)||pool.length!==16||new Set(pool).size!==16||pool.some(v=>typeof v!=='number'||!Number.isInteger(v)||v<0||v>=shape[1]))fail('Invalid first-16 continuation pool mask.');
 if(typeof artifact.sha256!=='string'||!/^[a-f0-9]{64}$/.test(artifact.sha256))fail('Invalid activation checksum.');
 const rawName=`runs/${r.id}/activations.npz`;if(files[rawName]&&await sha(files[rawName])!==r.activation_artifact.sha256)fail('Activation checksum mismatch.')}runs.push(r as unknown as RunUnit)}
 const ids=new Set(runs.map(r=>r.id));if(ids.size!==runs.length)fail('Duplicate run identifiers.');if(a.included_units.some(id=>typeof id!=='string'||!ids.has(id)))fail('Analysis references a missing run.');
 const layerCount=e.model_profile.layers;if(typeof layerCount!=='number'||!Number.isInteger(layerCount)||layerCount<1||layerCount>128)fail('Unsupported layer count.');if(a.token_maps.length>2000)fail('Too many token maps.');
 for(const map of a.token_maps){
 if(!record(map)||typeof map.scenario_id!=='string'||typeof map.template!=='string'||typeof map.contrast!=='string'||!Array.isArray(map.pool_mask)||map.pool_mask.length!==16||!Array.isArray(map.absolute_position_difference)||!Array.isArray(map.unit_ids)||map.unit_ids.some(id=>typeof id!=='string'||!ids.has(id))||!Array.isArray(map.values)||map.values.length!==layerCount)fail('Invalid token map references.');
 for(const row of map.values){if(!Array.isArray(row)||row.length!==16||row.some(v=>v!==null&&(typeof v!=='number'||v<0)))fail('Invalid token distance map.');}
 const paired=map.unit_ids.map(id=>runs.find(r=>r.id===id));
 if(paired.length!==2||paired.some(unit=>unit?.status!=='completed'||unit.mode!=='fixed_continuation'||!unit.activation_artifact||!unit.continuation_ids||unit.continuation_ids.length<16))fail('Token map does not reference two completed aligned captures.');
 const first=paired[0]!,second=paired[1]!;
 if(first.continuation_ids!.length!==second.continuation_ids!.length||first.continuation_ids!.some((id,index)=>id!==second.continuation_ids![index]))fail('Token map pairs nonidentical continuation token IDs.');
 if(map.pool_mask.some((slot,index)=>slot!==first.activation_artifact!.pool_mask[index]))fail('Token map mask disagrees with activation metadata.');
 }

 const testFixture=a.measurement_status==='synthetic_test_fixture'||String(e.fixture_type||'').includes('synthetic')||String(e.fixture_type||'').includes('illustrative');
 if(a.measurement_status==='measured'&&testFixture)fail('Synthetic fixtures cannot be labeled measured.');
 if(a.measurement_status==='measured'&&a.included_units.length===0)fail('Measured analysis has no included units.');
 const rawAvailable=inventory.includes_raw_activations===true&&runs.some(r=>r.activation_artifact&&files[`runs/${r.id}/activations.npz`]);
 return {experiment:e as unknown as Experiment,analysis:a as unknown as Analysis,runs,source:'recorded',rawAvailable,fileName:file.name,testFixture,legacyBinding,archiveFile:file,report:files['report.md']?strFromU8(files['report.md']):undefined,verification:`${legacyBinding?'LEGACY: only selected template families are bound to the manifest; control-template binding is incomplete. ':''}Frozen manifest base, canonical model profile, dataset and ${fullPromptBinding?'complete prompt inventory':'selected template families'} verified. SHA-256 verified for ${Object.keys(checksums).length} files. ${rawAvailable?'Raw arrays included.':'Raw arrays omitted.'} Browser replay displays saved analysis; it does not recompute metrics or authenticate the experiment’s author.`};
}
export function saveJSON(name:string,value:unknown){const href=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=href;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(href),1000)}

export function downloadBlob(name:string,blob:Blob){const href=URL.createObjectURL(blob);const a=document.createElement('a');a.href=href;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(href),1000)}
