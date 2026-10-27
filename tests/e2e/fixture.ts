/** Content-addressed SYNTHETIC UI test bundle; never measured research evidence. */
import {createHash} from 'node:crypto';
import {zipSync,strToU8} from 'fflate';
function canonical(value:unknown):string{if(Array.isArray(value))return '['+value.map(canonical).join(',')+']';if(value&&typeof value==='object')return '{'+Object.keys(value).sort().map(k=>JSON.stringify(k)+':'+canonical((value as Record<string,unknown>)[k])).join(',')+'}';return JSON.stringify(value)}
const digest=(v:unknown)=>createHash('sha256').update(canonical(v)).digest('hex');
export function testBundle({corrupt=false,unsafe=false,badCapture=false,badProfile=false}={}){
 const experimentId='00000000-0000-4000-8000-000000000001',conditions=['neutral','remorseful','unremorseful'];
 const scenario={id:'test-only-01',family_id:'test_family',split:'development',text:'SYNTHETIC TEST SCENARIO. No research measurement is implied.',common_continuation:'Test-only fixed continuation.'};
 const templates=conditions.map(condition=>({id:`A-${condition}`,family:'A',condition,system_text:`TEST ONLY ${condition} instruction`}));
 const profile={id:'test-only',repo_id:'TEST ONLY synthetic adapter',device:'cpu',compute_dtype:'float32',revision:'a'.repeat(40),tokenizer_revision:'a'.repeat(40),layers:3,hidden_size:4};
 const config={schema_version:'1.0',name:'Synthetic UI test — not measured',dataset:'test-only',model_profile:'test-only',templates:['A'],conditions,modes:['fixed_continuation','greedy_generation']};
 const base={schema_version:'1.0',config,dataset_hash:digest([scenario]),factual_dataset_hash:null,model_profile_hash:digest(profile),template_hashes:{A:digest(templates)},prompt_templates_hash:digest(templates),preregistration_hash:null};
 const configHash=digest(base),runs:any[]=[],ordered:any[]=[];
 for(let conditionIndex=0;conditionIndex<conditions.length;conditionIndex++)for(const mode of ['greedy_generation','fixed_continuation']){
  const identity={scenario_id:scenario.id,condition:conditions[conditionIndex],template_id:'A',mode,control:null,seed:null};
  const unit={...identity,id:digest({experiment_hash:configHash,...identity})};ordered.push(unit);
  const ids=Array.from({length:16},(_,i)=>i+10),positions=[3,4,...Array.from({length:16},(_,i)=>i+5)];
  runs.push({...unit,schema_version:'1.0',experiment_id:experimentId,status:'completed',fixture:true,scenario,model_profile:profile,backend_fingerprint:digest(profile),text:`SYNTHETIC UI TEST RESPONSE ${conditions[conditionIndex]}. Not a model measurement.`,stop_reason:'eos',truncated:false,generated_token_count:10,token_ids:[1,2,3,4,5,...ids],continuation_ids:ids,continuation_offset:5,visible_tokens:[...Array(5).fill('prefix'),...Array.from({length:16},(_,i)=>`token${i}`)],rendered_messages:[{role:'system',content:`TEST ONLY ${conditions[conditionIndex]} instruction`}],activation_artifact:mode==='fixed_continuation'?{site:'residual_post_block_pre_final_norm',shape:[3,18,4],dtype:'float32',sha256:'a'.repeat(64),axis_names:['layer','selected_token','hidden'],layer_indices:[0,1,2],token_indices:positions,pool_mask:Array.from({length:16},(_,i)=>i+2),normalization_state:'before_final_model_norm'}:undefined});
 }
 const captures=runs.filter(r=>r.mode==='fixed_continuation');
 if(badCapture)captures[0].activation_artifact.normalization_state='after_final_model_norm';
 const primary={layer:0,mean:0,lower:0,upper:0,n:1};
 const settings={split:'development',primary_layer:0,metric_version:'normalized-l2-v1',bootstrap_draws:2000,bootstrap_seed:20261027};
 const analysis={schema_version:'1.0',id:'00000000-0000-4000-8000-000000000002',experiment_id:experimentId,measurement_status:'synthetic_test_fixture',settings,settings_hash:digest(settings),included_units:captures.map(r=>r.id),exclusions_with_reasons:[],expected_unit_count:6,observed_unit_count:6,limitations:['SYNTHETIC UI TEST: no actual experiment or model inference.'],aggregates:{'remorseful-neutral':{primary,layers:[primary,{...primary,layer:1,mean:.125,lower:.125,upper:.125},{...primary,layer:2,mean:null,lower:null,upper:null,n:0}],family_sensitivity:{test_family:{mean:0,n:1}}},'neutral-repeat':{primary,layers:[primary,{...primary,layer:1},{...primary,layer:2}],family_sensitivity:{}}},token_maps:[{scenario_id:scenario.id,template:'A',contrast:'remorseful-neutral',unit_ids:[captures[1].id,captures[0].id],pool_mask:Array.from({length:16},(_,i)=>i+2),absolute_position_difference:Array(18).fill(0),values:[Array(16).fill(0),Array.from({length:16},(_,i)=>i/100),[null,...Array(15).fill(.25)]]}]};
 const manifest={...base,id:experimentId,config_hash:configHash,fixture_type:'synthetic_ui_test_only',created_at:'2026-10-27T00:00:00Z',estimated_units:6,model_profile:badProfile?{...profile,device:'tampered-inline-profile'}:profile,scenarios:[scenario],control_scenarios:[],prompt_templates:templates,ordered_units:ordered,ordered_unit_ids:ordered.map(u=>u.id)};
 const encode=(value:unknown)=>strToU8(canonical(value));
 const files:Record<string,Uint8Array>={'manifest.json':encode(manifest),'analysis.json':encode(analysis),'model-profile.json':encode(profile),'prompts.json':encode(templates),'dataset.jsonl':strToU8(canonical(scenario)+'\n')};
 // JSON serialization removes undefined optional fields, matching actual exports.
 for(const run of runs)files[`runs/${run.id}/record.json`]=encode(JSON.parse(JSON.stringify(run)));
 if(unsafe)files['../unsafe.txt']=strToU8('unsafe');
 const checksums=Object.fromEntries(Object.entries(files).map(([name,bytes])=>[name,createHash('sha256').update(bytes).digest('hex')]));
 files['inventory.json']=encode({schema_version:'1.0',manifest_hash:configHash,includes_raw_activations:false,file_inventory_with_checksums:checksums});
 if(corrupt)files['analysis.json']=encode({...analysis,altered:true});
 return Buffer.from(zipSync(files));
}
