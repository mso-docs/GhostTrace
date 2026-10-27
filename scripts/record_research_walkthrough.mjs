#!/usr/bin/env node
/** Record the running measured viewer, never fabricate UI or numerical evidence.
 * Run npm run build:pages, then npm run preview:pages -- --port 4180 --strictPort first.
 * Run WALKTHROUGH_URL=http://127.0.0.1:4180 node scripts/record_research_walkthrough.mjs
 */
import {chromium} from '@playwright/test';
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
const chapters=JSON.parse(await fs.readFile('web/src/help-content.json','utf8'));
const output=path.resolve('web/public/help');await fs.mkdir(output,{recursive:true});
const temporary=path.resolve('artifacts/help-recording');await fs.mkdir(temporary,{recursive:true});
const browser=await chromium.launch({headless:true});
const context=await browser.newContext({viewport:{width:1280,height:800},deviceScaleFactor:1,reducedMotion:'reduce',recordVideo:{dir:temporary,size:{width:1280,height:800}}});
const page=await context.newPage();const failures=[];page.on('pageerror',e=>failures.push(e.message));
const start=Date.now();
async function until(seconds){const remaining=start+seconds*1000-Date.now();if(remaining>0)await new Promise(r=>setTimeout(r,remaining))}
async function section(name){await page.getByRole('heading',{name,exact:true}).evaluate(n=>n.scrollIntoView({block:'start',behavior:'instant'}));await page.evaluate(()=>window.scrollBy(0,-20))}
try{
 await page.goto(process.env.WALKTHROUGH_URL||'http://127.0.0.1:4180/Ghost-Trace/');
 await page.getByRole('heading',{name:'Computation graph',exact:true}).waitFor();
 await page.evaluate(()=>window.scrollTo(0,0));
 await page.screenshot({path:path.join(output,'walkthrough-poster.jpg'),type:'jpeg',quality:85});
 await until(15);await page.getByRole('tab',{name:'Case study',exact:true}).click();await section('The question, the test, and the limits');
 await until(30);await page.getByRole('tab',{name:'Linked workspace',exact:true}).click();await section('Computation graph');
 const node=page.getByRole('button',{name:/^Select .+; /}).filter({hasText:'resid post'}).first();if(await node.count())await node.click();
 await until(45);await section('Activation microscope');
 const coordinate=page.getByRole('group',{name:'Signed raw activation map; arrow keys navigate token and channel coordinates'}).getByRole('button').nth(1);await coordinate.click();
 await until(60);await section('Target-output readout');
 await until(75);await page.getByRole('tab',{name:'Interventions',exact:true}).click();
 await section('Controlled intervention');
 const effect=page.locator('.lab-effect-summary').first();if(await effect.count())await effect.evaluate(n=>n.scrollIntoView({block:'center',behavior:'instant'}));
 await until(90);await page.getByRole('tab',{name:'Features & circuits',exact:true}).click();await section('Feature attribution estimates');
 await until(97.5);const comparison=page.locator('table').filter({has:page.locator('th').filter({hasText:/^Exact raw Δ$/})});await comparison.evaluate(n=>n.scrollIntoView({block:'center',behavior:'instant'}));
 await until(105);await page.getByRole('tab',{name:'Case study',exact:true}).click();await section('The question, the test, and the limits');
 await until(120);
 if(failures.length)throw Error(failures.join('\n'));
 const video=page.video();await context.close();await video.saveAs(path.join(output,'researcher-walkthrough.webm'));
 const timestamp=s=>`${String(Math.floor(s/3600)).padStart(2,'0')}:${String(Math.floor(s/60)%60).padStart(2,'0')}:${String(Math.floor(s)%60).padStart(2,'0')}.${String(Math.round((s%1)*1000)).padStart(3,'0')}`;
 let vtt='WEBVTT\n\n',transcript='GhostTrace: first researcher session\n\nActual screen recording of the measured Qwen2.5-0.5B CPU/FP32 case. Silent video; English descriptive captions. No inference is executed.\n\n';
 chapters.forEach((c,i)=>{const end=chapters[i+1]?.start??120;vtt+=`${timestamp(c.start)} --> ${timestamp(c.start+7.5)}\n${c.action}\n\n${timestamp(c.start+7.5)} --> ${timestamp(end)}\n${c.meaning}\n\n`;transcript+=`${timestamp(c.start)} — ${c.title}\n${c.action}\n${c.meaning}\n\n`});
 await fs.writeFile(path.join(output,'researcher-walkthrough.en.vtt'),vtt);await fs.writeFile(path.join(output,'walkthrough-transcript.txt'),transcript);
 const hash=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
 const manifest={schema_version:'1.0-walkthrough',recorded_at:new Date().toISOString(),duration_seconds:120,audio:'none; descriptive English captions and transcript',capture:'Playwright screen recording of the actual running static research viewer',viewport:{width:1280,height:800},model_inference_executed:false,case_sha256:hash(await fs.readFile('web/public/research/demo.json')),chapter_script_sha256:hash(await fs.readFile('web/src/help-content.json')),video_sha256:hash(await fs.readFile(path.join(output,'researcher-walkthrough.webm'))),chapters:chapters.map(({id,start,title})=>({id,start,title}))};
 await fs.writeFile(path.join(output,'walkthrough-manifest.json'),JSON.stringify(manifest,null,2)+'\n');
 console.log(JSON.stringify({output,video_bytes:(await fs.stat(path.join(output,'researcher-walkthrough.webm'))).size,manifest}));
}finally{await context.close().catch(()=>{});await browser.close()}
