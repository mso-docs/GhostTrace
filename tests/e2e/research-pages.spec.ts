import {test,expect} from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import fs from 'node:fs';
import crypto from 'node:crypto';

test('Pages subpath serves measured research without API or inference and remains usable offline',async({page,context})=>{
 const api:string[]=[],errors:string[]=[];page.on('request',r=>{if(/\/api\//.test(new URL(r.url()).pathname))api.push(r.url())});page.on('pageerror',e=>errors.push(e.message));
 await page.goto('./');await expect(page.getByRole('heading',{name:'Computation graph',exact:true})).toBeVisible();
 await expect(page.getByRole('heading',{name:'Activation microscope',exact:true})).toBeVisible();
 await expect(page.getByRole('heading',{name:'Target-output readout',exact:true})).toBeVisible();
 await context.setOffline(true);
 for(const name of ['Interventions','Features & circuits','Case study','Linked workspace']){await page.getByRole('tab',{name,exact:true}).click();await expect(page.getByRole('tab',{name,exact:true})).toHaveAttribute('aria-selected','true')}
 await page.getByRole('tab',{name:'Features & circuits',exact:true}).click();
 await expect(page.getByRole('heading',{name:'Feature attribution estimates',exact:true})).toBeVisible();
 const actual=JSON.parse(fs.readFileSync('web/public/research/demo.json','utf8')).feature_attributions[0].exact_intervention_comparisons[0];
 const comparison=page.locator('table').filter({has:page.locator('th').filter({hasText:/^Exact raw Δ$/})});
 await expect(comparison.locator('tbody tr').first().locator('td').nth(0)).toHaveText(JSON.stringify(actual.feature_indices));
 await expect(comparison.locator('tbody tr').first().locator('td').nth(1)).toHaveText(actual.estimated_delta.toPrecision(7));
 await expect(comparison.locator('tbody tr').first().locator('td').nth(2)).toHaveText(actual.exact_delta.toPrecision(7));
 await expect(comparison.locator('tbody tr').first().locator('td').nth(3)).toHaveText(actual.linearization_error.toPrecision(7));
 await page.getByRole('tab',{name:'Linked workspace',exact:true}).click();
 expect(api).toEqual([]);expect(errors).toEqual([]);
 await page.screenshot({path:'artifacts/qa/pages-research-desktop.png',fullPage:true});
});

test('Pages research selection supports keyboard and exposes literal component data',async({page})=>{
 await page.goto('./');await expect(page.getByRole('heading',{name:'Computation graph',exact:true})).toBeVisible();
 const node=page.getByRole('button',{name:/^Select .+; /}).first();await node.focus();await page.keyboard.press('Enter');await expect(node).toHaveAttribute('aria-pressed','true');
 await expect(page.getByLabel('Research capture condition')).toBeVisible();await expect(page.getByLabel('Research block',{exact:true})).toBeVisible();
 await expect(page.getByRole('group',{name:'Research prompt token strip'})).toBeVisible();
});

test('Pages research mobile views meet automated accessibility checks',async({page})=>{
 await page.setViewportSize({width:390,height:844});await page.goto('./');await expect(page.getByRole('heading',{name:'Computation graph',exact:true})).toBeVisible();
 const reports=[];
 for(const theme of ['dark','light']){
 if(theme==='light')await page.getByRole('button',{name:'Use high contrast light appearance',exact:true}).click();
 for(const name of ['Linked workspace','Interventions','Features & circuits','Case study']){await page.getByRole('tab',{name,exact:true}).click();const results=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();reports.push({name:theme+' '+name,violations:results.violations,overflow:await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)})}
 }
 fs.mkdirSync('artifacts/qa',{recursive:true});fs.writeFileSync('artifacts/qa/pages-axe.json',JSON.stringify(reports,null,2));
 expect(reports.flatMap(r=>r.violations.map(v=>({view:r.name,id:v.id,nodes:v.nodes}))), 'All four research mobile views').toEqual([]);
 expect(reports.filter(r=>r.overflow),'No document horizontal overflow').toEqual([]);
 await page.screenshot({path:'artifacts/qa/pages-research-mobile.png',fullPage:true});
});

// The same onboarding and media must work under the repository subpath.
import {researcherHelpTests} from './help-scenarios';
researcherHelpTests();


test('Pages serves the complete recorded video, captions and byte ranges from the project path',async({page,request,baseURL})=>{
 const root=new URL(baseURL!);await page.goto('./#help');
 const video=page.getByLabel('GhostTrace researcher walkthrough');
 const expected=new URL('help/researcher-walkthrough.webm',root);
 expect(new URL((await video.locator('source').getAttribute('src'))!,root).href).toBe(expected.href);
 const full=await request.get(expected.href);expect(full.status()).toBe(200);expect(full.headers()['content-type']).toMatch(/^video\/webm(?:;|$)/);
 const manifest=JSON.parse(fs.readFileSync('web/public/help/walkthrough-manifest.json','utf8'));
 const bytes=await full.body();expect(crypto.createHash('sha256').update(bytes).digest('hex')).toBe(manifest.video_sha256);
 const partial=await request.get(expected.href,{headers:{Range:'bytes=0-31'}});
 expect(partial.status()).toBe(206);expect(partial.headers()['content-range']).toBe(`bytes 0-31/${bytes.length}`);expect(await partial.body()).toEqual(bytes.subarray(0,32));
 for(const [file,type] of [['researcher-walkthrough.en.vtt','text/vtt'],['walkthrough-transcript.txt','text/plain'],['walkthrough-poster.jpg','image/jpeg']]){
  const response=await request.get(new URL('help/'+file,root).href);expect(response.status(),file).toBe(200);expect(response.headers()['content-type'],file).toContain(type);expect((await response.body()).length,file).toBeGreaterThan(0);
 }
});
