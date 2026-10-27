import {test,expect} from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
export function researcherHelpTests(){
 test('researcher help plays its actual captioned video and chapter navigation seeks',async({page})=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await page.goto('./#help');
  await expect(page.getByRole('heading',{name:'Watch the running workbench',exact:true})).toBeVisible();
  const video=page.getByLabel('GhostTrace researcher walkthrough');
  await expect(video.locator('track[kind="captions"]')).toHaveAttribute('default','');
  await page.getByRole('button',{name:/0:45 Inspect literal activations/}).click();
  await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.currentTime)).toBeGreaterThanOrEqual(44.9);
  await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.duration)).toBeGreaterThanOrEqual(119);
  await video.evaluate(async(v:HTMLVideoElement)=>{v.muted=true;await v.play()});
  await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.currentTime)).toBeGreaterThan(45.1);
  await video.evaluate((v:HTMLVideoElement)=>v.pause());
  await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.textTracks[0]?.cues?.length||0)).toBe(16);
  await page.getByText('Read the full descriptive transcript',{exact:true}).click();
  await expect(page.getByRole('heading',{name:'0:45 · Inspect literal activations',exact:true})).toBeVisible();
  const captionResponse=await page.request.get('./help/researcher-walkthrough.en.vtt');expect(captionResponse.ok()).toBeTruthy();expect(await captionResponse.text()).toContain('WEBVTT');
  expect(errors).toEqual([]);
 });
 test('caption failures do not masquerade as video codec errors, and media recovery clears the message',async({page})=>{
  await page.goto('./#help');const video=page.getByLabel('GhostTrace researcher walkthrough');
  await video.locator('track').evaluate(track=>track.dispatchEvent(new Event('error',{bubbles:true})));
  await expect(page.getByText(/This browser could not decode/)).toHaveCount(0);
  await video.evaluate(v=>{Object.defineProperty(v,'error',{configurable:true,value:{code:3}});v.dispatchEvent(new Event('error'))});
  await expect(page.getByText(/open the app address in Firefox, Chrome or Edge/)).toBeVisible();
  await expect(page.getByRole('link',{name:'Open video in a separate tab',exact:true})).toHaveAttribute('target','_blank');
  await video.evaluate(v=>{Object.defineProperty(v,'error',{configurable:true,value:{code:2}});v.dispatchEvent(new Event('error'))});
  await expect(page.getByText(/The video could not be downloaded/)).toBeVisible();
  await expect(page.getByText(/This browser could not decode/)).toHaveCount(0);
  await video.evaluate(v=>{Reflect.deleteProperty(v,'error')});
  await page.getByRole('button',{name:'Retry video loading',exact:true}).click();
  await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.readyState)).toBeGreaterThanOrEqual(2);
  await expect(page.getByRole('button',{name:'Retry video loading',exact:true})).toHaveCount(0);
 });
 test('help launches and restarts the measured guided demo without live computation',async({page})=>{
  const api:string[]=[];page.on('request',r=>{if(/\/api\//.test(new URL(r.url()).pathname))api.push(r.url())});
  await page.goto('./#help');await page.getByRole('button',{name:'Start guided recorded demo',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Computation graph',exact:true})).toBeVisible();
  await expect(page.getByText('GUIDED INSPECTION · 1 / 6',{exact:true})).toBeVisible();
  await page.getByRole('button',{name:'Next',exact:true}).click();
  await expect(page.getByText('GUIDED INSPECTION · 2 / 6',{exact:true})).toBeVisible();
  await page.getByRole('button',{name:'Researcher getting-started guide',exact:true}).click();
  await page.getByRole('button',{name:'Start guided recorded demo',exact:true}).click();
  await expect(page.getByText('GUIDED INSPECTION · 1 / 6',{exact:true})).toBeVisible();
  await expect(page.getByRole('heading',{name:'Baseline',exact:true})).toBeVisible();expect(api).toEqual([]);
 });
 test('help search, learning checklist and mobile accessibility work in both themes',async({page})=>{
  test.setTimeout(60000);await page.setViewportSize({width:390,height:844});await page.goto('./#help');
  await expect(page.getByRole('navigation',{name:'Walkthrough video chapters'}).getByText('0:00',{exact:true})).toBeVisible();
  await page.getByLabel('Search help questions').fill('undefined');
  await expect(page.getByText('1 matching question',{exact:true})).toBeVisible();
  await page.getByText('Why are some values missing or recovery undefined?',{exact:true}).click();
  await expect(page.getByText(/Normalized recovery is undefined/)).toBeVisible();
  await page.getByLabel('Search help questions').fill('no-such-help-question');await expect(page.getByText(/No matching questions/)).toBeVisible();
  await page.getByLabel('Search help questions').fill('');
  await page.getByLabel('I can explain this step',{exact:true}).first().check();await expect(page.getByText(/1 of 4 steps checked/)).toBeVisible();
  for(const theme of ['dark','light']){
   if(theme==='light')await page.getByRole('button',{name:'Use high contrast light appearance',exact:true}).click();
   const a=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();expect(a.violations,theme).toEqual([]);
   expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),theme).toBeTruthy();
  }
  await page.screenshot({path:'artifacts/qa/help-mobile.png',fullPage:true});
 });
}
