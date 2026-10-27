import {test,expect} from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import {writeFileSync} from 'node:fs';
import {testBundle} from './fixture';

test('production starts without invented measurements and local failures stay explicit',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/api/v1/health',route=>route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({error:{code:'WORKER_UNAVAILABLE',message:'Test: local API unavailable'}})}));
 await page.goto('/');
 await expect(page.getByRole('heading',{name:'Start with a careful question.'})).toBeVisible();
 await expect(page.getByText('SYNTHETIC UI TEST RESPONSE')).toHaveCount(0);
 await expect(page.getByRole('img',{name:/Illustrative/})).toHaveCount(0);
 await page.getByRole('button',{name:'Experiments',exact:true}).click();
 await page.getByRole('button',{name:'Connect local API',exact:true}).click();
 await expect(page.getByRole('alert')).toContainText(/API|JSON|connect|unavailable/i);
 expect(errors).toEqual([]);
});

test('keyboard reaches navigation and narrow layout keeps the main flow usable',async({page})=>{
 await page.setViewportSize({width:390,height:844});await page.goto('/');
 await page.keyboard.press('Tab');await expect(page.locator(':focus')).toHaveCount(1);
 await page.getByRole('button',{name:'Method & exports',exact:true}).focus();
 await page.keyboard.press('Enter');await expect(page.getByRole('heading',{name:/Look inside/})).toBeVisible();
 await page.getByRole('button',{name:'Workbench',exact:true}).focus();await page.keyboard.press('Enter');
 await expect(page.getByRole('heading',{name:'Start with a careful question.'})).toBeVisible();
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();
 await page.screenshot({path:'artifacts/qa/mobile-empty-workbench.png',fullPage:true});
});

test('checksummed synthetic UI fixture replays offline, compares tokens and preserves zero/missing values',async({page,context})=>{
 await page.goto('/');await context.setOffline(true);
 await page.getByLabel('Open evidence ZIP').setInputFiles({name:'synthetic-ui-test-only.zip',mimeType:'application/zip',buffer:testBundle()});
 await expect(page.getByText('Synthetic test fixture — NOT MEASURED',{exact:true})).toBeVisible();
 await expect(page.getByText('SYNTHETIC UI TEST RESPONSE neutral. Not a model measurement.',{exact:true})).toBeVisible();
 await page.getByRole('button',{name:'Token alignment',exact:true}).click();
 await expect(page.getByRole('button',{name:'token0',exact:true})).toHaveCount(3);
 await page.getByRole('button',{name:'token1',exact:true}).first().click();
 await expect(page.getByText(/Selected continuation position 1/)).toHaveCount(3);
 await page.getByRole('button',{name:'Activation explorer',exact:true}).click();
 const first=page.getByRole('button',{name:/Block 0, aligned token 0/});await expect(first).toHaveAttribute('aria-label',/0\.000000/);
 await first.focus();await page.keyboard.press('ArrowRight');await expect(page.getByRole('button',{name:/Block 0, aligned token 1 /})).toBeFocused();
 await page.getByText('Accessible heatmap values and exact masks',{exact:true}).click();
 await expect(page.getByRole('cell',{name:'Unavailable',exact:true})).toHaveCount(1);
 await page.getByRole('button',{name:'Aggregate evidence',exact:true}).click();
 await expect(page.getByRole('cell',{name:'0.000000',exact:true}).first()).toBeVisible();
 await page.getByRole('button',{name:'Method & exports',exact:true}).click();
 await expect(page.getByText(/Browser replay verifies inventory checksums, not authorship/)).toBeVisible();
});

test('corrupt bundle is rejected without replacing the current research',async({page})=>{
 await page.goto('/');await page.getByLabel('Open evidence ZIP').setInputFiles({name:'corrupt.zip',mimeType:'application/zip',buffer:testBundle({corrupt:true})});
 await expect(page.getByRole('alert')).toContainText(/checksum|SHA-256/i);
 await expect(page.getByRole('heading',{name:'Start with a careful question.'})).toBeVisible();
});

test('unsafe archive paths are rejected',async({page})=>{
 await page.goto('/');await page.getByLabel('Open evidence ZIP').setInputFiles({name:'unsafe.zip',mimeType:'application/zip',buffer:testBundle({unsafe:true})});
 await expect(page.getByRole('alert')).toContainText(/unsafe|path/i);
 await expect(page.getByRole('heading',{name:'Start with a careful question.'})).toBeVisible();
});

// This is the genuine bundled smoke, not the synthetic interaction fixture above.
test('bundled real smoke opens, carries provenance and remains usable offline',async({page,context})=>{
 await page.goto('/');await page.getByRole('button',{name:'Open recorded smoke',exact:true}).click();
 await expect(page.getByText('Recorded experiment',{exact:true})).toBeVisible();
 await expect(page.getByText(/Qwen\/Qwen2.5-0.5B-Instruct/).first()).toBeVisible();
 await expect(page.locator('.response-body').first()).not.toContainText('SYNTHETIC UI TEST');
 await expect(page.locator('.response-body')).toHaveCount(3);
 await page.screenshot({path:'artifacts/qa/measured-workbench.png',fullPage:true});
 await context.setOffline(true);await page.getByRole('button',{name:'Activation explorer',exact:true}).click();
 await expect(page.getByRole('group',{name:'Measured distance heatmap; use arrow keys to navigate cells'})).toBeVisible();
 await page.screenshot({path:'artifacts/qa/measured-activation-explorer.png',fullPage:true});
 await page.getByRole('button',{name:'Aggregate evidence',exact:true}).click();
 await expect(page.getByRole('heading',{name:'Evidence, with its uncertainty'})).toBeVisible();
 await expect(page.getByRole('cell',{name:/Neutral exact repeat/}).first()).toBeVisible();
 await page.getByRole('button',{name:'Method & exports',exact:true}).click();
 await expect(page.getByText(/Raw arrays are optional/)).toBeVisible();
});

test('production research views meet automated WCAG A/AA checks',async({page})=>{
 test.setTimeout(90000);await page.goto('/');
 const entry=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21a','wcag21aa']).analyze();
 writeFileSync('artifacts/qa/accessibility-entry.json',JSON.stringify(entry,null,2));
 expect.soft(entry.violations.map(v=>({id:v.id,targets:v.nodes.map(n=>n.target)}))).toEqual([]);
 await page.getByRole('button',{name:'Open recorded smoke',exact:true}).click();
 await expect(page.getByText('Recorded experiment',{exact:true})).toBeVisible();
 for(const view of ['Workbench','Activation explorer','Aggregate evidence','Method & exports','Behavior ratings','Experiments']){
  await page.getByRole('button',{name:view,exact:true}).click();
  const measured=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21a','wcag21aa']).analyze();
  writeFileSync(`artifacts/qa/accessibility-${view.toLowerCase().replaceAll(/[^a-z]+/g,'-')}.json`,JSON.stringify(measured,null,2));
  expect.soft(measured.violations.map(v=>({id:v.id,targets:v.nodes.map(n=>n.target)})),view).toEqual([]);
 }
});

test('recorded comparison works on mobile and high contrast light meets WCAG checks',async({page})=>{
 test.setTimeout(60000);await page.setViewportSize({width:390,height:844});await page.goto('/');
 await page.getByRole('button',{name:'Open recorded smoke',exact:true}).click();
 await expect(page.getByText('Recorded experiment',{exact:true})).toBeVisible();
 await expect(page.locator('.response-body')).toHaveCount(3);
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();
 await page.screenshot({path:'artifacts/qa/mobile-measured-workbench.png',fullPage:true});
 await page.getByRole('button',{name:'Use high contrast light appearance',exact:true}).click();
 const light=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21a','wcag21aa']).analyze();
 writeFileSync('artifacts/qa/accessibility-light.json',JSON.stringify(light,null,2));
 await page.screenshot({path:'artifacts/qa/mobile-measured-light.png',fullPage:true});
 expect(light.violations.map(v=>({id:v.id,targets:v.nodes.map(n=>n.target)}))).toEqual([]);
});

test('a checksummed archive with the wrong declared capture site is rejected',async({page})=>{
 await page.goto('/');await page.getByLabel('Open evidence ZIP').setInputFiles({name:'wrong-normalization.zip',mimeType:'application/zip',buffer:testBundle({badCapture:true})});
 await expect(page.getByRole('alert')).toContainText(/capture|normalization|artifact|site/i);
 await expect(page.getByRole('heading',{name:'Start with a careful question.'})).toBeVisible();
});

test('a rechecksummed inline model profile cannot escape its immutable profile hash',async({page})=>{
 await page.goto('/');await page.getByLabel('Open evidence ZIP').setInputFiles({name:'tampered-model.zip',mimeType:'application/zip',buffer:testBundle({badProfile:true})});
 await expect(page.getByRole('alert')).toContainText(/model|profile|hash|manifest/i);
 await expect(page.getByRole('heading',{name:'Start with a careful question.'})).toBeVisible();
});
