import {defineConfig,devices} from '@playwright/test';
const publicURL=process.env.GHOSTTRACE_PAGES_URL;
if(publicURL){const url=new URL(publicURL);if(!['https:','http:'].includes(url.protocol)||url.username||url.password||url.search||url.hash||!url.pathname.endsWith('/'))throw Error('GHOSTTRACE_PAGES_URL must be an HTTP(S) site root with a trailing slash, without credentials, query or hash.')}
export default defineConfig({
 testDir:'./tests/e2e',testMatch:'research-pages.spec.ts',fullyParallel:false,workers:1,retries:0,forbidOnly:!!process.env.CI,timeout:60000,
 reporter:[['list'],['json',{outputFile:'artifacts/qa/pages-playwright-report.json'}]],outputDir:'artifacts/qa/pages-browser-results',
 use:{baseURL:publicURL||'http://127.0.0.1:4174/GhostTrace/',trace:'retain-on-failure',screenshot:'only-on-failure',reducedMotion:'reduce'},
 projects:[{name:'pages-chromium',use:{...devices['Desktop Chrome']}}],
 webServer:publicURL?undefined:{command:'npm run preview:pages -- --port 4174 --strictPort',url:'http://127.0.0.1:4174/GhostTrace/',reuseExistingServer:false,timeout:30000},
});
