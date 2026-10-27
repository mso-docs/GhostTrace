import {defineConfig,devices} from '@playwright/test';
export default defineConfig({
 testDir:'./tests/e2e',testIgnore:'research-pages.spec.ts',fullyParallel:false,forbidOnly:!!process.env.CI,retries:0,workers:1,timeout:30000,
 reporter:[['list'],['json',{outputFile:'artifacts/qa/playwright-report.json'}],['html',{outputFolder:'artifacts/qa/playwright-html',open:'never'}]],
 outputDir:'artifacts/qa/browser-results',
 use:{baseURL:'http://127.0.0.1:4173',trace:'retain-on-failure',screenshot:'only-on-failure',reducedMotion:'reduce'},
 projects:[{name:'chromium',use:{...devices['Desktop Chrome']}}],
 webServer:{command:'npm run preview -- --port 4173',url:'http://127.0.0.1:4173',reuseExistingServer:!process.env.CI,timeout:30000},
});
