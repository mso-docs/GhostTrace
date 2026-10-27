#!/usr/bin/env node
/** Integrity and accessibility companion checks for the checked-in recording. */
import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import path from 'node:path';
const root='web/public/help/',hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const m=JSON.parse(fs.readFileSync(root+'walkthrough-manifest.json','utf8'));
const chapters=JSON.parse(fs.readFileSync('web/src/help-content.json','utf8'));
assert.equal(m.schema_version,'1.0-walkthrough');assert.equal(m.model_inference_executed,false);
assert.equal(m.video_sha256,hash(root+'researcher-walkthrough.webm'),'Recording checksum changed');
assert.equal(m.case_sha256,hash('web/public/research/demo.json'),'Re-record walkthrough after changing its measured case');
assert.equal(m.chapter_script_sha256,hash('web/src/help-content.json'),'Re-record after changing the chapter script');
assert.equal(m.duration_seconds,120);assert.equal(m.chapters.length,8);
const video=fs.readFileSync(root+'researcher-walkthrough.webm');assert.equal(video.subarray(0,4).toString('hex'),'1a45dfa3','Expected an actual WebM container');assert.ok(video.length>100000&&video.length<25000000,'Keep the walkthrough within its 25 MB static-media budget');
const captions=fs.readFileSync(root+'researcher-walkthrough.en.vtt','utf8'),transcript=fs.readFileSync(root+'walkthrough-transcript.txt','utf8');
assert.ok(captions.startsWith('WEBVTT\n'));assert.equal((captions.match(/ --> /g)||[]).length,chapters.length*2);
for(const c of chapters){assert.ok(captions.includes(c.action)&&captions.includes(c.meaning));assert.ok(transcript.includes(c.title)&&transcript.includes(c.action)&&transcript.includes(c.meaning))}
assert.ok(fs.statSync(root+'walkthrough-poster.jpg').size>1000);
const args=process.argv.slice(2);let builtArtifact=null;
if(args.includes('--dist')){
 const directory=args[args.indexOf('--dist')+1],base=args.includes('--base')?args[args.indexOf('--base')+1]:'/Ghost-Trace/';
 assert.ok(directory&&!directory.startsWith('--'),'Supply a built directory after --dist');
 assert.ok(base.startsWith('/')&&base.endsWith('/'),'Expected an absolute project-site base with trailing slash');
 const assets=['researcher-walkthrough.webm','researcher-walkthrough.en.vtt','walkthrough-transcript.txt','walkthrough-poster.jpg','walkthrough-manifest.json'];
 for(const asset of assets)assert.equal(hash(path.join(directory,'help',asset)),hash(root+asset),'Built media differs: '+asset);
 const html=fs.readFileSync(path.join(directory,'index.html'),'utf8');
 const urls=[...html.matchAll(/(?:src|href)="([^"]+)"/g)].map(m=>m[1]).filter(url=>url.includes('/assets/'));
 assert.ok(urls.length>0,'No built app assets found');
 for(const url of urls){assert.ok(url.startsWith(base+'assets/'),'App asset escapes project-site base: '+url);assert.ok(fs.statSync(path.join(directory,url.slice(base.length))).isFile())}
 builtArtifact={directory,base,identical_media_files:assets.length};
}
console.log(JSON.stringify({status:'passed',chapters:chapters.length,video_bytes:video.length,descriptive_captions:true,transcript:true,built_artifact:builtArtifact,case_sha256:m.case_sha256}));
