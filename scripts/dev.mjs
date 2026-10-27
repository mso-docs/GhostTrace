// Start the local API, one explicit worker and Vite; stop every owned process.
import {spawn} from 'node:child_process';
import {existsSync} from 'node:fs';
import {resolve} from 'node:path';
const python=process.platform==='win32'?resolve('.venv/Scripts/python.exe'):resolve('.venv/bin/python');
if(!existsSync(python)||!existsSync('node_modules/vite/bin/vite.js')){console.error('Install first: uv sync --frozen --extra test && npm ci');process.exit(1)}
const env={...process.env};if(!env.HF_HOME&&existsSync('artifacts/huggingface'))env.HF_HOME=resolve('artifacts/huggingface');
const children=[];let stopping=false,shutdownTimer;
const exited=p=>p.exitCode!==null||p.signalCode!==null;
function signalOwned(p,signal){if(!p.pid)return;try{if(process.platform==='win32')p.kill(signal);else process.kill(-p.pid,signal)}catch(e){if(e.code!=='ESRCH')console.error('Could not signal owned child:',e.message)}}
function stop(code=0){
 if(stopping)return;stopping=true;process.exitCode=code;
 for(const p of children)signalOwned(p,'SIGTERM');
 shutdownTimer=setTimeout(()=>{for(const p of children)if(!exited(p))signalOwned(p,'SIGKILL');process.exit(code)},2000);
 if(children.every(exited)){clearTimeout(shutdownTimer);process.exit(code)}
}
const commands=[[python,['-m','ghosttrace.cli','serve','--host','127.0.0.1']],[python,['-m','ghosttrace.cli','worker']],[process.execPath,[resolve('node_modules/vite/bin/vite.js'),'--host','127.0.0.1','--strictPort']]];
for(const [cmd,args] of commands){
 const p=spawn(cmd,args,{stdio:'inherit',env,detached:process.platform!=='win32'});children.push(p);
 p.on('error',e=>{console.error(e.message);stop(1)});
 p.on('exit',code=>{if(!stopping)stop(code||0);else if(children.every(exited)){clearTimeout(shutdownTimer);process.exit(process.exitCode||0)}});
}
process.on('SIGINT',()=>stop());process.on('SIGTERM',()=>stop());
console.log('GhostTrace local stack: API http://127.0.0.1:8000 · Workbench http://127.0.0.1:5173 · one explicit worker. No models are downloaded.');
