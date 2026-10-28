import {defineConfig,loadEnv} from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig(({mode})=>{
 const env=loadEnv(mode,process.cwd(),'VITE_');
 const staticDemo=(process.env.VITE_STATIC_DEMO||env.VITE_STATIC_DEMO)==='true';
 const requested=process.env.VITE_BASE_PATH||env.VITE_BASE_PATH||(staticDemo?'/GhostTrace/':'/');
 const base=requested.startsWith('/')&&requested.endsWith('/')?requested:'/'+requested.replace(/^\/+|\/+$/g,'')+'/';
 return {plugins:[react()],base,publicDir:'web/public',server:{proxy:staticDemo?undefined:{'/api':{target:'http://127.0.0.1:8000',changeOrigin:true}}},build:{outDir:'dist'}};
});
