"""Bound bytes delivered by ASGI receive, including chunked requests."""
from starlette.types import ASGIApp,Scope,Receive,Send
from ghosttrace.storage.replay import MAX_ARCHIVE_BYTES
from starlette.responses import JSONResponse

class BodyLimitExceeded(Exception):pass

class BodyLimitMiddleware:
    def __init__(self,app:ASGIApp):self.app=app
    async def __call__(self,scope:Scope,receive:Receive,send:Send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        maximum=MAX_ARCHIVE_BYTES if scope.get('path')=='/api/v1/imports' else (500_000_000 if scope.get('path') in ('/api/v2/research/imports','/api/v2/research/dictionaries/imports','/api/v2/research/lenses/imports','/api/v2/research/lenses/train') else 5_000_000)
        total=0;exceeded=False;started=False
        async def bounded_receive():
            nonlocal total,exceeded
            message=await receive()
            if message['type']=='http.request':
                total+=len(message.get('body',b''))
                if total>maximum:exceeded=True;raise BodyLimitExceeded()
            return message
        async def guarded_send(message):
            nonlocal started
            if exceeded:return
            if message['type']=='http.response.start':started=True
            await send(message)
        try:await self.app(scope,bounded_receive,guarded_send)
        except BaseException:
            if not exceeded:raise
        if exceeded and not started:
            request_id=getattr(scope.get('state',{}),'request_id',None) if not isinstance(scope.get('state'),dict) else scope.get('state',{}).get('request_id')
            await JSONResponse({'error':{'code':'PAYLOAD_TOO_LARGE','message':f'Maximum request body is {maximum} bytes'},'request_id':request_id},413)(scope,receive,send)
