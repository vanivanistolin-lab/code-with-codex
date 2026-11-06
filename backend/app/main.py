import asyncio
import io
import os
import re
import uuid
import zipfile
from pathlib import Path
from PIL import Image
from typing import Literal
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .processing import decode, png, process, capabilities, Unavailable

ROOT = Path(__file__).resolve().parents[2]
STORE = Path(os.getenv('PHOTOMASTER_STORAGE', str(ROOT / 'storage')))
STORE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('U2NET_HOME', str(ROOT / 'models' / 'u2net'))
app = FastAPI(title='PhotoMaster AI', version='1.0.0')
lock = asyncio.Semaphore(1)

class Options(BaseModel):
    effect: Literal['enhance','lighting','sharpen','denoise','retouch','restore','colorize','upscale','upscale_ai','restore_face','remove_background','replace_background','document','clothes'] = 'enhance'
    intensity: int = Field(default=60, ge=0, le=100)
    official: bool = False
    scale: Literal[2,4] = 2
    background: str = Field(default='#ffffff', pattern=r'^#[0-9a-fA-F]{6}$')
    document_format: Literal['35x45','30x40','50x50'] = '35x45'
    prompt: str = Field(default='', max_length=1000)

class Batch(BaseModel):
    ids: list[str] = Field(min_length=1,max_length=20)
    options: Options

class Export(BaseModel):
    ids: list[str] = Field(min_length=1,max_length=20)
    format: Literal['png','jpeg','webp'] = 'png'

def path_for(identifier):
    if not re.fullmatch(r'[a-f0-9]{32}',identifier):
        raise HTTPException(404,'Изображение не найдено')
    path = STORE / f'{identifier}.png'
    if not path.is_file():
        raise HTTPException(404,'Изображение не найдено')
    return path

def persist(image):
    identifier = uuid.uuid4().hex
    (STORE/f'{identifier}.png').write_bytes(png(image))
    return {'id':identifier,'url':f'/api/images/{identifier}','width':image.width,'height':image.height}

@app.get('/api/health')
def health():
    return {'status':'ok'}

@app.get('/api/capabilities')
def available():
    return capabilities()

@app.post('/api/upload')
async def upload(file: UploadFile = File(...)):
    data = await file.read(20*1024*1024+1)
    if len(data)>20*1024*1024:
        raise HTTPException(413,'Максимальный размер файла — 20 МБ')
    try:
        async with lock:
            result = await asyncio.to_thread(lambda: persist(decode(data)))
        return result
    except ValueError as exc:
        raise HTTPException(422,str(exc))

@app.get('/api/images/{identifier}')
def get_image(identifier: str):
    return FileResponse(path_for(identifier),media_type='image/png',headers={'Cache-Control':'private, max-age=3600'})

async def apply_one(identifier, options):
    source = path_for(identifier)
    if options.official and options.effect not in ('enhance','lighting','document'):
        raise ValueError('Минимальное вмешательство: доступны свет, улучшение и формат документа')
    if options.effect in ('clothes','colorize'):
        endpoint, key = os.getenv('GENERATIVE_API_URL'), os.getenv('GENERATIVE_API_KEY')
        if not endpoint or not key:
            raise Unavailable('Подключите GENERATIVE_API_URL и GENERATIVE_API_KEY на сервере. Фото передаётся внешнему провайдеру только при запуске этой функции.')
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(endpoint, headers={'Authorization':f'Bearer {key}'}, files={'image':('image.png',source.read_bytes(),'image/png')}, data={'operation':options.effect,'prompt':options.prompt,'intensity':str(options.intensity),'preserve_identity':'true','preserve_composition':'true'})
        if response.status_code != 200 or len(response.content)>20*1024*1024:
            raise Unavailable('Внешний API вернул ошибку или слишком большой результат')
        result = decode(response.content)
        original = decode(source.read_bytes())
        if result.size != original.size:
            raise ValueError('API изменил размеры фотографии. Результат отклонён для сохранения композиции.')
        return persist(result)
    def work():
        image = decode(source.read_bytes())
        if options.effect == 'upscale_ai' and image.width*image.height*options.scale**2 > 48_000_000:
            raise ValueError('Результат превышает 48 мегапикселей')
        return persist(process(image, options))
    return await asyncio.to_thread(work)

@app.post('/api/process')
async def batch(request: Batch):
    results=[]
    async with lock:
        for identifier in request.ids:
            try:
                results.append({'source_id':identifier,'result':await apply_one(identifier,request.options)})
            except (ValueError,Unavailable,httpx.HTTPError) as exc:
                results.append({'source_id':identifier,'error':str(exc)})
    return {'items':results}

@app.post('/api/export')
def export(request: Export):
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for index,identifier in enumerate(request.ids,1):
            image=decode(path_for(identifier).read_bytes())
            if request.format=='jpeg':
                base=Image.new('RGB',image.size,'white')
                base.paste(image,mask=image.getchannel('A'))
                image=base
            data=io.BytesIO()
            image.save(data,request.format.upper(),quality=95,dpi=(300,300))
            archive.writestr(f'PhotoMaster-{index}.{request.format}',data.getvalue())
    return Response(output.getvalue(),media_type='application/zip',headers={'Content-Disposition':'attachment; filename="PhotoMaster.zip"'})

if (ROOT/'frontend/dist').is_dir():
    app.mount('/', StaticFiles(directory=ROOT/'frontend/dist',html=True),name='frontend')
