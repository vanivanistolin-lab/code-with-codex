import io
import zipfile
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from backend.app import main

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(main,'STORE',tmp_path)
    return TestClient(main.app)

def upload(client, size=(96,64)):
    image=Image.new('RGBA',size,(90,120,150,180))
    data=io.BytesIO(); image.save(data,'PNG')
    r=client.post('/api/upload',files={'file':('test.png',data.getvalue(),'image/png')})
    assert r.status_code==200
    return r.json()

@pytest.mark.parametrize('effect',['enhance','lighting','sharpen','denoise','restore'])
def test_effect_preserves_original_and_alpha(client,effect):
    original=upload(client)
    before=client.get(original['url']).content
    r=client.post('/api/process',json={'ids':[original['id']],'options':{'effect':effect}})
    result=r.json()['items'][0]['result']
    image=Image.open(io.BytesIO(client.get(result['url']).content))
    assert image.size==(96,64)
    assert image.getpixel((0,0))[3]==180
    assert result['id']!=original['id']
    assert client.get(original['url']).content==before

def test_zero_intensity(client):
    original=upload(client)
    result=client.post('/api/process',json={'ids':[original['id']],'options':{'intensity':0}}).json()['items'][0]['result']
    assert client.get(result['url']).content==client.get(original['url']).content

def test_document_and_upscale(client):
    photo=upload(client)
    for effect,size in [('document',(413,531)),('upscale',(192,128))]:
        result=client.post('/api/process',json={'ids':[photo['id']],'options':{'effect':effect}}).json()['items'][0]['result']
        assert (result['width'],result['height'])==size

def test_unavailable_api_and_official(client,monkeypatch):
    monkeypatch.delenv('GENERATIVE_API_URL',raising=False)
    photo=upload(client)
    for options in [{'effect':'clothes'},{'effect':'retouch','official':True}]:
        r=client.post('/api/process',json={'ids':[photo['id']],'options':options})
        assert r.status_code==200
        assert r.json()['items'][0]['error']

def test_batch_zip_and_validation(client):
    photos=[upload(client),upload(client)]
    results=client.post('/api/process',json={'ids':[p['id'] for p in photos],'options':{}}).json()['items']
    assert len(results)==2 and all('result' in r for r in results)
    r=client.post('/api/export',json={'ids':[x['result']['id'] for x in results],'format':'jpeg'})
    archive=zipfile.ZipFile(io.BytesIO(r.content))
    assert len(archive.namelist())==2
    assert Image.open(archive.open(archive.namelist()[0])).format=='JPEG'
    assert client.post('/api/upload',files={'file':('bad.jpg',b'bad','image/jpeg')}).status_code==422
    assert client.get('/api/images/not-a-valid-id').status_code==404
    assert client.post('/api/process',json={'ids':[photos[0]['id']],'options':{'intensity':101}}).status_code==422
