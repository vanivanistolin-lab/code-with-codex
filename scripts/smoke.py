"""Live-server smoke test. Run from project root; --with-model checks actual U2NET."""
import io
import sys
import zipfile
import httpx
from PIL import Image, ImageDraw

image = Image.new('RGB', (192, 192), '#eeeeee')
draw = ImageDraw.Draw(image)
draw.ellipse((60, 25, 130, 100), fill='#b77b58')
draw.rectangle((48, 95, 143, 191), fill='#205080')
data = io.BytesIO()
image.save(data, 'PNG')
with httpx.Client(base_url='http://127.0.0.1:8000', trust_env=False, timeout=180) as client:
    assert client.get('/api/health').json()['status'] == 'ok'
    assert 'PhotoMaster' in client.get('/').text
    source = client.post('/api/upload', files={'file': ('smoke.png', data.getvalue(), 'image/png')}).json()
    effect = 'remove_background' if '--with-model' in sys.argv else 'enhance'
    item = client.post('/api/process', json={'ids': [source['id']], 'options': {'effect': effect, 'intensity': 100}}).json()['items'][0]
    assert 'result' in item, item
    response = client.post('/api/export', json={'ids': [item['result']['id']], 'format': 'png'})
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        result = Image.open(archive.open('PhotoMaster-1.png'))
        assert result.size == image.size
        if effect == 'remove_background':
            assert result.getchannel('A').getextrema() == (0, 255)
    print(f'HTTP smoke passed: frontend, upload, {effect}, ZIP, dimensions')
