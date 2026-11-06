import io
import os
import importlib.util
from functools import lru_cache
import numpy as np
import cv2
from PIL import Image, ImageOps, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 24_000_000

class Unavailable(Exception):
    pass

def decode(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        if image.width * image.height > 24_000_000:
            raise ValueError('Изображение превышает 24 мегапикселя')
        image.load()
        return ImageOps.exif_transpose(image).convert('RGBA')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Не удалось прочитать изображение') from exc

def png(image):
    out = io.BytesIO()
    image.save(out, 'PNG')
    return out.getvalue()

def capabilities():
    return {
        'remove_background': bool(importlib.util.find_spec('rembg')),
        'upscale_ai': bool(os.getenv('REALESRGAN_MODEL') and os.path.isfile(os.getenv('REALESRGAN_MODEL', '')) and importlib.util.find_spec('realesrgan')),
        'restore_face': bool(os.getenv('GFPGAN_MODEL') and os.path.isfile(os.getenv('GFPGAN_MODEL', '')) and importlib.util.find_spec('gfpgan')),
        'colorize': bool(os.getenv('GENERATIVE_API_URL') and os.getenv('GENERATIVE_API_KEY')),
        'clothes': bool(os.getenv('GENERATIVE_API_URL') and os.getenv('GENERATIVE_API_KEY')),
    }

@lru_cache(maxsize=1)
def background_session():
    from rembg import new_session
    return new_session('u2net')

def process(image, options):
    effect = options.effect
    amount = options.intensity / 100
    if options.official and effect not in ('enhance', 'lighting', 'document'):
        raise ValueError('Минимальное вмешательство: доступны свет, улучшение и формат документа')
    if effect in ('colorize', 'clothes'):
        raise Unavailable('Для этой функции подключите генеративный API')
    if effect == 'document':
        sizes = {'35x45': (413, 531), '30x40': (354, 472), '50x50': (591, 591)}
        canvas = Image.new('RGBA', sizes[options.document_format], options.background)
        fitted = ImageOps.contain(image, canvas.size, Image.Resampling.LANCZOS)
        canvas.alpha_composite(fitted, ((canvas.width-fitted.width)//2, (canvas.height-fitted.height)//2))
        return canvas
    if effect in ('remove_background', 'replace_background'):
        try:
            from rembg import remove
            result = remove(image, session=background_session())
        except Exception as exc:
            raise Unavailable('Удаление фона недоступно. Установите rembg[cpu] и загрузите модель u2net (см. README).') from exc
        # Intensity blends alpha rather than changing composition.
        result = Image.blend(image, result.convert('RGBA'), amount)
        if effect == 'replace_background':
            canvas = Image.new('RGBA', image.size, options.background)
            canvas.alpha_composite(result)
            result = canvas
        return result
    if effect in ('upscale_ai', 'restore_face'):
        try:
            if not capabilities()[effect]:
                raise RuntimeError('model missing')
            if effect == 'upscale_ai':
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                model = RealESRGANer(scale=4, model_path=os.environ['REALESRGAN_MODEL'], model=net, tile=256, half=False)
                output, _ = model.enhance(cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR), outscale=options.scale)
            else:
                from gfpgan import GFPGANer
                model = GFPGANer(model_path=os.environ['GFPGAN_MODEL'], upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
                _, _, output = model.enhance(cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR), has_aligned=False, only_center_face=False, paste_back=True)
            restored = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)).convert('RGBA')
            base = image.resize(restored.size, Image.Resampling.LANCZOS)
            restored.putalpha(base.getchannel('A'))
            return Image.blend(base, restored, amount)
        except Exception as exc:
            raise Unavailable('Модель недоступна или несовместима. Проверьте отдельное ML-окружение и путь к весам в README.') from exc
    if effect == 'upscale':
        if image.width * image.height * options.scale**2 > 48_000_000:
            raise ValueError('Результат превышает 48 мегапикселей')
        return image.resize((image.width*options.scale, image.height*options.scale), Image.Resampling.LANCZOS)
    rgb = np.array(image.convert('RGB'))
    if effect in ('enhance', 'lighting', 'restore'):
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8,8)).apply(lab[:,:,0])
        output = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        if effect in ('enhance', 'restore'):
            output = cv2.fastNlMeansDenoisingColored(output, None, 3, 3, 7, 21)
            output = cv2.addWeighted(output, 1.3, cv2.GaussianBlur(output, (0,0), 1), -0.3, 0)
    elif effect == 'sharpen':
        output = cv2.addWeighted(rgb, 1.7, cv2.GaussianBlur(rgb, (0,0), 1.2), -0.7, 0)
    elif effect == 'denoise':
        output = cv2.fastNlMeansDenoisingColored(rgb, None, 7, 7, 7, 21)
    elif effect == 'retouch':
        # Face-localized, weak bilateral smoothing preserves fine texture.
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(40,40))
        if len(faces) == 0:
            raise ValueError('Лицо не обнаружено. Попробуйте фронтальный портрет с лучшим освещением.')
        mask = np.zeros(gray.shape, np.uint8)
        for x,y,w,h in faces:
            cv2.ellipse(mask, (x+w//2,y+h//2), (int(w*.38),int(h*.43)), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask,(0,0),5).astype(np.float32)[:,:,None]/255 * .28
        smooth = cv2.bilateralFilter(rgb, 7, 22, 22)
        output = np.clip(rgb*(1-mask)+smooth*mask,0,255).astype(np.uint8)
    else:
        raise ValueError('Неизвестный эффект')
    strength = min(amount, .25) if options.official else amount
    result = Image.blend(image.convert('RGB'), Image.fromarray(output), strength).convert('RGBA')
    result.putalpha(image.getchannel('A'))
    return result
