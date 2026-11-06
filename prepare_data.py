"""Downloads the datasets used in this project.

Kodak24      -> data/kodak  (test set, 24 images)
DIV2K valid  -> data/div2k  (training set, 100 images, downscaled x2)

The RAW inputs are synthesised from these sRGB images (see raw_sim.py),
so no RAW dataset download is needed.
"""
import io
import os
import zipfile
import urllib.request

from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
KODAK_URL = 'https://r0k.us/graphics/kodak/kodak/kodim{:02d}.png'
DIV2K_URL = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip'


def get_kodak():
    out = os.path.join(ROOT, 'kodak')
    os.makedirs(out, exist_ok=True)
    for i in range(1, 25):
        path = os.path.join(out, f'kodim{i:02d}.png')
        if not os.path.exists(path):
            urllib.request.urlretrieve(KODAK_URL.format(i), path)
    print('kodak ok')


def get_div2k():
    out = os.path.join(ROOT, 'div2k')
    os.makedirs(out, exist_ok=True)
    if len(os.listdir(out)) >= 100:
        return
    zpath = os.path.join(ROOT, 'div2k.zip')
    if not os.path.exists(zpath):
        print('downloading DIV2K (~450 MB)...')
        urllib.request.urlretrieve(DIV2K_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.endswith('.png'):
                continue
            im = Image.open(io.BytesIO(z.read(name))).convert('RGB')
            # x2 downscale gives cleaner targets with more detail per pixel
            im = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
            im.save(os.path.join(out, os.path.basename(name)))
    os.remove(zpath)
    print('div2k ok')


if __name__ == '__main__':
    get_kodak()
    get_div2k()
