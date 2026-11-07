"""Interactive demo of the ISP pipelines.

    streamlit run app.py

Either simulate a phone capture of a Kodak image (choose light level, sensor
noise and illuminant) or upload a real RAW / DNG file.
"""
import glob
import io
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

import classical_isp as cl
import hybrid_isp as hy
import raw_sim as rs
from metrics import quality
from prepare_data import get_kodak

MAX_SIDE = 1536  # crop big RAW files so the CPU demo stays responsive

st.set_page_config(page_title='Hybrid Smartphone ISP', layout='wide')


@st.cache_resource
def load_pipelines():
    torch.backends.quantized.engine = 'onednn'
    core, e2e = hy.load_model('checkpoints/core.pt'), hy.load_model('checkpoints/e2e.pt')
    return {'Classical ISP': cl.classical_isp,
            'Learned ISP (e2e)': hy.LearnedISP(e2e),
            'Hybrid ISP': hy.HybridISP(core),
            'Hybrid-Lite INT8': hy.HybridISP(torch.jit.load('checkpoints/lite_int8.pt'))}


st.title('Hybrid learned smartphone ISP')
st.caption('Classical front end (BLC, gain, WB, noise profile) + learned demosaic/denoise core '
           '+ classical colour and tone back end, compared with a classical and an end-to-end learned ISP.')
pipes = load_pipelines()

src = st.sidebar.radio('Input', ['Simulated capture (Kodak24)', 'Upload RAW / DNG'])
gt = None
if src.startswith('Simulated'):
    files = sorted(glob.glob('data/kodak/*.png'))
    if not files:  # fresh deployment: fetch the 24 Kodak test images (~15 MB) once
        with st.spinner('Downloading the Kodak test images...'):
            get_kodak()
        files = sorted(glob.glob('data/kodak/*.png'))
    f = st.sidebar.selectbox('Scene', files, index=18, format_func=lambda p: p[-11:-4])
    gain = st.sidebar.select_slider('Light level (digital gain)', [1, 2, 4, 8, 16, 32, 64], value=16,
                                    help='x1 = daylight, x32 = night, x64 is outside the training range')
    shot = st.sidebar.select_slider('Sensor noise (shot noise)', [1e-4, 2e-4, 5e-4, 1e-3, 2e-3], value=5e-4)
    r = st.sidebar.slider('Illuminant: red gain', 1.2, 3.0, 2.1, 0.05)
    b = st.sidebar.slider('Illuminant: blue gain', 1.2, 3.0, 1.7, 0.05)
    blind = st.sidebar.checkbox('Drop WB and noise metadata (estimate from the image)')
    gt = np.asarray(Image.open(f).convert('RGB'), np.float32) / 255
    mosaic, meta = rs.capture(gt, gain, shot, (r, 1.0, b), seed=0)
    if blind:
        meta = {**meta, 'wb': None, 'noise': None}
else:
    up = st.sidebar.file_uploader('RAW file (DNG, CR2, NEF, ARW, ...)')
    if up is None:
        st.info('Upload a RAW file from the sidebar.')
        st.stop()
    mosaic, meta = hy.load_raw(io.BytesIO(up.getvalue()))
    if max(mosaic.shape) > MAX_SIDE:
        mosaic = hy.center_crop(mosaic, MAX_SIDE)
        st.sidebar.caption(f'Centre {MAX_SIDE}px crop processed.')

m = cl.normalize(mosaic, meta)
gain_used, wb_used, noise_used = cl.resolve_meta(m, meta)
st.sidebar.markdown(f'**Gain** x{gain_used:.1f} &nbsp; **WB** R {wb_used[0]:.2f} / B {wb_used[2]:.2f}  \n'
                    f'**Noise** shot {noise_used[0]:.1e}, read {noise_used[1]:.1e}')

outs, rows = {}, []
for name, isp in pipes.items():
    t = time.perf_counter()
    outs[name] = isp(mosaic, meta)
    row = {'pipeline': name, 'time (ms)': round((time.perf_counter() - t) * 1000)}
    if gt is not None:
        row.update({k: round(v, 3) for k, v in quality(outs[name], gt).items()})
    rows.append(row)

raw_view = np.clip(np.minimum(m * gain_used, 1) ** (1 / 2.2), 0, 1)
tiles = [('RAW Bayer mosaic (gain applied)', raw_view)] + ([('Ground truth', gt)] if gt is not None else [])
tiles += list(outs.items())
for i in range(0, len(tiles), 3):
    for col, (name, img) in zip(st.columns(3), tiles[i:i + 3]):
        col.image(np.clip(img, 0, 1), caption=name, use_container_width=True)

st.subheader('Metrics')
st.dataframe(pd.DataFrame(rows).set_index('pipeline'), use_container_width=True)
