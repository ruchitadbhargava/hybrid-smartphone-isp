"""Builds colab/hybrid_isp_colab.ipynb, a self-contained Colab notebook.

The project modules are embedded with %%writefile, and the notebook asks
for a T4 GPU runtime, so training is just "Runtime > Run all". The last cell
downloads the trained checkpoints and results as a zip.

    python colab/build_notebook.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = ['raw_sim.py', 'classical_isp.py', 'model.py', 'hybrid_isp.py', 'train.py',
           'prepare_data.py', 'metrics.py', 'benchmark.py', 'evaluate.py']
DNG = 'https://raw.pixls.us/download/data/Google/Pixel%204a/PXL_20201121_100251397.dng'
CHECK = "assert globals().get('_exit_code', 0) == 0, 'step failed, see output above'"


def code(src):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': src.splitlines(keepends=True)}


def main():
    cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': [
        '# Hybrid ISP: training on Colab GPU\n', '\n',
        'Use **Runtime > Run all**. The notebook asks for a T4 GPU runtime.\n',
        'It writes the project code, downloads the data, trains the three models, exports the INT8 model, ',
        'runs the evaluation and finally downloads `hybrid_isp_colab_results.zip` (checkpoints + results).']}]
    cells.append(code("!nvidia-smi -L\nimport torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"))
    for m in MODULES:
        with open(os.path.join(ROOT, m), encoding='utf-8') as f:
            cells.append(code(f'%%writefile {m}\n' + f.read()))
    cells.append(code(f'!pip -q install lpips rawpy\n!python prepare_data.py\n{CHECK}\n'
                      f'!mkdir -p data/raw && wget -q -O data/raw/pixel4a.dng "{DNG}" && ls -la data/raw'))
    cells.append(code(f'!python train.py core --width 24 --steps 15000 --batch 32\n{CHECK}'))
    cells.append(code(f'!python train.py e2e --width 24 --steps 8000 --batch 32 --init checkpoints/core.pt\n{CHECK}'))
    cells.append(code(f'!python train.py lite --width 16 --steps 12000 --batch 32 --teacher checkpoints/core.pt\n{CHECK}'))
    cells.append(code(f'!python -c "import benchmark; benchmark.export_int8()"\n{CHECK}'))
    cells.append(code(f'!python evaluate.py\n{CHECK}\n!cat results/metrics.md'))
    cells.append(code("!zip -qr hybrid_isp_colab_results.zip checkpoints results && ls -la hybrid_isp_colab_results.zip\n"
                      "from google.colab import files\nfiles.download('hybrid_isp_colab_results.zip')"))
    nb = {'nbformat': 4, 'nbformat_minor': 0, 'cells': cells,
          'metadata': {'accelerator': 'GPU',
                       'colab': {'provenance': [], 'gpuType': 'T4', 'name': 'hybrid_isp_colab.ipynb'},
                       'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                       'language_info': {'name': 'python'}}}
    with open(os.path.join(ROOT, 'colab', 'hybrid_isp_colab.ipynb'), 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(len(cells), 'cells written')


if __name__ == '__main__':
    main()
