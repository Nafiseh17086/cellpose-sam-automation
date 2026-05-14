# cellpose-sam-automation

Automated multichannel cell segmentation with [Cellpose-SAM](https://github.com/MouseLand/cellpose), following **Pachitariu, Rariden, Stringer (2025)** — *Cellpose-SAM: superhuman generalization for cellular segmentation* ([bioRxiv 2025.04.28.651001](https://doi.org/10.1101/2025.04.28.651001)).

A Jupyter notebook for interactive exploration, a CLI script for batch jobs, and a one-shot `gh` bootstrap to push it all to GitHub.

## What's in here

```
.
├── notebooks/
│   └── cellpose_sam_demo.ipynb     ← run me first (interactive demo)
├── cellpose_sam.py                 ← CLI version: folder in, folder out
├── examples/
│   └── fluorescence_sample.png     ← demo image (multichannel fluorescence)
├── docs/
│   └── approach.pdf                ← short write-up of the approach
├── scripts/
│   └── bootstrap_github.sh         ← one-shot `gh repo create` helper
├── requirements.txt
├── .gitignore
├── LICENSE                         ← Apache-2.0
└── README.md
```

## Quick start

### 1. Install

```bash
pip install -r requirements.txt
# GPU (recommended for speed):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# Apple Silicon: just `pip install torch torchvision` (MPS works out of the box)
```

### 2. Run the notebook

```bash
jupyter lab notebooks/cellpose_sam_demo.ipynb
```

Or open it in VS Code, Colab, etc. First run downloads the `cpsam` weights (~370 MB) to `~/.cellpose/models/`.

### 3. Or use the CLI

```bash
# one image
python cellpose_sam.py --input examples/fluorescence_sample.png --output ./out

# a folder
python cellpose_sam.py --input ./images --output ./out --pattern "*.tif"

# folder + auto-commit results to git
python cellpose_sam.py --input ./images --output ./out \
  --git-push --git-branch "results/$(date +%Y%m%d)" --git-pr
```

## What this script does

```
input image(s)
   │
   ├─ load
   ├─ Cellpose-SAM (cpsam)  →  integer instance mask
   ├─ regionprops across ALL channels
   │     (area, centroid, eccentricity, axes, solidity,
   │      mean & max intensity per channel)
   │
   └─ writes:
        out/<name>_masks.tif      ← int32, opens in Fiji / napari / QuPath
        out/<name>_overlay.png    ← 3-panel figure (input, masks, outlines)
        out/<name>_cells.csv      ← one row per cell
        out/all_cells.csv         ← combined across all inputs
```

## Why so little preprocessing?

Because Cellpose-SAM is engineered to need almost none. From the paper:

- **Channel-order invariant** (Fig 3a) — no need to tell it which channel is DAPI
- **Size-invariant** within 7.5–120 px diameter (Fig 3b) — no diameter guessing
- **Robust to noise, blur, downsampling, anisotropic blur** (Fig 3c–f) — no denoising step needed

So the pipeline is just `load → eval → quantify`. Anything more is friction.

## Performance expectations

From paper Table S1 (1200×1200 image):

| Hardware | Runtime |
|---|---|
| A100 (cloud) | ~0.4 s |
| RTX 4070 Super | ~3.2 s |
| RTX 4060 | ~6 s |
| T4 (Colab free tier) | ~7 s |
| CPU | minutes (not recommended) |

## Push your fork to GitHub

```bash
./scripts/bootstrap_github.sh my-cellpose-fork --public
```

That runs `gh repo create my-cellpose-fork --public --source=. --remote=origin --push` under the hood. After it's done:

```bash
gh repo view --web
```

opens the repo in your browser.

## License

Apache-2.0 for the code. The Cellpose-SAM model weights are CC-BY-NC; see the [upstream license](https://github.com/MouseLand/cellpose#licenses) before using them in commercial work.

## Citation

```bibtex
@article{pachitariu2025cellposesam,
  title={Cellpose-SAM: superhuman generalization for cellular segmentation},
  author={Pachitariu, Marius and Rariden, Michael and Stringer, Carsen},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.04.28.651001}
}
```
