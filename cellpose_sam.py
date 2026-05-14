#!/usr/bin/env python3
"""
cellpose_sam.py — segmentation faithful to Pachitariu, Rariden, Stringer (2025)
"Cellpose-SAM: superhuman generalization for cellular segmentation"
https://doi.org/10.1101/2025.04.28.651001

Design choices that follow the paper:
  - single model: 'cpsam' (the only Cellpose-SAM model)
  - no pre-specification of nuclear/cytoplasmic channel (channel-order invariant, Fig 3a)
  - no image resizing or diameter guessing (size-invariant, Fig 3b)
  - no denoising/deblurring/upsampling preprocessing (Fig 3c-f)
  - default thresholds flow_threshold=0.4, cellprob_threshold=0.0 (Methods)
  - test-time augmentation OFF, tile overlap 0.1 (Methods)
"""
from __future__ import annotations
import argparse, glob, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage.measure import regionprops_table
from skimage.segmentation import find_boundaries


def load_image(path: Path) -> tuple[np.ndarray, bool]:
    suf = path.suffix.lower()
    arr = tifffile.imread(str(path)) if suf in {".tif", ".tiff"} else skio.imread(str(path))
    arr = np.asarray(arr)
    is_3d = arr.ndim == 4 or (arr.ndim == 3 and arr.shape[0] > 6 and arr.shape[-1] > 6)
    print(f"[load] {path.name}: shape={arr.shape}, dtype={arr.dtype}, 3D={is_3d}")
    return arr, is_3d


def segment(model, img, is_3d: bool):
    masks, _flows, _styles = model.eval(
        img,
        batch_size=8,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        do_3D=is_3d,
        normalize=True,
        augment=False,
        tile_overlap=0.1,
    )
    print(f"[seg] {int(masks.max())} object(s)")
    return masks.astype(np.int32)


def quantify(masks, img, image_name: str) -> pd.DataFrame:
    if masks.max() == 0:
        return pd.DataFrame()
    if img.ndim == 2:
        chan = img[None, ...]
    elif img.ndim == 3 and img.shape[-1] <= 6:
        chan = np.moveaxis(img, -1, 0)
    elif img.ndim == 3 and img.shape[0] <= 6:
        chan = img
    else:
        chan = img.reshape(1, *img.shape)
    df = pd.DataFrame(regionprops_table(
        masks, properties=("label", "area", "centroid",
                           "eccentricity", "major_axis_length",
                           "minor_axis_length", "solidity")))
    if chan.ndim == masks.ndim + 1:
        for c in range(chan.shape[0]):
            p = regionprops_table(masks, intensity_image=chan[c],
                                  properties=("label", "intensity_mean", "intensity_max"))
            df[f"ch{c}_mean"] = p["intensity_mean"]
            df[f"ch{c}_max"]  = p["intensity_max"]
    df.insert(0, "image", image_name)
    return df


def save_outputs(out_dir: Path, name: str, img, masks, df) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / f"{name}_masks.tif"; tifffile.imwrite(str(p1), masks)
    p2 = out_dir / f"{name}_cells.csv"; df.to_csv(p2, index=False)
    # overlay (2D or middle slice for 3D)
    if masks.ndim == 3:
        m2 = masks[masks.shape[0] // 2]
        i2 = img[img.shape[0] // 2] if img.ndim >= 3 else img
    else:
        m2 = masks
        i2 = img if img.ndim == 2 else (
            img.mean(axis=-1) if img.shape[-1] <= 6 else img.mean(axis=0))
    i2 = (i2.astype(np.float32) - i2.min()) / (np.ptp(i2) + 1e-9)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(i2, cmap="gray");             ax[0].set_title("input")
    ax[1].imshow(m2, cmap="nipy_spectral");    ax[1].set_title(f"masks (n={int(m2.max())})")
    ov = np.stack([i2]*3, axis=-1)
    ov[find_boundaries(m2, mode="outer")] = [1.0, 0.2, 0.2]
    ax[2].imshow(ov);                          ax[2].set_title("overlay")
    for a in ax: a.axis("off")
    fig.tight_layout()
    p3 = out_dir / f"{name}_overlay.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight"); plt.close(fig)
    return [p1, p2, p3]


def git_push(repo_dir: Path, files, message: str,
             branch: str | None = None, create_pr: bool = False):
    def run(cmd):
        print(f"[gh] $ {' '.join(cmd)}")
        subprocess.run(cmd, cwd=repo_dir, check=True)
    if not (repo_dir / ".git").exists():
        run(["git", "init", "-b", "main"])
    if branch:
        run(["git", "checkout", "-B", branch])
    rel = [str(p.relative_to(repo_dir)) for p in files if p.exists()]
    if not rel: print("[gh] nothing to add"); return
    run(["git", "add", *rel])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo_dir).returncode == 0:
        print("[gh] no staged changes"); return
    run(["git", "commit", "-m", message])
    if subprocess.run(["git", "remote"], cwd=repo_dir,
                      capture_output=True, text=True).stdout.strip():
        run(["git", "push", "-u", "origin", branch] if branch else ["git", "push"])
        if create_pr and shutil.which("gh"):
            run(["gh", "pr", "create", "--fill", "--base", "main"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--pattern", default="*.tif")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--git-push", action="store_true")
    p.add_argument("--git-branch", default=None)
    p.add_argument("--git-message", default="auto: cellpose-SAM run")
    p.add_argument("--git-pr", action="store_true")
    args = p.parse_args()

    files = [args.input] if args.input.is_file() else \
            sorted(Path(x) for x in glob.glob(str(args.input / args.pattern)))
    if not files:
        print(f"[main] no inputs in {args.input}"); return 1

    from cellpose import models
    print(f"[model] loading Cellpose-SAM (cpsam), gpu={not args.no_gpu}")
    model = models.CellposeModel(gpu=not args.no_gpu, pretrained_model="cpsam")

    args.output.mkdir(parents=True, exist_ok=True)
    all_rows, produced = [], []
    for f in files:
        print("\n" + "-" * 60); print(f"[main] {f.name}")
        try:
            img, is_3d = load_image(f)
            masks = segment(model, img, is_3d)
            df = quantify(masks, img, f.name)
            produced += save_outputs(args.output, f.stem, img, masks, df)
            all_rows.append(df)
        except Exception as e:
            print(f"[main] FAILED on {f.name}: {e}")
    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        cp = args.output / "all_cells.csv"; combined.to_csv(cp, index=False)
        produced.append(cp)
        print(f"\n[main] wrote {cp}: {len(combined)} cells")
    if args.git_push:
        git_push(args.output, produced, args.git_message,
                 branch=args.git_branch, create_pr=args.git_pr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
