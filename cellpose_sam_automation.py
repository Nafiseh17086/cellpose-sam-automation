#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cellpose_sam_automation.py
==========================

Automated multichannel cell segmentation + per-cell quantification using
Cellpose-SAM, structured after the MONAI spleen_segmentation_3d tutorial
and the workflow of joaomamede/SpiningMonkeysAI (AIMonkeys2022-Pass2).

Pipeline (mirrors the MONAI sections):
  1. Setup environment           -> install / import
  2. Setup data directory        -> input / output folders
  3. Transforms                  -> read multichannel TIFF, choose nuclear channel
  4. Create Model                -> CellposeModel(pretrained_model="cpsam")
  5. Execute                     -> model.eval() per image (2D or 3D)
  6. Quantify                    -> per-cell intensity in every channel
  7. Save outputs                -> masks (.tif), overlays (.png), table (.csv)
  8. GitHub CLI automation       -> commit/push results via `gh` / `git`

Designed for fluorescence images like the one you uploaded:
  - blue (DAPI)     -> nuclei  -> segmentation channel
  - cyan / yellow / magenta / red / green -> markers -> quantified per cell

Usage
-----
    # one image
    python cellpose_sam_automation.py --input  sample.tif \
                                      --output ./out \
                                      --nuc-channel 0

    # a whole folder, no GPU, dump to GitHub
    python cellpose_sam_automation.py --input  ./images \
                                      --output ./out \
                                      --pattern "*.tif" \
                                      --no-gpu \
                                      --git-push \
                                      --git-message "auto: cellpose-SAM run"

Tested with cellpose >= 4.0 (Cellpose-SAM, model name 'cpsam').
"""

# ---------------------------------------------------------------------------
# 1. Setup environment  (run once; safe to comment out if already installed)
# ---------------------------------------------------------------------------
#   pip install "cellpose>=4.0" tifffile scikit-image pandas matplotlib numpy
#   # GPU (CUDA 12.x):
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
#   # GitHub CLI (one of):
#   #   macOS:   brew install gh
#   #   Ubuntu:  sudo apt install gh
#   #   Windows: winget install --id GitHub.cli
#   gh auth login          # interactive, once per machine

# ---------------------------------------------------------------------------
# 2. Setup imports
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage import io as skio
from skimage.measure import regionprops_table
from skimage.segmentation import find_boundaries

# Cellpose-SAM. Import lazily inside main() so --help works without it.
def _import_cellpose():
    from cellpose import models, io as cpio  # noqa: WPS433  (runtime import)
    return models, cpio


# ---------------------------------------------------------------------------
# 3. "Transforms" — light, NumPy-based equivalents of the MONAI dict pipeline
# ---------------------------------------------------------------------------
@dataclass
class LoadedImage:
    """Container for a loaded image + its provenance."""
    path: Path
    array: np.ndarray            # always shaped (C, [Z,] Y, X), float32
    is_3d: bool
    channel_names: list[str]


def load_imaged(path: Path) -> LoadedImage:
    """LoadImaged equivalent.

    Reads any TIFF / OME-TIFF / PNG / JPG with tifffile or skimage and
    normalizes the axis order to (C, [Z,] Y, X). Handles the common
    pitfalls: trailing channel axis, missing channel axis, single-slice
    z stacks.
    """
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        arr = tifffile.imread(str(path))
    else:
        arr = skio.imread(str(path))

    arr = np.asarray(arr)

    # Heuristics to put channels first. tifffile mostly returns (Z,Y,X,C)
    # or (Y,X,C) for OME-TIFFs; (Y,X) for single-channel.
    if arr.ndim == 2:                       # (Y, X)            -> (1, Y, X)
        arr = arr[None, ...]
        is_3d = False
    elif arr.ndim == 3:
        # ambiguous: (C,Y,X), (Y,X,C), or (Z,Y,X)
        # smallest dim with size <= 6 is almost certainly channels
        if arr.shape[-1] <= 6 and arr.shape[-1] < arr.shape[0]:
            arr = np.moveaxis(arr, -1, 0)   # (Y,X,C) -> (C,Y,X)
            is_3d = False
        elif arr.shape[0] <= 6 and arr.shape[0] < arr.shape[-1]:
            is_3d = False                   # already (C,Y,X)
        else:
            arr = arr[None, ...]            # treat as (1, Z, Y, X) — single channel z-stack
            is_3d = True
    elif arr.ndim == 4:
        # (Z,Y,X,C) -> (C,Z,Y,X)  or  (C,Z,Y,X) untouched  or  (Z,C,Y,X) -> (C,Z,Y,X)
        if arr.shape[-1] <= 6:
            arr = np.moveaxis(arr, -1, 0)
        elif arr.shape[1] <= 6 and arr.shape[0] > 6:
            arr = np.moveaxis(arr, 1, 0)
        is_3d = True
    else:
        raise ValueError(f"Unsupported array shape {arr.shape} for {path.name}")

    arr = arr.astype(np.float32)
    n_ch = arr.shape[0]
    channel_names = [f"ch{i}" for i in range(n_ch)]
    return LoadedImage(path=path, array=arr, is_3d=is_3d, channel_names=channel_names)


def scale_intensity_ranged(
    img: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.5,
    clip: bool = True,
) -> np.ndarray:
    """ScaleIntensityRanged equivalent — percentile-based contrast stretch.

    Done per-channel so dim markers don't get crushed by the brightest one.
    """
    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[0]):
        lo, hi = np.percentile(img[c], (p_low, p_high))
        if hi <= lo:
            out[c] = 0.0
            continue
        out[c] = (img[c] - lo) / (hi - lo)
        if clip:
            out[c] = np.clip(out[c], 0.0, 1.0)
    return out


def select_segmentation_channel(img: np.ndarray, nuc_channel: int) -> np.ndarray:
    """Pick the channel Cellpose-SAM will segment on.

    Cellpose-SAM is channel-agnostic and works fine on a single 2D plane,
    but giving it the cleanest nuclear / cytoplasmic channel almost always
    helps for fluorescence data.
    """
    if not 0 <= nuc_channel < img.shape[0]:
        raise IndexError(
            f"nuc_channel={nuc_channel} out of range for {img.shape[0]} channels"
        )
    return img[nuc_channel]


# ---------------------------------------------------------------------------
# 4. Create Model
# ---------------------------------------------------------------------------
def build_model(use_gpu: bool = True):
    """Equivalent to MONAI's `UNet(...).to(device)` step.

    Cellpose-SAM exposes exactly one pretrained model: 'cpsam'.
    Weights download to ~/.cellpose/models/ on first run.
    """
    models, _ = _import_cellpose()
    print(f"[model] loading Cellpose-SAM (cpsam), gpu={use_gpu}")
    model = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")
    return model


# ---------------------------------------------------------------------------
# 5. Execute — segment one image
# ---------------------------------------------------------------------------
def segment(
    model,
    seg_image: np.ndarray,
    is_3d: bool,
    diameter: Optional[float] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
) -> np.ndarray:
    """Run model.eval() on a single image; return integer instance mask.

    Mirrors the MONAI training/inference call. `diameter=None` lets
    Cellpose-SAM auto-size (range it was trained on: 7.5–120 px).
    """
    print(f"[seg] input shape={seg_image.shape}, 3D={is_3d}, diameter={diameter}")
    eval_kwargs = dict(
        batch_size=8,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        channel_axis=None,
        z_axis=0 if is_3d else None,
        do_3D=is_3d,
        normalize=True,         # cellpose does its own per-tile percentile norm
    )
    masks, _flows, _styles = model.eval(seg_image, **eval_kwargs)
    n_cells = int(masks.max())
    print(f"[seg] segmented {n_cells} cell(s)")
    return masks.astype(np.int32)


# ---------------------------------------------------------------------------
# 6. Quantify — per-cell intensity table across ALL channels
# ---------------------------------------------------------------------------
def quantify(masks: np.ndarray, img: np.ndarray, image_name: str) -> pd.DataFrame:
    """Build a tidy per-cell DataFrame: morphology + per-channel mean intensity.

    Roughly the SpinningMonkAI 'extract features per ROI' step.
    """
    if masks.max() == 0:
        return pd.DataFrame()

    base_props = ("label", "area", "centroid", "eccentricity",
                  "major_axis_length", "minor_axis_length", "solidity")
    # regionprops_table needs a 2D mask + 2D intensity image; loop channels.
    # For 3D, use the same approach on the 3D mask + 3D channel volume.
    per_channel_means: dict[str, np.ndarray] = {}
    for c in range(img.shape[0]):
        props = regionprops_table(
            masks, intensity_image=img[c],
            properties=("label", "intensity_mean", "intensity_max"),
        )
        per_channel_means[f"ch{c}_mean"] = props["intensity_mean"]
        per_channel_means[f"ch{c}_max"] = props["intensity_max"]

    morph = regionprops_table(masks, properties=base_props)
    df = pd.DataFrame(morph)
    for k, v in per_channel_means.items():
        df[k] = v
    df.insert(0, "image", image_name)
    return df


# ---------------------------------------------------------------------------
# 7. Save outputs — masks, overlays, table
# ---------------------------------------------------------------------------
def _random_label_cmap(n: int, seed: int = 0) -> ListedColormap:
    rng = np.random.default_rng(seed)
    colors = rng.random((max(n, 1) + 1, 3))
    colors[0] = (0, 0, 0)  # background
    return ListedColormap(colors)


def save_outputs(
    out_dir: Path,
    name: str,
    seg_img: np.ndarray,
    full_img: np.ndarray,
    masks: np.ndarray,
    df: pd.DataFrame,
) -> list[Path]:
    """Write mask TIFF, overlay PNG, and per-image CSV. Returns paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # --- mask as int32 TIFF (lossless, opens in Fiji / napari / QuPath) ---
    mask_path = out_dir / f"{name}_masks.tif"
    tifffile.imwrite(str(mask_path), masks.astype(np.int32))
    paths.append(mask_path)

    # --- per-image CSV ---
    csv_path = out_dir / f"{name}_cells.csv"
    df.to_csv(csv_path, index=False)
    paths.append(csv_path)

    # --- overlay PNG (only meaningful for 2D — show mid-slice for 3D) ---
    if seg_img.ndim == 3:
        seg2d = seg_img[seg_img.shape[0] // 2]
        m2d = masks[masks.shape[0] // 2]
    else:
        seg2d = seg_img
        m2d = masks

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(seg2d, cmap="gray")
    axes[0].set_title("segmentation channel")
    axes[1].imshow(m2d, cmap=_random_label_cmap(int(m2d.max())))
    axes[1].set_title(f"masks (n={int(m2d.max())})")
    boundaries = find_boundaries(m2d, mode="outer")
    overlay = np.stack([seg2d] * 3, axis=-1)
    overlay = (overlay - overlay.min()) / (np.ptp(overlay) + 1e-9)
    overlay[boundaries] = [1.0, 0.2, 0.2]   # red outlines
    axes[2].imshow(overlay)
    axes[2].set_title("overlay")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    overlay_path = out_dir / f"{name}_overlay.png"
    fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(overlay_path)
    return paths


# ---------------------------------------------------------------------------
# 8. GitHub CLI automation
# ---------------------------------------------------------------------------
def _run(cmd: list[str], cwd: Optional[Path] = None, check: bool = True) -> str:
    """Thin wrapper around subprocess.run that streams stderr to our stderr."""
    print(f"[gh] $ {' '.join(cmd)}")
    proc = subprocess.run(
        cmd, cwd=cwd, check=check,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
    return proc.stdout


def git_commit_and_push(
    repo_dir: Path,
    files: list[Path],
    message: str,
    branch: Optional[str] = None,
    create_pr: bool = False,
) -> None:
    """Add + commit + push results via git/gh. Idempotent: skips if nothing changed.

    Prereqs (one-time):
        gh auth login
        git config --global user.email "you@example.com"
        git config --global user.name  "Your Name"
    """
    if shutil.which("git") is None:
        raise RuntimeError("git not found on PATH")
    if create_pr and shutil.which("gh") is None:
        raise RuntimeError("GitHub CLI (gh) not found on PATH; install from https://cli.github.com/")

    # init if needed (so the function also works on a fresh output folder)
    if not (repo_dir / ".git").exists():
        _run(["git", "init", "-b", "main"], cwd=repo_dir)

    if branch:
        # create/switch (-B is create-or-reset; safer for automation)
        _run(["git", "checkout", "-B", branch], cwd=repo_dir)

    rel = [str(p.relative_to(repo_dir)) for p in files if p.exists()]
    if not rel:
        print("[gh] no files to commit")
        return
    _run(["git", "add", *rel], cwd=repo_dir)

    # `git diff --cached --quiet` exits 1 if there is staged change, 0 if clean
    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=repo_dir,
    )
    if diff.returncode == 0:
        print("[gh] no staged changes; nothing to commit")
        return

    _run(["git", "commit", "-m", message], cwd=repo_dir)

    # only push if a remote is configured
    remotes = _run(["git", "remote"], cwd=repo_dir).strip().splitlines()
    if not remotes:
        print("[gh] no remote configured; skipping push. "
              "Add one with: gh repo create <name> --source=. --remote=origin --push")
        return

    push_args = ["git", "push", "-u", "origin", branch] if branch else ["git", "push"]
    _run(push_args, cwd=repo_dir)

    if create_pr:
        _run(["gh", "pr", "create", "--fill", "--base", "main"], cwd=repo_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path,
                   help="image file or folder of images")
    p.add_argument("--output", required=True, type=Path,
                   help="folder to write masks / overlays / CSVs into")
    p.add_argument("--pattern", default="*.tif",
                   help="glob (used only when --input is a folder)")
    p.add_argument("--nuc-channel", type=int, default=0,
                   help="0-based index of the channel to segment on (DAPI by default)")
    p.add_argument("--diameter", type=float, default=None,
                   help="expected cell diameter in px; None lets Cellpose-SAM choose")
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--cellprob-threshold", type=float, default=0.0)
    p.add_argument("--no-gpu", action="store_true", help="force CPU (slow)")
    p.add_argument("--git-push", action="store_true",
                   help="git add/commit/push results after each run")
    p.add_argument("--git-message", default="auto: cellpose-SAM segmentation results")
    p.add_argument("--git-branch", default=None,
                   help="branch to push to (created if missing)")
    p.add_argument("--git-pr", action="store_true",
                   help="open a PR with `gh pr create --fill` after push")
    return p.parse_args()


def gather_inputs(in_path: Path, pattern: str) -> list[Path]:
    if in_path.is_file():
        return [in_path]
    if in_path.is_dir():
        return sorted(Path(p) for p in glob.glob(str(in_path / pattern)))
    raise FileNotFoundError(in_path)


def main() -> int:
    args = parse_args()
    files = gather_inputs(args.input, args.pattern)
    if not files:
        print(f"[main] no inputs found at {args.input}")
        return 1
    print(f"[main] {len(files)} image(s) to process")

    args.output.mkdir(parents=True, exist_ok=True)
    model = build_model(use_gpu=not args.no_gpu)

    all_rows: list[pd.DataFrame] = []
    produced: list[Path] = []

    for f in files:
        print("\n" + "-" * 60)
        print(f"[main] {f.name}")
        try:
            loaded = load_imaged(f)
            scaled = scale_intensity_ranged(loaded.array)
            seg_img = select_segmentation_channel(scaled, args.nuc_channel)
            masks = segment(
                model, seg_img,
                is_3d=loaded.is_3d,
                diameter=args.diameter,
                flow_threshold=args.flow_threshold,
                cellprob_threshold=args.cellprob_threshold,
            )
            df = quantify(masks, scaled, image_name=f.name)
            paths = save_outputs(args.output, f.stem, seg_img, scaled, masks, df)
            produced.extend(paths)
            all_rows.append(df)
        except Exception as exc:                # noqa: BLE001
            print(f"[main] FAILED on {f.name}: {exc}")

    # --- combined results table -------------------------------------------
    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined_path = args.output / "all_cells.csv"
        combined.to_csv(combined_path, index=False)
        produced.append(combined_path)
        print(f"\n[main] wrote {combined_path} "
              f"({len(combined)} cells across {combined['image'].nunique()} images)")

    # --- optional GitHub push ---------------------------------------------
    if args.git_push:
        try:
            git_commit_and_push(
                repo_dir=args.output,
                files=produced,
                message=args.git_message,
                branch=args.git_branch,
                create_pr=args.git_pr,
            )
        except Exception as exc:                # noqa: BLE001
            print(f"[gh] push failed: {exc}", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
