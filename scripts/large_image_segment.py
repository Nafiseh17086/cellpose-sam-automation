#!/usr/bin/env python3
"""
large_image_segment.py — Cellpose-SAM for whole-slide and multiplex OME-TIFFs.

Designed for huge OME-TIFFs (~30,000 x 31,000 pixels, 10-40 channels) where:
  - the images don't fit in RAM
  - dimensions vary by a few pixels between files
  - you want to segment on the nuclear channel only
  - downstream you may want to stack the sections (which needs matching shapes)

Three subcommands:
  inspect   — print shape, dtype, channels of every OME-TIFF in a folder
  segment   — Cellpose-SAM with memory-mapped tiled inference
  pad       — pad all masks to the largest common (H, W) for stacking

Usage:
    python scripts/large_image_segment.py inspect /path/to/ome_tiffs
    python scripts/large_image_segment.py segment /path/to/ome_tiffs \\
        --output ./out --nuc-channel 0 --tile 4096 --overlap 256
    python scripts/large_image_segment.py pad ./out --pattern "*_masks.tif"
"""
from __future__ import annotations
import argparse, glob, gc, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops_table


def cmd_inspect(args):
    folder = Path(args.folder)
    files = sorted(folder.glob(args.pattern))
    if not files:
        print(f"[inspect] no files matching {args.pattern} in {folder}")
        return 1
    print(f"\n[inspect] {len(files)} file(s) in {folder}\n")
    rows = []
    for p in files:
        try:
            with tifffile.TiffFile(p) as tif:
                s = tif.series[0]
                axes, shape, dtype = s.axes, tuple(s.shape), str(s.dtype)
                c_idx = axes.index("C") if "C" in axes else None
                y_idx = axes.index("Y") if "Y" in axes else None
                x_idx = axes.index("X") if "X" in axes else None
                channels = shape[c_idx] if c_idx is not None else 1
                H = shape[y_idx] if y_idx is not None else "?"
                W = shape[x_idx] if x_idx is not None else "?"
                size_gb = p.stat().st_size / 1e9
                ome = tif.ome_metadata
                ch_names = []
                if ome and "<Channel" in ome:
                    import re
                    ch_names = re.findall(r'<Channel[^>]*Name="([^"]+)"', ome)[:channels]
            print(f"  {p.name}")
            print(f"    axes:       {axes}")
            print(f"    shape:      {shape}    dtype: {dtype}")
            print(f"    H x W:      {H} x {W}")
            print(f"    channels:   {channels}")
            if ch_names:
                print(f"    ch. names:  {', '.join(ch_names[:6])}"
                      + (f"  ... (+{len(ch_names)-6} more)" if len(ch_names) > 6 else ""))
            print(f"    file size:  {size_gb:.2f} GB\n")
            rows.append((p.name, H, W))
        except Exception as e:
            print(f"  {p.name}  -- failed: {e}\n")
    hw_set = {(r[1], r[2]) for r in rows if isinstance(r[1], int)}
    if len(hw_set) > 1:
        print(f"[inspect] WARNING: {len(hw_set)} different (H x W) shapes:")
        for hw in sorted(hw_set):
            print(f"    {hw}")
        print("\n  -> for stacking, run the `pad` subcommand after segmenting.\n")
    elif len(hw_set) == 1:
        print(f"[inspect] all files share (H x W) = {next(iter(hw_set))}\n")
    return 0


def load_channel(path, nuc_channel):
    with tifffile.TiffFile(path) as tif:
        s = tif.series[0]
        axes = s.axes
    arr = tifffile.memmap(path)
    sl = [slice(None)] * arr.ndim
    if "C" in axes: sl[axes.index("C")] = nuc_channel
    if "T" in axes: sl[axes.index("T")] = 0
    if "Z" in axes: sl[axes.index("Z")] = arr.shape[axes.index("Z")] // 2
    plane = np.array(arr[tuple(sl)])
    if plane.ndim != 2: plane = plane.squeeze()
    return plane, axes


def percentile_normalize(plane, lo=1.0, hi=99.5):
    a, b = np.percentile(plane, (lo, hi))
    if b <= a: return np.zeros_like(plane, dtype=np.float32)
    out = (plane.astype(np.float32) - a) / (b - a)
    return np.clip(out, 0.0, 1.0)


def tiled_segment(model, plane, tile, overlap, flow_threshold, cellprob_threshold):
    H, W = plane.shape
    stride = tile - overlap
    n_y = max(1, int(np.ceil((H - overlap) / stride)))
    n_x = max(1, int(np.ceil((W - overlap) / stride)))
    print(f"[seg] {n_y} x {n_x} outer tiles  (tile={tile}, overlap={overlap})")
    out = np.zeros((H, W), dtype=np.int32)
    next_id = 1
    t0 = time.time()
    for iy in range(n_y):
        for ix in range(n_x):
            y0 = min(iy * stride, max(0, H - tile));  y1 = min(y0 + tile, H)
            x0 = min(ix * stride, max(0, W - tile));  x1 = min(x0 + tile, W)
            patch = plane[y0:y1, x0:x1]
            if patch.size == 0: continue
            t_tile = time.time()
            masks, _, _ = model.eval(
                patch, batch_size=4,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                normalize=True, augment=False, tile_overlap=0.1)
            masks = masks.astype(np.int32)
            n_local = int(masks.max())
            if n_local == 0:
                print(f"  tile ({iy},{ix}) -> 0 cells ({time.time()-t_tile:.1f}s)")
                continue
            masks[masks > 0] += (next_id - 1)
            ov = out[y0:y1, x0:x1]
            for new_id in np.unique(masks):
                if new_id == 0: continue
                mask_new = masks == new_id
                vals, counts = np.unique(ov[mask_new], return_counts=True)
                if len(vals) > 1 or (len(vals) == 1 and vals[0] != 0):
                    nz = vals != 0
                    if nz.any():
                        winner = vals[nz][counts[nz].argmax()]
                        masks[mask_new] = winner
            empty = ov == 0
            ov[empty] = np.where(masks[empty] > 0, masks[empty], 0)
            out[y0:y1, x0:x1] = ov
            next_id = int(out.max()) + 1
            print(f"  tile ({iy},{ix}) -> +{n_local} cells "
                  f"({time.time()-t_tile:.1f}s)   total: {next_id - 1}")
            del masks, patch; gc.collect()
    print(f"[seg] total {int(out.max())} cells in {time.time()-t0:.1f}s")
    return out


def cmd_segment(args):
    in_path = Path(args.input)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    files = [in_path] if in_path.is_file() else sorted(in_path.glob(args.pattern))
    if not files: print(f"[segment] no inputs"); return 1
    from cellpose import models
    print(f"[model] loading Cellpose-SAM (cpsam), gpu={not args.no_gpu}")
    model = models.CellposeModel(gpu=not args.no_gpu, pretrained_model="cpsam")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for f in files:
        print("\n" + "=" * 60); print(f"[segment] {f.name}")
        try:
            plane, axes = load_channel(f, args.nuc_channel)
            print(f"[load] axes={axes}, plane shape={plane.shape}")
            plane = percentile_normalize(plane)
            masks = tiled_segment(model, plane, args.tile, args.overlap,
                                  args.flow_threshold, args.cellprob_threshold)
            mp = out_dir / f"{f.stem}_masks.tif"
            tifffile.imwrite(str(mp), masks, bigtiff=True,
                             compression="zlib", tile=(512, 512))
            if masks.max() > 0:
                df = pd.DataFrame(regionprops_table(
                    masks, properties=("label", "area", "centroid",
                                       "eccentricity", "major_axis_length",
                                       "minor_axis_length", "solidity")))
                p = regionprops_table(masks, intensity_image=plane,
                                      properties=("label", "intensity_mean",
                                                  "intensity_max"))
                df["nuc_mean"] = p["intensity_mean"]
                df["nuc_max"]  = p["intensity_max"]
                df.insert(0, "image", f.name)
                cp = out_dir / f"{f.stem}_cells.csv"
                df.to_csv(cp, index=False)
                all_rows.append(df)
                print(f"[save] {mp.name} + {cp.name}: {len(df)} cells")
        except Exception as e:
            print(f"[segment] FAILED: {e}")
            import traceback; traceback.print_exc()
    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        cp = out_dir / "all_cells.csv"; combined.to_csv(cp, index=False)
        print(f"\n[segment] wrote {cp}: {len(combined)} cells")
    return 0


def cmd_pad(args):
    folder = Path(args.folder)
    files = sorted(folder.glob(args.pattern))
    if not files: print(f"[pad] no files"); return 1
    shapes = []
    for p in files:
        with tifffile.TiffFile(p) as tif:
            shapes.append(tif.series[0].shape)
    H_max = max(s[-2] for s in shapes); W_max = max(s[-1] for s in shapes)
    print(f"[pad] padding all to (..., {H_max}, {W_max})")
    pad_dir = folder / "padded"; pad_dir.mkdir(exist_ok=True)
    for p in files:
        img = tifffile.imread(p)
        H, W = img.shape[-2], img.shape[-1]
        if (H, W) == (H_max, W_max):
            print(f"  {p.name}  already at target shape")
            tifffile.imwrite(str(pad_dir / p.name), img, bigtiff=True, compression="zlib")
            continue
        pad_widths = [(0, 0)] * (img.ndim - 2) + [(0, H_max - H), (0, W_max - W)]
        padded = np.pad(img, pad_widths, mode="constant", constant_values=0)
        tifffile.imwrite(str(pad_dir / p.name), padded, bigtiff=True, compression="zlib")
        print(f"  {p.name}  {img.shape} -> {padded.shape}")
    print(f"\n[pad] wrote {len(files)} file(s) to {pad_dir}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    pi = sub.add_parser("inspect")
    pi.add_argument("folder", type=Path); pi.add_argument("--pattern", default="*.ome.tif")
    ps = sub.add_parser("segment")
    ps.add_argument("input", type=Path); ps.add_argument("--output", type=Path, required=True)
    ps.add_argument("--pattern", default="*.ome.tif")
    ps.add_argument("--nuc-channel", type=int, default=0)
    ps.add_argument("--tile", type=int, default=4096)
    ps.add_argument("--overlap", type=int, default=256)
    ps.add_argument("--flow-threshold", type=float, default=0.4)
    ps.add_argument("--cellprob-threshold", type=float, default=0.0)
    ps.add_argument("--no-gpu", action="store_true")
    pp = sub.add_parser("pad")
    pp.add_argument("folder", type=Path); pp.add_argument("--pattern", default="*_masks.tif")
    args = p.parse_args()
    return {"inspect": cmd_inspect, "segment": cmd_segment, "pad": cmd_pad}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
