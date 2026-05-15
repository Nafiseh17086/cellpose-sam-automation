# Large-image workflow: whole-slide and multiplex OME-TIFFs

For data like Ryan's — huge OME-TIFFs (~30,000 × 31,000 px each, many channels,
dimensions varying by a few pixels between files) — use
`scripts/large_image_segment.py` instead of the regular CLI.

## Three subcommands

### 1. inspect — look first

```bash
python scripts/large_image_segment.py inspect /path/to/ome_tiffs
```

Prints shape, channels, channel names, and file size for every file.
Warns if shapes don't match.

### 2. segment — Cellpose-SAM with tiled inference

```bash
python scripts/large_image_segment.py segment /path/to/ome_tiffs \
    --output ./out --pattern "*.ome.tif" --nuc-channel 0 \
    --tile 4096 --overlap 256
```

Memory-mapped load, 4096×4096 outer tiles with 256 px overlap, label-merging
across tiles. Writes BigTIFF masks + per-image CSVs.

### 3. pad — equalize shapes for stacking

```bash
python scripts/large_image_segment.py pad ./out --pattern "*_masks.tif"
```

Zero-pads all masks to the largest (H, W) for serial-section stacking.

⚠️ Padding aligns *shapes*, not *content*. For true serial-section
registration use pystackreg or VALIS.

## Performance

Per 30k × 30k image: ~2-4 min on A100, ~20-40 min on Apple Silicon (MPS).

## Troubleshooting

- **CUDA OOM**: lower `--tile` to 2048, or add `--no-gpu`
- **0 cells**: wrong `--nuc-channel` — run `inspect` again
- **Cells split at tile borders**: raise `--overlap` to 512
