# VETI

Takes a photo and per-object segmentation masks, runs intrinsic decomposition + soft colour unmixing (FSCS/SCU), builds a MoGe depth mesh, and outputs a self-contained Blender scene with an editable colour palette and per-object material controls.

## Dependencies

```
torch  torchvision  opencv-contrib-python  numpy  Pillow  trimesh  utils3d
```

Local packages (clone into repo root):
- `intrinsic/` — intrinsic image decomposition (v2)
- `MoGe/` — monocular geometry estimation
- `FSCS/src/` — fast soft colour segmentation

Also needs **Blender** for the second stage.

## Usage

**Stage 1 — pipeline:**
```bash
python pipeline_final.py --img <image> --mask_dir <mask_folder> --output_dir ./output
```

**Stage 2 — Blender scene:**
```bash
blender --background --python output/blender_palette.py -- --output_dir output
```

Edit `output/materials.json` to adjust per-object roughness/metallic/specular, then re-run stage 2 to rebuild the `.blend`.

## Masks

One `<name>_mask.png` per object in the mask folder. `bg_mask.png` is optional — if missing, background is inferred from the remaining masks.

## Key flags

| Flag | Default | Effect |
|------|---------|--------|
| `--tau` | 5.0 | Colour threshold — lower means more palette layers |
| `--min_vote` | 100 | Higher means fewer layers |
| `--device` | cuda | Falls back to cpu if CUDA unavailable |
| `--no_edge_cull` | off | Skip depth-edge masking on the mesh |
| `--fill_mask` | off | Use alpha mask without valid-depth filter |
