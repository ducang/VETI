import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

from PIL import Image as PILImage
import cv2
import numpy as np
import torch
import trimesh
import utils3d

from color_model import extract_color_model
from colorunmixing import (
    solver_SCU, matte_regu, color_refine, distr_to_torch,
    DEVICE as UNMIX_DEVICE, DTYPE as UNMIX_DTYPE,
)
from moge.model.v2 import MoGeModel
from intrinsic.pipeline import load_models as load_intrinsic_models
from intrinsic.pipeline import run_pipeline as run_intrinsic_pipeline

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "FSCS" / "src"))
sys.path.insert(0, str(REPO_ROOT / "MoGe"))

ALPHA_THRESHOLD = 1 / 255.0
MESH_DEPTH_EDGE_RTOL = 0.02

DEFAULT_MATERIAL = {
    "roughness": 0.7, 
    "metallic": 0.0, 
    "specular": 0.1,
}

MATERIAL_CHANNELS = list(DEFAULT_MATERIAL.keys())

# Blender renamed the Specular socket between 3.x and 4.x/5.x.
BSDF_SOCKET_CANDIDATES = {
    "roughness": ["Roughness"],
    "metallic": ["Metallic"],
    "specular": ["Specular IOR Level", "Specular"],
}

MATERIAL_PRESETS = [
    {"preset": "chalk", "roughness": 1.00, "metallic": 0.0, "specular": 0.00},
    {"preset": "matte", "roughness": 0.90, "metallic": 0.0, "specular": 0.15},
    {"preset": "plastic", "roughness": 0.40, "metallic": 0.0, "specular": 0.50},
    {"preset": "brushed_metal", "roughness": 0.40, "metallic": 1.0, "specular": 0.50},
    {"preset": "mirror", "roughness": 0.02, "metallic": 1.0, "specular": 1.00},
]


def extract_albedo(img_bgr, out_dir, input_alpha=None):
    intrinsic_model = load_intrinsic_models('v2')

    alb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    results = run_intrinsic_pipeline(intrinsic_model, alb)

    albedo_rgb = np.clip(results['hr_alb'], 0.0, 1.0).astype(np.float32)

    if input_alpha is not None:
        alpha = input_alpha.astype(np.float32)
        if alpha.shape[:2] != albedo_rgb.shape[:2]:
            alpha = cv2.resize(alpha, (albedo_rgb.shape[1], albedo_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        albedo_rgb = albedo_rgb * alpha[:, :, None]

    albedo_bgr = (cv2.cvtColor(albedo_rgb, cv2.COLOR_RGB2BGR) * 255).round().astype(np.uint8)
    cv2.imwrite(str(out_dir / "albedo.png"), albedo_bgr)
    print(f"albedo -> {out_dir}/albedo.png")
    return albedo_bgr


def run_soft_color_seg(
    albedo_bgr,
    input_alpha,
    layers_dir,
    device,
    tau=5.0,
    matte_radius=60,
    gf_eps=1e-2,
    min_vote=100,
    min_eig_scu=5e-3,
    min_eig_rep=1e-4,
    gf_radius=None,
    neighborhood_radius=None,
    sigmaa=10.0
):
    layers_dir = Path(layers_dir)
    layers_dir.mkdir(parents=True, exist_ok=True)

    solver_device = UNMIX_DEVICE
    solver_dtype = UNMIX_DTYPE
    return_device = torch.device(device)
    return_debug = False
    alb = cv2.cvtColor(albedo_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    h, w = alb.shape[:2]

    t0 = time.time()
    color_model_out = extract_color_model(
        alb,
        tau=tau,
        gf_eps=gf_eps,
        min_vote=min_vote,
        min_eig_scu=min_eig_scu,
        min_eig_rep=min_eig_rep,
        gf_radius=gf_radius,
        neighborhood_radius=neighborhood_radius,
        return_debug=return_debug,
    )

    color_distr = color_model_out

    if not return_debug :
        color_distr, _ = color_model_out

    if len(color_distr) == 0:
        raise RuntimeError("Colour model extraction produced zero distributions.")

    N = len(color_distr)
    print(f"Colour model: {N} distributions ({time.time() - t0:.1f}s)")

    color_distr = distr_to_torch(color_distr)
    mus = torch.stack([d["mu"] for d in color_distr], dim=0).contiguous()
    sigma_invs = torch.stack([d["sigma_inv"] for d in color_distr], dim=0).contiguous()

    P = h * w
    c = torch.tensor(alb.reshape(P, 3), device=solver_device, dtype=solver_dtype)

    t0 = time.time()
    alphas_flat, colors_flat = solver_SCU(c, mus, sigma_invs, sigmaa=sigmaa)
    print(f"SCU unmixing: {time.time() - t0:.1f}s")

    alphas_np = alphas_flat.detach().cpu().numpy().reshape(h, w, N)
    colors_np = colors_flat.detach().cpu().numpy().reshape(h, w, N, 3)

    t0 = time.time()
    matreg_np = matte_regu(alb, alphas_np, rad=matte_radius)
    print(f"Matte regularisation: {time.time() - t0:.1f}s")

    t0 = time.time()
    matreg_flat = torch.tensor(matreg_np.reshape(P, N), device=solver_device, dtype=solver_dtype)
    colors_init = torch.tensor(colors_np.reshape(P, N, 3), device=solver_device, dtype=solver_dtype)
    final_a_flat, final_c_flat = color_refine(c, mus, sigma_invs, matreg_flat, colors_init)
    print(f"Colour refinement: {time.time() - t0:.1f}s")

    final_alphas = final_a_flat.detach().cpu().numpy().reshape(h, w, N)
    final_colors = final_c_flat.detach().cpu().numpy().reshape(h, w, N, 3)

    s = np.maximum(final_alphas.sum(axis=2, keepdims=True), 1e-8)
    final_alphas = np.clip(final_alphas / s, 0.0, 1.0)

    if input_alpha is not None:
        ia = input_alpha.astype(np.float32)
        if ia.shape[:2] != (h, w):
            ia = cv2.resize(ia, (w, h), interpolation=cv2.INTER_LINEAR)
        final_alphas = np.clip(final_alphas * ia[:, :, None], 0.0, 1.0)

    cv2.imwrite(str(layers_dir / "target_img.png"), albedo_bgr)

    for i in range(N):
        a = np.clip(final_alphas[:, :, i], 0.0, 1.0)
        col = np.clip(final_colors[:, :, i], 0.0, 1.0)

        rgb_u8 = (col * 255.0).round().astype(np.uint8)
        a_u8 = (a * 255.0).round().astype(np.uint8)

        bgra_u8 = np.dstack([rgb_u8[:, :, 2], rgb_u8[:, :, 1], rgb_u8[:, :, 0], a_u8])
        cv2.imwrite(str(layers_dir / f"layer-{i:02d}.png"), bgra_u8)
        cv2.imwrite(str(layers_dir / f"alpha-{i:02d}.png"), a_u8)

    clipped_mus = np.clip(np.array([d["mu"] for d in color_distr]), 0.0, 1.0)
    swatch = np.zeros((50, N * 80, 3), dtype=np.uint8)
    for i, mu in enumerate(clipped_mus):
        swatch[:, i * 80:(i + 1) * 80] = (mu * 255.0).astype(np.uint8)
    cv2.imwrite(str(layers_dir / "palette.png"), cv2.cvtColor(swatch, cv2.COLOR_RGB2BGR))

    print(f"Saved {N} layers + reconstruction -> {layers_dir}/")

    alpha = torch.from_numpy(final_alphas.transpose(2, 0, 1).astype(np.float32)).unsqueeze(1).to(return_device)
    rgb_layers = torch.from_numpy(np.clip(final_colors, 0.0, 1.0).transpose(2, 3, 0, 1).astype(np.float32)).to(return_device)

    palette = [
        {
            "cluster": i,
            "color_rgb": [float(mu[0]), float(mu[1]), float(mu[2])],
        }
        for i, mu in enumerate(clipped_mus)
    ]
    return palette, rgb_layers, alpha

def export_alpha_textures(out_dir, alpha, palette, image_hw):
    tex_dir = out_dir / "alpha_textures"
    tex_dir.mkdir(parents=True, exist_ok=True)

    h, w = image_hw
    alpha_np = alpha[:, 0, :, :].cpu().numpy()
    K = alpha_np.shape[0]

    if alpha_np.shape[1:] != (h, w):
        alpha_np = np.stack([cv2.resize(alpha_np[i], (w, h), interpolation=cv2.INTER_LINEAR) for i in range(K)])

    total = np.maximum(alpha_np.sum(axis=0, keepdims=True), 1e-6)
    normed = np.clip(alpha_np / total, 0.0, 1.0)

    tex_paths = []
    for i, entry in enumerate(palette):
        fname = f"weight_{int(entry['cluster']):02d}.png"
        cv2.imwrite(str(tex_dir / fname), cv2.flip((normed[i] * 65535).astype(np.uint16), 0))
        rel = f"alpha_textures/{fname}"
        tex_paths.append(rel)
        entry["alpha_texture"] = rel

    with open(out_dir / "palette.json", "w") as f:
        json.dump(palette, f, indent=2)
    print(f"  Exported {K} alpha textures -> {tex_dir}/")
    return tex_paths


def export_object_masks(mask_dir, out_dir, image_hw):
    h, w = image_hw
    od = out_dir / "object_masks"
    od.mkdir(parents=True, exist_ok=True)

    masks, obj_names = [], []
    bg_mask = None

    semantic_files = sorted(p for ext in ("png", "jpg", "jpeg") for p in mask_dir.glob(f"*_mask.{ext}"))
    for p in semantic_files:
        stem = p.stem
        semantic = stem[:-5] if stem.lower().endswith("_mask") else stem
        arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        m = arr.astype(np.float32) / 255.0
        if semantic.lower() == "bg":
            bg_mask = m
        else:
            masks.append(m)
            obj_names.append(semantic)

    rel_paths = []
    accum = np.zeros((h, w), np.float32)
    for name, m in zip(obj_names, masks):
        accum += m
        fname = f"{name}_mask.png"

        cv2.imwrite(str(od / fname), cv2.flip((m * 255).astype(np.uint8), 0))
        rel_paths.append(f"object_masks/{fname}")

    bg = bg_mask if bg_mask is not None else np.clip(1.0 - accum, 0.0, 1.0)
    cv2.imwrite(str(od / "bg_mask.png"), cv2.flip((bg * 255).astype(np.uint8), 0))
    rel_paths.append("object_masks/bg_mask.png")
    obj_names.append("bg")

    print(f"  Exported {len(obj_names)} object masks: {obj_names}")
    return rel_paths, obj_names


def write_materials_json(out_dir, obj_names):
    mp = out_dir / "materials.json"
    existing = {}
    if mp.exists():
        with open(mp) as f:
            for entry in json.load(f):
                key = entry.get("name") or str(entry.get("object_id", ""))
                if key:
                    existing[key] = entry

    materials = []
    for name in obj_names:
        entry = dict(existing.get(name, {}))
        entry["name"] = name
        for k, v in DEFAULT_MATERIAL.items():
            entry.setdefault(k, v)
        materials.append(entry)

    with open(mp, "w") as f:
        json.dump(materials, f, indent=2)
    print(f"Wrote materials.json ({len(materials)} entries)")
    return materials


def write_material_presets(out_dir):
    mp = out_dir / "materials_presets.json"
    with open(mp, "w") as f:
        json.dump(MATERIAL_PRESETS, f, indent=2)
    print(f"Wrote {len(MATERIAL_PRESETS)} material presets")


def build_mesh(image_path, out_dir, moge_model_name, device, no_edge_cull=False, fill_mask=False):
    arr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        bgr, alpha = arr[:, :, :3], arr[:, :, 3].astype(np.float32) / 255.0
    else:
        bgr, alpha = arr[:, :, :3], np.ones(arr.shape[:2], np.float32)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    print("  Loading MoGe model...")
    model = MoGeModel.from_pretrained(moge_model_name).to(device)
    input_t = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)

    t0 = time.time()
    output = model.infer(input_t)
    print(f"  MoGe inference: {time.time()-t0:.2f}s")

    pts = output["points"].float().cpu().numpy()
    depth = output["depth"].float().cpu().numpy()
    normal = output["normal"].float().cpu().numpy() if output.get("normal") is not None else None
    valid = output["mask"].float().cpu().numpy() > 0.5
    intrinsics = output["intrinsics"].cpu().numpy().tolist()
    h, w = pts.shape[:2]

    if alpha.shape[:2] != (h, w):
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    opaque = alpha >= ALPHA_THRESHOLD

    mask = opaque if fill_mask else (opaque & valid)
    if not no_edge_cull and math.isfinite(MESH_DEPTH_EDGE_RTOL):
        mask = mask & ~utils3d.depth_map_edge(depth, rtol=MESH_DEPTH_EDGE_RTOL)

    rgb_float = rgb.astype(np.float32) / 255.0
    if rgb_float.shape[:2] != (h, w):
        rgb_float = cv2.resize(rgb_float, (w, h))

    uv = utils3d.uv_map(h, w)
    if normal is None:
        faces, verts, vcolors, vuvs = utils3d.build_mesh_from_map(pts, rgb_float, uv, mask=mask, tri=True)
        vnormals = None
    else:
        faces, verts, vcolors, vuvs, vnormals = utils3d.build_mesh_from_map(pts, rgb_float, uv, normal, mask=mask, tri=True)

    verts = verts * np.array([1, -1, -1], dtype=np.float32)
    if vnormals is not None:
        vnormals = vnormals * np.array([1, -1, -1], dtype=np.float32)

    rgb_u8 = (np.clip(vcolors, 0, 1) * 255).astype(np.uint8)
    rgba = np.concatenate([rgb_u8, np.full((len(rgb_u8), 1), 255, np.uint8)], axis=1)

    mesh_dir = out_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    edge = utils3d.depth_map_edge(depth, rtol=MESH_DEPTH_EDGE_RTOL)
    seam_mask = ((~valid) | edge).astype(np.uint8) * 255

    seam_mask = cv2.dilate(seam_mask, np.ones((7, 7), np.uint8), iterations=1)

    cv2.imwrite(str(mesh_dir / "seam_mask.png"), cv2.flip(seam_mask, 0))
    tmesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=rgba, vertex_normals=vnormals, process=False)

    tex_path = mesh_dir / "base_texture.png"
    cv2.imwrite(str(tex_path), cv2.flip(bgr, 0))
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=PILImage.open(str(tex_path)), metallicFactor=0.0, roughnessFactor=0.5)
    tmesh.visual = trimesh.visual.TextureVisuals(uv=vuvs, material=material)

    glb_path = mesh_dir / "scene.glb"
    tmesh.export(str(glb_path))
    cv2.imwrite(str(mesh_dir / "texture_original.png"), bgr)
    print(f"Saved mesh -> {glb_path}")
    return intrinsics, h, w


BLENDER_SCRIPT = REPO_ROOT / "blender_palette.py"


def write_scene_config(out_dir, palette, h, w, intrinsics, obj_mask_paths, materials):
    config = {
        "num_layers": len(palette),
        "colors": [[*entry["color_rgb"], 1.0] for entry in palette],
        "tex_relpaths": [str((out_dir / entry["alpha_texture"]).resolve()) for entry in palette],
        "img_h": h,
        "img_w": w,
        "intrinsics": intrinsics,
        "glb_relpath": str((out_dir / "mesh/scene.glb").resolve()),
        "obj_names": [m["name"] for m in materials],
        "obj_mask_paths": [str((out_dir / p).resolve()) for p in obj_mask_paths],
        "material_channels": MATERIAL_CHANNELS,
        "bsdf_socket_candidates": BSDF_SOCKET_CANDIDATES,
        "default_material": DEFAULT_MATERIAL,
    }
    with open(out_dir / "scene_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Wrote scene_config.json")

    shutil.copy2(BLENDER_SCRIPT, out_dir / BLENDER_SCRIPT.name)
    print(f"Copied {BLENDER_SCRIPT.name} -> {out_dir}/")

def load_image_with_alpha(img_path):
    raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    if raw.ndim == 2:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR), None
    if raw.shape[2] == 4:
        return raw[:, :, :3], raw[:, :, 3]
    return raw, None


def parse_args():
    p = argparse.ArgumentParser(description="3D image manipulation pipeline.")
    p.add_argument("--img", required=True)
    p.add_argument("--mask_dir", required=True)
    p.add_argument("--output_dir", default="./output_final")
    p.add_argument("--no_edge_cull", action="store_true")
    p.add_argument("--fill_mask", action="store_true")
    p.add_argument("--moge_model", default="Ruicheng/moge-2-vitl-normal")
    p.add_argument("--tau", type=float, default=5.0,
                   help="Colour model representation threshold (lower = more layers)")
    p.add_argument("--gf_eps", type=float, default=1e-2,
                   help="Guided-filter epsilon for colour model estimation")
    p.add_argument("--min_vote", type=int, default=100,
                   help="Vote-count termination threshold; higher = fewer layers")
    p.add_argument("--min_eig_rep", type=float, default=5e-3,
                   help="Rep covariance floor (min_eig_rep); higher = more colors ")
    p.add_argument("--min_eig_scu", type=float, default=5e-3,
                   help="SCU covariance floor (min_eig_scu); higher = broader "
                        "distributions = less leaking but fewer distinct colours")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    out = Path(args.output_dir)
    layers_dir = out / "layers"
    out.mkdir(parents=True, exist_ok=True)

    img_path = Path(args.img)
    dst_img = out / img_path.name
    if img_path.resolve() != dst_img.resolve():
        shutil.copy2(img_path, dst_img)

    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    _, input_alpha = load_image_with_alpha(str(img_path))

    print("[1/6] Intrinsic decomposition...")
    albedo_bgr = extract_albedo(img_bgr, out, input_alpha=input_alpha)
    img_h, img_w = albedo_bgr.shape[:2]

    print("\n[2/6] Soft colour segmentation...")
    palette, _, alpha = run_soft_color_seg(
        albedo_bgr, input_alpha, layers_dir, device,
        tau=args.tau, gf_eps=args.gf_eps,
        min_vote=args.min_vote, min_eig_rep=args.min_eig_rep, min_eig_scu=args.min_eig_scu,
    )

    print("\n[3/6] Exporting alpha textures...")
    export_alpha_textures(out, alpha, palette, (img_h, img_w))

    print("\n[4/6] MoGe depth -> mesh...")
    intrinsics, h, w = build_mesh(dst_img, out, args.moge_model, device, no_edge_cull=args.no_edge_cull, fill_mask=args.fill_mask)

    print("\n[5/6] Object masks + materials...")
    obj_paths, obj_names = export_object_masks(Path(args.mask_dir), out, (h, w))
    materials = write_materials_json(out, obj_names)
    write_material_presets(out)

    print("\n[6/6] Writing Blender scene config...")
    write_scene_config(out, palette, h, w, intrinsics, obj_paths, materials)

    metadata = {
        "input_image": img_path.name, "image_width": w, "image_height": h,
        "intrinsics": intrinsics,
        "intrinsic_model": "v2", "no_edge_cull": bool(args.no_edge_cull),
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Output: {out}/")
    print(f"Next: blender --background --python {out}/blender_palette.py -- --output_dir {out}")
    print(f"(edit materials.json in {out}/ to tune per-object channels, then re-run)")


if __name__ == "__main__":
    main()
