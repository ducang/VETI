#!/usr/bin/env python3
"""Build a Blender scene from the pipeline output directory.

Reads `scene_config.json` (scene structure — colors, texture paths, object
names, intrinsics) and `materials.json` (editable per-object material values)
from the output directory, then assembles a mesh with a palette-based
Principled BSDF material, 3-point lighting, an orbit light, and a procedural
sky world.

The Principled BSDF is placed directly on the material (no wrapping node
group), so you can swap it for another shader (Toon, Diffuse, etc.) via the
Material Properties → Surface dropdown.

Usage:
    blender --background --python blender_palette.py -- --output_dir <dir>
"""

import bpy
import json
import math
import sys
from pathlib import Path

import mathutils


def parse_output_dir():
    argv = sys.argv
    if "--" in argv:
        args = argv[argv.index("--") + 1:]
        for i, a in enumerate(args):
            if a == "--output_dir" and i + 1 < len(args):
                return Path(args[i + 1])
            if a.startswith("--output_dir="):
                return Path(a.split("=", 1)[1])
    return Path(__file__).resolve().parent


def load_material_overrides(output_dir):
    path = output_dir / "materials.json"
    if not path.exists():
        return {}
    with open(path) as f:
        entries = json.load(f)
    return {e["name"]: e for e in entries if e.get("name")}


def channel_value(overrides, obj_name, channel, default):
    return float(overrides.get(obj_name, {}).get(channel, default))


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block)
    for block in list(bpy.data.materials):
        bpy.data.materials.remove(block)
    for block in list(bpy.data.cameras):
        bpy.data.cameras.remove(block)


def import_mesh(glb_path):
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    mesh = next((o for o in bpy.context.scene.objects if o.type == 'MESH'), None)
    if mesh is None:
        raise RuntimeError(f"No mesh found after importing {glb_path}")
    return mesh


def find_socket(node, candidates):
    for name in candidates:
        if name in node.inputs:
            return node.inputs[name]
    return None


def build_palette_material(cfg, overrides, output_dir, uv_name):
    mat = bpy.data.materials.new(name="PaletteMaterial")
    mat.use_nodes = True
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    out_node = nodes.new('ShaderNodeOutputMaterial')
    out_node.location = (1100, 0)

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (800, 0)
    links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])

    base_color = find_socket(bsdf, ['Base Color', 'Color'])

    accum = None
    for idx in range(cfg["num_layers"]):
        y = 200 - idx * 220

        uv = nodes.new('ShaderNodeUVMap')
        uv.location = (-1000, y)
        if uv_name:
            uv.uv_map = uv_name

        tex = nodes.new('ShaderNodeTexImage')
        tex.location = (-800, y)
        img = bpy.data.images.load(
            str(output_dir / cfg["tex_relpaths"][idx]), check_existing=False)
        img.colorspace_settings.name = 'Non-Color'
        img.pack()
        tex.image = img
        links.new(uv.outputs['UV'], tex.inputs['Vector'])

        rgb = nodes.new('ShaderNodeRGB')
        rgb.location = (-800, y - 150)
        rgb.outputs[0].default_value = cfg["colors"][idx]
        rgb.label = f"Color {idx}"

        mul = nodes.new('ShaderNodeMixRGB')
        mul.blend_type = 'MULTIPLY'
        mul.location = (-500, y)
        mul_fac = [s for s in mul.inputs if s.type == 'VALUE'][0]
        mul_in  = [s for s in mul.inputs if s.type == 'RGBA']
        mul_out = [s for s in mul.outputs if s.type == 'RGBA'][0]
        mul_fac.default_value = 1.0
        links.new(tex.outputs['Color'], mul_in[0])
        links.new(rgb.outputs['Color'], mul_in[1])

        if accum is None:
            accum = mul_out
        else:
            add = nodes.new('ShaderNodeMixRGB')
            add.blend_type = 'ADD'
            add.location = (-250, y)
            add_fac = [s for s in add.inputs if s.type == 'VALUE'][0]
            add_in  = [s for s in add.inputs if s.type == 'RGBA']
            add_out = [s for s in add.outputs if s.type == 'RGBA'][0]
            add_fac.default_value = 1.0
            if idx == cfg["num_layers"] - 1:
                if hasattr(add, 'use_clamp'):
                    add.use_clamp = True
                if hasattr(add, 'clamp_result'):
                    add.clamp_result = True
            links.new(accum, add_in[0])
            links.new(mul_out, add_in[1])
            accum = add_out

    if base_color is not None and accum is not None:
        links.new(accum, base_color)

    obj_mask_nodes = []
    for j, (name, rel) in enumerate(zip(cfg["obj_names"], cfg["obj_mask_paths"])):
        uv = nodes.new('ShaderNodeUVMap')
        uv.location = (-2200, -1200 - j * 220)
        if uv_name:
            uv.uv_map = uv_name

        tex = nodes.new('ShaderNodeTexImage')
        tex.location = (-2000, -1200 - j * 220)
        img = bpy.data.images.load(
            str(output_dir / rel), check_existing=False)
        img.colorspace_settings.name = 'Non-Color'
        img.pack()
        tex.image = img
        tex.label = f"ObjMask: {name}"
        tex.name = f"ObjMask_{name}"
        img.name = f"mask_{name}"
        links.new(uv.outputs['UV'], tex.inputs['Vector'])
        obj_mask_nodes.append(tex)

    defaults = cfg["default_material"]
    channels = cfg["material_channels"]
    socket_candidates = cfg["bsdf_socket_candidates"]

    for ci, ch in enumerate(channels):
        x_base = -1700 + ci * 700
        chain_out = None
        for j, obj_name in enumerate(cfg["obj_names"]):
            y = 200 - j * 180
            value = channel_value(overrides, obj_name, ch, defaults[ch])

            val_node = nodes.new('ShaderNodeValue')
            val_node.location = (x_base, y - 80)
            val_node.outputs[0].default_value = value
            val_node.label = f"{ch}: {obj_name}"
            val_node.name = f"{ch}_{obj_name}"

            mul = nodes.new('ShaderNodeMath')
            mul.operation = 'MULTIPLY'
            mul.location = (x_base + 250, y)
            links.new(obj_mask_nodes[j].outputs['Color'], mul.inputs[0])
            links.new(val_node.outputs[0], mul.inputs[1])

            if chain_out is None:
                chain_out = mul.outputs[0]
            else:
                add = nodes.new('ShaderNodeMath')
                add.operation = 'ADD'
                add.location = (x_base + 500, y)
                if j == len(cfg["obj_names"]) - 1:
                    add.use_clamp = True
                links.new(chain_out, add.inputs[0])
                links.new(mul.outputs[0], add.inputs[1])
                chain_out = add.outputs[0]

        target = find_socket(bsdf, socket_candidates[ch])
        if target is not None and chain_out is not None:
            links.new(chain_out, target)

    return mat


def setup_camera(intrinsics, img_w, img_h):
    bpy.ops.object.camera_add(location=(0, 0, 0))
    cam = bpy.context.object
    cam.name = "PaletteCam"
    fx_norm = intrinsics[0][0]
    cam.data.sensor_fit = 'HORIZONTAL'
    cam.data.angle = 2.0 * math.atan(0.5 / fx_norm)
    cam.data.clip_start = 0.01
    cam.data.clip_end = 1000.0
    cam.rotation_euler = (math.pi / 2, 0.0, 0.0)
    bpy.context.scene.camera = cam
    bpy.context.scene.render.resolution_x = img_w
    bpy.context.scene.render.resolution_y = img_h


def mesh_bbox(mesh_obj):
    mw = mesh_obj.matrix_world
    corners = [mw @ mathutils.Vector(c) for c in mesh_obj.bound_box]
    xs = [v.x for v in corners]
    ys = [v.y for v in corners]
    zs = [v.z for v in corners]
    centre = mathutils.Vector((
        (min(xs) + max(xs)) / 2,
        (min(ys) + max(ys)) / 2,
        (min(zs) + max(zs)) / 2,
    ))
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    return centre, span


def add_area_light(name, energy, size, color, location, centre):
    data = bpy.data.lights.new(name, type='AREA')
    data.energy = energy
    data.size = size
    data.color = color
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = (centre - location).normalized().to_track_quat('-Z', 'Y').to_euler()
    return obj


def setup_lighting(mesh_obj):
    centre, span = mesh_bbox(mesh_obj)
    orbit_r = span * 1.5
    key_energy = 50.0 * orbit_r
    fill_energy = key_energy * 0.4

    for obj in list(bpy.context.scene.objects):
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)

    add_area_light(
        "KeyLight", key_energy, span * 0.5, (1.0, 0.98, 0.92),
        centre + mathutils.Vector((orbit_r * 0.7, -orbit_r * 0.7, orbit_r * 0.8)),
        centre)
    add_area_light(
        "FillLight", fill_energy, span * 0.8, (0.88, 0.92, 1.0),
        centre + mathutils.Vector((-orbit_r * 0.6, orbit_r * 0.4, orbit_r * 0.3)),
        centre)

    # orbit light keeps moving in front of the mesh
    ORBIT_FRAMES = 120
    ORBIT_TILT = math.radians(30)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = ORBIT_FRAMES

    data = bpy.data.lights.new("OrbitLight", type='POINT')
    data.energy = key_energy * 0.8
    data.color = (1.0, 1.0, 1.0)
    data.shadow_soft_size = span * 0.05
    obj = bpy.data.objects.new("OrbitLight", data)
    scene.collection.objects.link(obj)

    front_y = -orbit_r * 0.7
    for frame in range(1, ORBIT_FRAMES + 1):
        t = (frame - 1) / ORBIT_FRAMES
        angle = 2.0 * math.pi * t
        x = orbit_r * math.cos(angle)
        z = orbit_r * math.sin(angle) * math.cos(ORBIT_TILT)
        y = front_y + orbit_r * math.sin(angle) * math.sin(ORBIT_TILT) * 0.15
        obj.location = centre + mathutils.Vector((x, y, z))
        obj.keyframe_insert(data_path="location", frame=frame)

    for fc in _get_fcurves(obj):
        for mod in list(fc.modifiers):
            fc.modifiers.remove(mod)
        mod = fc.modifiers.new(type='CYCLES')
        mod.mode_before = 'REPEAT'
        mod.mode_after = 'REPEAT'


def _get_fcurves(obj):
    # Blender 4.x has action.fcurves directly; 5.x uses layers/strips/channelbags
    ad = obj.animation_data
    if ad is None or ad.action is None:
        return []
    action = ad.action
    if hasattr(action, 'fcurves'):
        return list(action.fcurves)
    slot = getattr(ad, 'action_slot', None)
    fcs = []
    for layer in getattr(action, 'layers', []):
        for strip in getattr(layer, 'strips', []):
            cb = strip.channelbag(slot) if slot is not None else None
            if cb is not None:
                fcs.extend(cb.fcurves)
    return fcs


def setup_world(scene):
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    tree = scene.world.node_tree
    for n in list(tree.nodes):
        tree.nodes.remove(n)
    out_node = tree.nodes.new('ShaderNodeOutputWorld')
    bg = tree.nodes.new('ShaderNodeBackground')
    tree.links.new(bg.outputs['Background'], out_node.inputs['Surface'])
    try:
        sky = tree.nodes.new('ShaderNodeTexSky')
        if hasattr(sky, 'sky_type'):
            sky.sky_type = 'NISHITA'
        if hasattr(sky, 'sun_elevation'):
            sky.sun_elevation = math.radians(25)
        if hasattr(sky, 'sun_rotation'):
            sky.sun_rotation = math.radians(45)
        if hasattr(sky, 'air_density'):
            sky.air_density = 1.0
        if hasattr(sky, 'dust_density'):
            sky.dust_density = 2.0
        tree.links.new(sky.outputs['Color'], bg.inputs['Color'])
        bg.inputs['Strength'].default_value = 0.3
    except Exception:
        bg.inputs['Color'].default_value = (0.08, 0.09, 0.10, 1.0)
        bg.inputs['Strength'].default_value = 0.3


def setup_render_settings(scene):
    engines = {e.identifier for e in
               bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items}
    for eng in ('BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'):
        if eng in engines:
            scene.render.engine = eng
            break
    else:
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = 64
        scene.cycles.use_denoising = True
    scene.render.film_transparent = False

    view_transforms = {e.identifier for e in
                       bpy.types.ColorManagedViewSettings.bl_rna
                       .properties['view_transform'].enum_items}
    for vt in ('AgX', 'Filmic', 'Standard'):
        if vt in view_transforms:
            scene.view_settings.view_transform = vt
            break
    scene.view_settings.exposure = 0.0


def main():
    output_dir = parse_output_dir()
    with open(output_dir / "scene_config.json") as f:
        cfg = json.load(f)
    overrides = load_material_overrides(output_dir)

    clear_scene()
    mesh_obj = import_mesh(output_dir / cfg["glb_relpath"])
    uv_name = mesh_obj.data.uv_layers[0].name if mesh_obj.data.uv_layers else None

    mat = build_palette_material(cfg, overrides, output_dir, uv_name)
    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(mat)

    setup_camera(cfg["intrinsics"], cfg["img_w"], cfg["img_h"])
    setup_lighting(mesh_obj)
    setup_world(bpy.context.scene)
    setup_render_settings(bpy.context.scene)

    blend_path = output_dir / "palette_scene.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print(f"Saved -> {blend_path}")


if __name__ == "__main__":
    main()
