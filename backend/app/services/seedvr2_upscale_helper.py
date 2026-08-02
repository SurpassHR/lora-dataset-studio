"""SeedVR2 super-resolution — the third local ComfyUI engine, alongside Krea and
Klein.

WHAT IT IS
----------
SeedVR2 is a DiT-based video/image super-resolution model that upscales
high-resolution input in tiles. This module submits the repository's ComfyUI
workflow to the user's ComfyUI, then runs a batch upscale after replacing the
input image and model paths.

WORKFLOW SOURCE
---------------
seedvr2_api.json was converted from SeedVR2.json (ComfyUI frontend format);
every node parameter was hand-verified against the current
seedvr2_videoupscaler (v2.5.14+) schema. The LoadImage input is replaced with the
real image and the SaveImage output carries a unique prefix.
"""
from __future__ import annotations
import logging
import os
import random
import time
import uuid

from .. import config as cfg
from . import comfy_model_paths
from ..utils import comfy_fs
from ..utils.comfyui import load_workflow_local
from ..job_queue import queue_manager

logger = logging.getLogger(__name__)

ENGINE_ID = 'seedvr2'
ENGINE_LABEL = 'SeedVR2 Upscale'

SEEDVR2_WORKFLOW_PATH = cfg.BACKEND_DIR / 'workflows' / 'seedvr2_api.json'

# Custom nodes this workflow needs and the pack that ships each
SEEDVR2_NODE_CLASSES = (
    'SeedVR2LoadVAEModel',
    'SeedVR2VideoUpscaler',
    'SeedVR2LoadDiTModel',
    'TTP_Image_Tile_Batch',
    'TTP_Tile_image_size',
    'TTP_Image_Assy',
    'easy imageScaleToNormPixels',
    'Get Image Size',
    '数学运算_孤海',
    'ImageResize+',
    'GetImageSize+',
    'FastFilmGrain',
)
SEEDVR2_REQUIRED_NODES = (
    'SeedVR2LoadVAEModel',
    'SeedVR2VideoUpscaler',
    'SeedVR2LoadDiTModel',
)
SEEDVR2_NODE_PACKS = {
    'SeedVR2LoadVAEModel': ('seedvr2_videoupscaler',
                            'https://github.com/huzixiao/ComfyUI-seedvr2-video-upscaler'),
    'SeedVR2VideoUpscaler': ('seedvr2_videoupscaler',
                             'https://github.com/huzixiao/ComfyUI-seedvr2-video-upscaler'),
    'SeedVR2LoadDiTModel': ('seedvr2_videoupscaler',
                            'https://github.com/huzixiao/ComfyUI-seedvr2-video-upscaler'),
    'TTP_Image_Tile_Batch': ('Comfyui_TTP_Toolset',
                             'https://github.com/TTPlanetPig/Comfyui_TTP_Toolset'),
    'TTP_Tile_image_size': ('Comfyui_TTP_Toolset',
                            'https://github.com/TTPlanetPig/Comfyui_TTP_Toolset'),
    'TTP_Image_Assy': ('Comfyui_TTP_Toolset',
                       'https://github.com/TTPlanetPig/Comfyui_TTP_Toolset'),
    'easy imageScaleToNormPixels': ('ComfyUI-Easy-Use',
                                    'https://github.com/yolain/ComfyUI-Easy-Use'),
    'Get Image Size': ('masquerade-nodes-comfyui',
                       'https://github.com/BadCafeCode/masquerade-nodes-comfyui'),
    '数学运算_孤海': ('Goohaitools-comfyui',
                    'https://github.com/goohai/Goohaitools-comfyui'),
    'ImageResize+': ('comfyui_essentials',
                     'https://github.com/cubiq/ComfyUI_essentials'),
    'GetImageSize+': ('comfyui_essentials',
                      'https://github.com/cubiq/ComfyUI_essentials'),
    'FastFilmGrain': ('comfyui-vrgamedevgirl',
                      'https://github.com/vrgamegirl19/comfyui-vrgamedevgirl'),
}

_MODEL_SUFFIXES = ('.safetensors', '.gguf', '.sft')


class SeedVR2ModelsMissing(Exception):
    """SeedVR2 model files are missing and/or ComfyUI lacks required custom nodes."""
    def __init__(self, missing, missing_nodes=None):
        self.missing = list(missing or [])
        self.missing_nodes = list(missing_nodes or [])
        super().__init__('SeedVR2 assets missing: '
                         + ', '.join(self.missing + self.missing_nodes))


# --- Model resolution ---


def _seedvr2_custom_root():
    """Return the absolute path to the ``models/SEEDVR2/`` directory that the
    seedvr2_videoupscaler plugin registers as its own model root (via
    ``folder_paths.add_model_folder_path('seedvr2', ...)``). Unlike the standard
    ComfyUI model roots (``diffusion_models/``, ``vae/``), this is a separate
    cache directory the plugin owns. The plugin's Combo input lists files from
    **both** its own root AND the standard roots, so we must search both.

    Returns ``None`` when ComfyUI is not configured (the caller degrades to
    standard-root search only, which is the pre-existing behaviour)."""
    try:
        base = cfg.comfyui_dir('models')
        if not base:
            base = cfg.get('comfyui.base_dir') or ''
            if base:
                base = os.path.join(str(base), 'models')
        if not base:
            return None
        cand = os.path.join(str(base), 'SEEDVR2')
        return cand if os.path.isdir(cand) else None
    except Exception:
        return None


def _search_roots_and_seedvr2(comfy_type):
    """Yield ``(folder, filenames)`` from the standard ``comfy_type`` search roots
    AND from the plugin's ``models/SEEDVR2/`` directory (when it exists). The
    plugin's Combo model list includes both, so the resolver must check both.

    Also scans SEEDVR2-named subfolders inside each standard root (e.g.
    ``diffusion_models/SEEDVR2/``) so models placed there by the user are
    discoverable."""
    seen = set()
    for folder in comfy_model_paths.search_roots(comfy_type):
        norm = os.path.normpath(folder)
        if norm in seen:
            continue
        seen.add(norm)
        try:
            names = sorted(n for n in os.listdir(folder)
                           if n.lower().endswith(_MODEL_SUFFIXES))
            yield folder, names
        except OSError:
            continue
        # Also scan SEEDVR2 subfolders
        try:
            for d in sorted(x for x in os.listdir(folder)
                            if os.path.isdir(os.path.join(folder, x))):
                if 'seedvr' in d.lower():
                    sub = os.path.join(folder, d)
                    names = sorted(n for n in os.listdir(sub)
                                   if n.lower().endswith(_MODEL_SUFFIXES))
                    if names:
                        yield sub, names
        except OSError:
            continue
    # Also search the plugin's own root
    sr = _seedvr2_custom_root()
    if sr and os.path.normpath(sr) not in seen:
        try:
            names = sorted(n for n in os.listdir(sr)
                           if n.lower().endswith(_MODEL_SUFFIXES))
            yield sr, names
        except OSError:
            pass


def _auto_search_dit():
    """Auto-search for a DiT model across standard roots AND the plugin's
    SEEDVR2/ directory. Returns the bare filename, or None."""
    # Canonical name first
    for _folder, names in _search_roots_and_seedvr2('diffusion_models'):
        if 'seedvr2_ema_3b_fp8_e4m3fn.safetensors' in names:
            return 'seedvr2_ema_3b_fp8_e4m3fn.safetensors'
    # Narrow token — prefer 3b over 7b (smaller, faster)
    for _folder, names in _search_roots_and_seedvr2('diffusion_models'):
        for n in names:
            if any(tok in n.lower() for tok in
                   ('seedvr2_ema_3b', 'seedvr2_3b', 'seedvr2_ema')):
                return n
    # 7b as last resort
    for _folder, names in _search_roots_and_seedvr2('diffusion_models'):
        for n in names:
            if 'seedvr2' in n.lower():
                return n
    return None


def _verify_pick_bare(pick_bare):
    """Verify that ``pick_bare`` exists in standard roots OR the plugin's SEEDVR2
    directory. Returns the bare name on success, None on failure."""
    for _folder, names in _search_roots_and_seedvr2('diffusion_models'):
        if pick_bare in names:
            return pick_bare
    return None


def resolve_seedvr2_dit_model(selected=None):
    """Resolve the DiT model BARE filename (seedvr2_videoupscaler's ``model`` is a
    Combo input — ComfyUI only accepts the bare filename that /object_info lists,
    never a subfolder-prefixed path). Explicit setting wins, else auto-search.

    The resolver searches **both** the standard ``diffusion_models/`` roots AND
    the plugin's own ``models/SEEDVR2/`` directory, because the plugin's Combo
    list includes files from both locations.

    A user-supplied absolute path like ``/media/.../models/SEEDVR2/foo.safetensors``
    is reduced to its basename; the file is only accepted when it actually resolves
    under a search root (so a typo degrades to auto-search instead of enqueuing a
    doomed job)."""
    pick = selected or cfg.get('seedvr2.dit_model') or ''
    pick_bare = os.path.basename(str(pick).replace('/', os.sep).replace('\\', os.sep))
    if pick_bare:
        found = _verify_pick_bare(pick_bare)
        if found:
            return found
        logger.warning('seedvr2.dit_model %r not found — falling back to auto', pick)
    return _auto_search_dit()


def _verify_vae_pick_bare(pick_bare):
    """Verify a VAE bare name across standard vae/ roots AND the plugin's SEEDVR2/."""
    for _folder, names in _search_roots_and_seedvr2('vae'):
        if pick_bare in names:
            return pick_bare
    return None


def _auto_search_vae():
    """Auto-search for a VAE model. Returns the bare filename, or None."""
    for _folder, names in _search_roots_and_seedvr2('vae'):
        if 'ema_vae_fp16.safetensors' in names:
            return 'ema_vae_fp16.safetensors'
    for _folder, names in _search_roots_and_seedvr2('vae'):
        for n in names:
            if any(tok in n.lower() for tok in ('ema_vae_fp16', 'seedvr2_vae', 'vae')):
                return n
    return None


def resolve_seedvr2_vae_model(selected=None):
    """Resolve the VAE model BARE filename. Explicit setting wins, else auto-search.
    Searches both standard ``vae/`` roots AND the plugin's ``models/SEEDVR2/``."""
    pick = selected or cfg.get('seedvr2.vae_model') or ''
    pick_bare = os.path.basename(str(pick).replace('/', os.sep).replace('\\', os.sep))
    if pick_bare:
        found = _verify_vae_pick_bare(pick_bare)
        if found:
            return found
        logger.warning('seedvr2.vae_model %r not found — falling back to auto', pick)
    return _auto_search_vae()


def seedvr2_missing_assets():
    """Which model files are missing: ['dit_model', 'vae_model'] or []."""
    missing = []
    if not resolve_seedvr2_dit_model():
        missing.append('dit_model')
    if not resolve_seedvr2_vae_model():
        missing.append('vae_model')
    return missing


# --- Custom-node preflight ---

_NODES_OK_TTL_S = 300
_nodes_ok_until = 0.0


def seedvr2_missing_nodes():
    """[class_type] of the required custom nodes the target ComfyUI does not expose."""
    global _nodes_ok_until
    if time.time() < _nodes_ok_until:
        return []
    from ..utils.comfyui import fetch_object_info_classes
    available = fetch_object_info_classes()
    if available is None:
        return []
    out = sorted(c for c in SEEDVR2_REQUIRED_NODES if c not in available)
    if not out:
        _nodes_ok_until = time.time() + _NODES_OK_TTL_S
    return out


def clear_nodes_cache():
    global _nodes_ok_until
    _nodes_ok_until = 0.0


def seedvr2_node_hints(nodes):
    """[{class_type, pack, url}] of the missing nodes, for the front-end hint."""
    out = []
    for ct in nodes or []:
        meta = SEEDVR2_NODE_PACKS.get(ct)
        if meta:
            out.append({'class_type': ct, 'pack': meta[0], 'url': meta[1]})
    return out


def seedvr2_missing_file_entries(missing):
    """[{path, kind, source}] of the missing files, for the front-end hint."""
    entries = {
        'dit_model': {'path': 'models/diffusion_models/SEEDVR2/',
                      'kind': 'SeedVR2 DiT model',
                      'source': 'https://huggingface.co/godly/seedvr2'},
        'vae_model': {'path': 'models/SEEDVR2/',
                      'kind': 'SeedVR2 VAE model',
                      'source': 'https://huggingface.co/godly/seedvr2'},
    }
    return [entries[m] for m in (missing or []) if m in entries]


def seedvr2_model_options():
    """[{value, label}] for the DiT and VAE model Combo inputs, straight from the
    target ComfyUI's /object_info (the SAME list the SeedVR2 nodes publish, so the
    settings dropdown offers exactly what ComfyUI will accept).

    Returns ``{'dit': [...], 'vae': [...]}`` where each entry is
    ``{'value': published_name, 'label': published_name}`` — or empty lists when
    /object_info is unreachable (fail-open: an unreachable ComfyUI must not break
    the settings screen; the resolver still falls back to its disk scan)."""
    from ..utils.comfyui import fetch_object_info_model_files
    files = fetch_object_info_model_files()
    out = {'dit': [], 'vae': []}
    if not isinstance(files, dict):
        return out
    for cls, published in (('SeedVR2LoadDiTModel', 'dit'),
                           ('SeedVR2LoadVAEModel', 'vae')):
        combos = files.get(cls) or {}
        models = combos.get('model')
        if not isinstance(models, dict):
            continue
        # `models` maps normalised -> published; dedupe published names.
        seen, entries = set(), []
        for name in models.values():
            if name not in seen:
                seen.add(name)
                entries.append({'value': name, 'label': name})
        out[published] = entries
    return out


def preflight():
    """Raise SeedVR2ModelsMissing when the engine cannot run."""
    missing = seedvr2_missing_assets()
    nodes = seedvr2_missing_nodes()
    if missing or nodes:
        raise SeedVR2ModelsMissing(missing, nodes)


# --- Workflow loading & enqueue ---

def _comfy_input_dir() -> str:
    d = cfg.comfyui_dir('input')
    if not d:
        raise RuntimeError('ComfyUI is not configured')
    return str(d)


def enqueue_seedvr2_upscale(user_id, source_filename, source_path=None,
                            extra_metadata=None):
    """Copy the source into ComfyUI's input folder, load the workflow, and enqueue it.

    Args:
        user_id: The user id.
        source_filename: The source image filename in the ComfyUI output dir (or source_path).
        source_path: Absolute path to the source (None = read from the ComfyUI output dir).
        extra_metadata: Extra metadata (e.g. dataset_id, image_id) for the completion callback.

    Returns:
        str: The job id.

    Raises:
        SeedVR2ModelsMissing: a model or a node is missing.
        ValueError: the source image does not exist.
        RuntimeError: ComfyUI is not configured.
    """
    if source_path is None:
        out_dir = cfg.comfyui_dir('output')
        if not out_dir:
            raise RuntimeError('ComfyUI is not configured')
        source_path = os.path.join(str(out_dir), source_filename)
    if not os.path.exists(source_path):
        raise ValueError(f'source image not found: {source_filename}')

    preflight()

    # Resolve the model paths
    dit_model = resolve_seedvr2_dit_model()
    vae_model = resolve_seedvr2_vae_model()
    if not dit_model:
        raise SeedVR2ModelsMissing(['dit_model'])
    if not vae_model:
        raise SeedVR2ModelsMissing(['vae_model'])

    # Copy the source image into ComfyUI's input folder
    comfy_input_dir = comfy_fs.ensure_input_usable(_comfy_input_dir())
    uid = uuid.uuid4().hex[:8]
    source_stem = os.path.splitext(os.path.basename(str(source_filename)))[0] or 'source'
    staged_source = comfy_fs.stage_input_image(
        source_path, f'seedvr2_source_{uid}_{source_stem}.png', comfy_input_dir)
    comfy_input = os.path.basename(staged_source)

    # Load the workflow
    workflow = load_workflow_local(str(SEEDVR2_WORKFLOW_PATH)) or {}
    if not workflow:
        raise RuntimeError(f'Could not load SeedVR2 workflow from {SEEDVR2_WORKFLOW_PATH}')

    # Replace the placeholders
    prefix = f'{user_id}_SeedVR2_{uid}'
    for node_id, node in workflow.items():
        inputs = node.get('inputs', {})
        for key, value in list(inputs.items()):
            if value == '__INPUT_IMAGE__':
                inputs[key] = comfy_input
            elif value == '__VAE_MODEL__':
                inputs[key] = vae_model
            elif value == '__DIT_MODEL__':
                inputs[key] = dit_model
            elif value == '__PREFIX__':
                inputs[key] = prefix

    # Enqueue
    job_id = str(uuid.uuid4())
    meta = {'model_name': 'seedvr2_upscale', 'staged_inputs': [comfy_input]}
    if extra_metadata:
        meta.update(extra_metadata)
    queue_manager.add_job(job_type='image', user_id=str(user_id),
                          workflow_data=workflow, prompt='seedvr2 upscale',
                          job_id=job_id, metadata=meta)
    return job_id