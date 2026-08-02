"""SeedVR2 super-resolution — the third local ComfyUI engine.

WHAT THESE TESTS ARE FOR
------------------------
- Model resolution: canonical name first, narrow-token fallback, None when nothing
  matches
- Missing-asset detection
- Workflow loading and placeholder replacement
- The correctness of the enqueued workflow JSON

All pure Python — no GPU or live ComfyUI involved.
"""
from __future__ import annotations
import importlib
import json
import os
import struct
import pytest


def _fresh_config(monkeypatch, tmp_path):
    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    import app.config as config
    importlib.reload(config)
    return config


def _comfy_tree(tmp_path):
    """A minimal ComfyUI tree (with input/output dirs)."""
    base = tmp_path / 'Comfy'
    for sub in ('diffusion_models', 'vae', 'input', 'output', 'SEEDVR2'):
        (base / 'models' if sub != 'input' and sub != 'output' else base / sub).mkdir(parents=True, exist_ok=True)
    return base


_VALID_ST = struct.pack('<Q', 2) + b'{}'


def _write(path, data=_VALID_ST):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def seedvr2(monkeypatch, tmp_path):
    """seedvr2_upscale_helper bound to a throwaway ComfyUI tree + config."""
    config = _fresh_config(monkeypatch, tmp_path)
    base = _comfy_tree(tmp_path)
    config.save_config({'comfyui': {'base_dir': str(base)}})
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    import app.services.seedvr2_upscale_helper as suh
    importlib.reload(suh)
    suh._nodes_ok_until = 0.0
    yield suh, base, config
    comfy_model_paths.clear_cache()


# --- DiT model resolution ------------------------------------------------------------

def test_no_dit_model_at_all_returns_none(seedvr2):
    """None when no DiT model is on disk, no exception."""
    suh, base, _ = seedvr2
    assert suh.resolve_seedvr2_dit_model() is None
    assert 'dit_model' in suh.seedvr2_missing_assets()


def test_dit_model_found_in_flat_diffusion_models(seedvr2):
    """A flat name in the diffusion_models root resolves."""
    suh, base, _ = seedvr2
    _write(base / 'models' / 'diffusion_models' / 'seedvr2_ema_3b_fp8_e4m3fn.safetensors')
    assert suh.resolve_seedvr2_dit_model() == 'seedvr2_ema_3b_fp8_e4m3fn.safetensors'


def test_dit_model_found_in_seedvr2_subfolder(seedvr2):
    """A model in a SEEDVR2 subfolder resolves to its BARE name (the node's Combo
    accepts no subfolder prefix — see the SeedVR2VideoUpscaler Combo constraint)."""
    suh, base, _ = seedvr2
    _write(base / 'models' / 'diffusion_models' / 'SEEDVR2' / 'seedvr2_ema_3b_fp8_e4m3fn.safetensors')
    result = suh.resolve_seedvr2_dit_model()
    assert result is not None
    # Bare filename only — a subfolder prefix would fail ComfyUI's Combo validation.
    assert result == 'seedvr2_ema_3b_fp8_e4m3fn.safetensors'


def test_dit_model_found_in_plugin_root(seedvr2):
    """A model in models/SEEDVR2/ (the plugin's own root directory) resolves."""
    suh, base, _ = seedvr2
    _write(base / 'models' / 'SEEDVR2' / 'seedvr2_ema_3b_fp8_e4m3fn.safetensors')
    result = suh.resolve_seedvr2_dit_model()
    assert result == 'seedvr2_ema_3b_fp8_e4m3fn.safetensors'


def test_dit_narrow_token_fallback(seedvr2):
    """Without the canonical name, a narrow-token match should work."""
    suh, base, _ = seedvr2
    _write(base / 'models' / 'diffusion_models' / 'my_custom_seedvr2_rename.safetensors')
    assert suh.resolve_seedvr2_dit_model() == 'my_custom_seedvr2_rename.safetensors'


def test_dit_explicit_choice_wins(seedvr2):
    """An explicit setting wins even when it is not the canonical name."""
    suh, base, config = seedvr2
    config.save_config({'seedvr2': {'dit_model': 'my_custom.safetensors'}})
    _write(base / 'models' / 'diffusion_models' / 'my_custom.safetensors')
    # Reload the helper to pick up the new config
    importlib.reload(suh)
    assert suh.resolve_seedvr2_dit_model() == 'my_custom.safetensors'


# --- VAE model resolution -----------------------------------------------------------

def test_no_vae_model_at_all_returns_none(seedvr2):
    """None when no VAE model is on disk."""
    suh, base, _ = seedvr2
    assert suh.resolve_seedvr2_vae_model() is None
    assert 'vae_model' in suh.seedvr2_missing_assets()


def test_vae_model_found(seedvr2):
    """A VAE model in the vae/ root resolves."""
    suh, base, _ = seedvr2
    _write(base / 'models' / 'vae' / 'ema_vae_fp16.safetensors')
    assert suh.resolve_seedvr2_vae_model() == 'ema_vae_fp16.safetensors'


# --- Missing-asset detection -----------------------------------------------------------

def test_fully_installed_engine_has_no_missing_assets(seedvr2):
    """Empty list when all models are present."""
    suh, base, _ = seedvr2
    _write(base / 'models' / 'diffusion_models' / 'seedvr2_ema_3b_fp8_e4m3fn.safetensors')
    _write(base / 'models' / 'vae' / 'ema_vae_fp16.safetensors')
    assert suh.seedvr2_missing_assets() == []


def test_both_models_missing_reported(seedvr2):
    """Both models listed when complete."""
    suh, base, _ = seedvr2
    missing = suh.seedvr2_missing_assets()
    assert 'dit_model' in missing
    assert 'vae_model' in missing


# --- Workflow loading and placeholder replacement -------------------------------------------------

def test_workflow_loads_and_has_placeholders(seedvr2, app):
    """seedvr2_api.json should load to a valid dict with placeholders."""
    import json
    suh, base, _ = seedvr2
    from app.utils.comfyui import load_workflow_local
    with app.app_context():
        wf = load_workflow_local(str(suh.SEEDVR2_WORKFLOW_PATH))
    assert wf, 'workflow should load'
    # Assert placeholders are present
    inputs_str = json.dumps(wf)
    assert '__INPUT_IMAGE__' in inputs_str
    assert '__VAE_MODEL__' in inputs_str
    assert '__DIT_MODEL__' in inputs_str
    assert '__PREFIX__' in inputs_str


def test_workflow_has_saveimage_output(seedvr2, app):
    """Output node should be SaveImage, not PreviewImage."""
    suh, base, _ = seedvr2
    from app.utils.comfyui import load_workflow_local
    with app.app_context():
        wf = load_workflow_local(str(suh.SEEDVR2_WORKFLOW_PATH))
    save_nodes = [n for n in wf.values() if n['class_type'] == 'SaveImage']
    assert len(save_nodes) == 1, 'workflow must have exactly one SaveImage node'
    preview_nodes = [n for n in wf.values() if n['class_type'] == 'PreviewImage']
    assert len(preview_nodes) == 0, 'workflow must not have PreviewImage nodes'


def test_enqueue_replaces_placeholders(app, tmp_path, monkeypatch):
    """enqueue_seedvr2_upscale should replace all placeholders."""
    # Build the comfy tree using the app fixture
    base = tmp_path / 'Comfy'
    for sub in ('diffusion_models', 'vae', 'input', 'output'):
        (base / 'models' if sub not in ('input', 'output') else base / sub).mkdir(parents=True, exist_ok=True)
    # Set the comfyui base dir
    from app import config as cfg
    cfg.save_config({'comfyui': {'base_dir': str(base)}})
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    import app.services.seedvr2_upscale_helper as suh
    importlib.reload(suh)
    suh._nodes_ok_until = 0.0
    # Create the model files
    _write(base / 'models' / 'diffusion_models' / 'seedvr2_ema_3b_fp8_e4m3fn.safetensors')
    _write(base / 'models' / 'vae' / 'ema_vae_fp16.safetensors')
    # Create the source image
    src = base / 'output' / 'test_source.png'
    src.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.new('RGB', (256, 256)).save(src)
    captured = {}
    from app.job_queue import queue_manager
    original_add = queue_manager.add_job
    def _capture(**kw):
        captured.update(kw)
        return kw.get('job_id') or 'test-job-id'
    queue_manager.add_job = _capture
    try:
        with app.app_context():
            job_id = suh.enqueue_seedvr2_upscale(
                user_id='test_user',
                source_filename='test_source.png',
            )
        assert job_id, 'should return a job id'
        assert 'workflow_data' in captured
        wf = captured['workflow_data']
        wf_str = json.dumps(wf)
        assert '__INPUT_IMAGE__' not in wf_str, 'input placeholder should be replaced'
        assert '__VAE_MODEL__' not in wf_str, 'vae placeholder should be replaced'
        assert '__DIT_MODEL__' not in wf_str, 'dit placeholder should be replaced'
        assert '__PREFIX__' not in wf_str, 'prefix placeholder should be replaced'
    finally:
        queue_manager.add_job = original_add


# --- model dropdown options ---------------------------------------------------
def test_model_options_from_object_info(app, monkeypatch):
    """seedvr2_model_options should read the exact model list from /object_info."""
    from app.services import seedvr2_upscale_helper as suh
    fake = {
        'SeedVR2LoadDiTModel': {'model': {
            'seedvr2_ema_3b_fp8_e4m3fn.safetensors': 'seedvr2_ema_3b_fp8_e4m3fn.safetensors',
            'seedvr2_ema_7b_fp16.safetensors': 'seedvr2_ema_7b_fp16.safetensors',
        }},
        'SeedVR2LoadVAEModel': {'model': {
            'ema_vae_fp16.safetensors': 'ema_vae_fp16.safetensors',
        }},
    }
    monkeypatch.setattr(
        'app.utils.comfyui.fetch_object_info_model_files', lambda: fake)
    opts = suh.seedvr2_model_options()
    assert [o['value'] for o in opts['dit']] == [
        'seedvr2_ema_3b_fp8_e4m3fn.safetensors', 'seedvr2_ema_7b_fp16.safetensors']
    assert [o['value'] for o in opts['vae']] == ['ema_vae_fp16.safetensors']


def test_model_options_fail_open_when_probe_unreachable(app, monkeypatch):
    """An unreachable /object_info must not break the options call."""
    from app.services import seedvr2_upscale_helper as suh
    monkeypatch.setattr(
        'app.utils.comfyui.fetch_object_info_model_files', lambda: None)
    assert suh.seedvr2_model_options() == {'dit': [], 'vae': []}