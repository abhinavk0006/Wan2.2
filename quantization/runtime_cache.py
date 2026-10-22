"""Bounded expert reuse and content-addressed, tensor-only disk caches.

Disk expert caching is opt-in: two INT8 14B experts need about 27 GiB before
source checkpoints, which can exceed a Kaggle working volume. CPU reuse keeps
only the high-noise expert by default and checks host headroom before use.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

import torch

CACHE_VERSION = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def available_ram():
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        try:
            for line in Path('/proc/meminfo').read_text().splitlines():
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
        except OSError:
            pass
    return 0  # Unknown capacity must not enable a multi-GiB allocation.


def file_identity(path):
    """Hash source contents once per worker, before caching a converted expert."""
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            sha.update(block)
    return {'sha256': sha.hexdigest(), 'bytes': Path(path).stat().st_size}


def storage_bytes(model):
    return sum(t.numel() * t.element_size() for t in model.state_dict().values())


def cache_source_identity():
    root = Path(__file__).parent
    identity = {name: file_identity(root / name)['sha256'] for name in
            ('kaggle_single_expert.py', 'int8_linear.py', 'runtime_cache.py',
             'optimization/lora.py')}
    identity['wan_model'] = file_identity(root.parent / 'wan/modules/model.py')['sha256']
    identity['torch'] = str(torch.__version__)
    return identity


class ExpertCache:
    def __init__(self, *, cpu_gib=14, reserve_gib=3, directory=None, disk_gib=32):
        self.cpu_limit = int(cpu_gib * 2**30)
        self.reserve = int(reserve_gib * 2**30)
        self.directory = Path(directory) if directory else None
        self.disk_limit = int(disk_gib * 2**30)
        self.cpu = {}

    def take_cpu(self, key):
        return self.cpu.pop(key, None)

    def retain(self, key, name, model):
        # A single pinned high expert avoids alternating two entries that do
        # not fit in host memory. Keep the low expert disposable.
        size = storage_bytes(model)
        if name != 'high_noise_model' or not self.cpu_limit or size > self.cpu_limit:
            return False
        self.cpu.clear()
        if available_ram() < size + self.reserve:
            logging.info('CPU expert cache skipped: insufficient host headroom')
            return False
        from accelerate.hooks import remove_hook_from_module
        try:
            remove_hook_from_module(model, recurse=True)
            model.cpu()
        except (MemoryError, RuntimeError):
            logging.warning('CPU expert retention failed; dropping optional cache entry', exc_info=True)
            return False
        self.cpu[key] = model
        logging.info('Retained %s in bounded CPU cache (%.2f GiB)', name, size / 2**30)
        return True

    def reserve_conversion_memory(self, required_bytes):
        if self.cpu and available_ram() < required_bytes + self.reserve:
            logging.info('Evicting CPU expert cache to leave conversion headroom')
            self.cpu.clear()
            import gc
            gc.collect()

    def load_disk(self, key):
        if not self.directory:
            return None
        folder = self.directory / key
        manifest_path = folder / 'manifest.json'
        if not manifest_path.is_file():
            return None
        try:
            from accelerate import init_empty_weights
            from accelerate.utils import set_module_tensor_to_device
            from safetensors import safe_open
            from wan.modules.model import WanModel
            from int8_linear import Int8Linear
            manifest = json.loads(manifest_path.read_text())
            if manifest['version'] != CACHE_VERSION or manifest['key'] != key:
                return None
            if available_ram() < manifest['storage_bytes'] + self.reserve:
                self.cpu.clear()
                import gc
                gc.collect()
                if available_ram() < manifest['storage_bytes'] + self.reserve:
                    logging.warning('Disk expert cache skipped: insufficient host RAM')
                    return None
            with init_empty_weights():
                model = WanModel(**manifest['model_config']).to(dtype=torch.float16)
                for name, spec in manifest['int8'].items():
                    parent_name, _, child = name.rpartition('.')
                    parent = model.get_submodule(parent_name) if parent_name else model
                    module = Int8Linear(spec['in'], spec['out'], bias=spec['bias'], dtype=torch.float16)
                    module.scale = torch.empty(spec['scale_shape'], dtype=torch.float32)
                    setattr(parent, child, module.to('meta'))
            expected = model.state_dict()
            loaded = set()
            for shard in manifest['shards']:
                filename = shard['file']
                if Path(filename).name != filename:
                    raise ValueError('Invalid cache shard path')
                path = folder / filename
                if file_identity(path)['sha256'] != shard['sha256']:
                    raise ValueError('Cache shard checksum mismatch')
                with safe_open(str(path), framework='pt', device='cpu') as handle:
                    for name in handle.keys():
                        tensor = handle.get_tensor(name)
                        if name in loaded or name not in expected or tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
                            raise ValueError(f'Invalid cached tensor: {name}')
                        set_module_tensor_to_device(model, name, 'cpu', value=tensor)
                        loaded.add(name)
            if loaded != set(expected):
                raise ValueError('Incomplete expert cache')
            model.eval().requires_grad_(False)
            os.utime(manifest_path, None)
            return model
        except Exception:
            logging.exception('Ignoring unusable expert cache %s; converting source instead', folder)
            return None

    def save_disk(self, key, model, model_config):
        if not self.directory:
            return False
        from safetensors.torch import save_file
        from int8_linear import Int8Linear
        self.directory.mkdir(parents=True, exist_ok=True)
        size = storage_bytes(model)
        # Do not evict another useful expert only to alternate cache misses.
        used = sum(p.stat().st_size for p in self.directory.rglob('*') if p.is_file())
        if used + size > self.disk_limit or shutil.disk_usage(self.directory).free < size + 2 * 2**30:
            logging.warning('Skipping disk expert cache: need %.1f GiB plus 2 GiB free; use mounted source weights or more disk', size / 2**30)
            return False
        destination = self.directory / key
        if destination.exists():
            return False
        temporary = Path(tempfile.mkdtemp(prefix='.writing-', dir=self.directory))
        try:
            shards, pending, pending_bytes = [], {}, 0
            def flush():
                nonlocal pending, pending_bytes
                if not pending:
                    return
                name = f'weights-{len(shards):04d}.safetensors'
                save_file(pending, str(temporary / name))
                shards.append({'file': name, 'sha256': file_identity(temporary / name)['sha256']})
                pending, pending_bytes = {}, 0
            for name, tensor in model.state_dict().items():
                byte_count = tensor.numel() * tensor.element_size()
                if pending_bytes + byte_count > 512 * 2**20:
                    flush()
                pending[name] = tensor.detach().cpu().contiguous()
                pending_bytes += byte_count
            flush()
            int8 = {name: {'in': m.in_features, 'out': m.out_features,
                          'bias': m.bias is not None, 'scale_shape': list(m.scale.shape)}
                    for name, m in model.named_modules() if isinstance(m, Int8Linear)}
            (temporary / 'manifest.json').write_text(json.dumps({'version': CACHE_VERSION,
                'key': key, 'storage_bytes': size, 'model_config': model_config,
                'int8': int8, 'shards': shards}, indent=2))
            temporary.rename(destination)
            return True
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


class PromptDiskCache:
    def __init__(self, directory, identity):
        self.directory = Path(directory) if directory else None
        self.identity = identity

    def key(self, prompt, negative, lightning):
        return digest({'version': CACHE_VERSION, 'encoder': self.identity,
                       'prompt': prompt, 'negative': '' if lightning else negative,
                       'lightning': lightning})

    def load(self, key):
        if not self.directory:
            return None
        path = self.directory / (key + '.safetensors')
        if not path.is_file():
            return None
        try:
            from safetensors import safe_open
            with safe_open(str(path), framework='pt', device='cpu') as handle:
                metadata = handle.metadata() or {}
                if metadata.get('key') != key:
                    return None
                groups = []
                for prefix in ('positive', 'negative'):
                    count = int(metadata[prefix])
                    tensors = [handle.get_tensor(f'{prefix}.{i}') for i in range(count)]
                    if any(t.ndim != 2 or not torch.isfinite(t).all() for t in tensors):
                        return None
                    groups.append(tensors)
                return tuple(groups) if groups[0] else None
        except Exception:
            logging.warning('Ignoring invalid prompt cache %s', path, exc_info=True)
            return None

    def save(self, key, contexts):
        if not self.directory:
            return
        from safetensors.torch import save_file
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / (key + '.safetensors')
        fd, name = tempfile.mkstemp(prefix='.prompt-', dir=self.directory)
        os.close(fd)
        temporary = Path(name)
        try:
            tensors = {f'{prefix}.{i}': t.detach().cpu().contiguous()
                       for prefix, group in zip(('positive', 'negative'), contexts)
                       for i, t in enumerate(group)}
            save_file(tensors, str(temporary), metadata={'key': key,
                'positive': str(len(contexts[0])), 'negative': str(len(contexts[1]))})
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
