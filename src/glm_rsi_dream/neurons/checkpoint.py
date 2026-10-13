"""Read the frozen EXL3 checkpoint and the inspected runtime's weight transforms."""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


class FrozenCheckpoint:
    """Disk-cached fixed matrices, with no full-model GPU or RAM copy."""

    def __init__(self, root: Any, cache: Any, device: Any = 'cuda', cache_bytes: Any = 24 * 1024**3) -> None:
        from safetensors import safe_open
        from tensorfold.families.glm5_next.cuda import exl3, qmm

        self.root = Path(root).resolve()
        self.cache = Path(cache).resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.cache_bytes = cache_bytes
        self.gpu_cache: Any = OrderedDict()
        self.gpu_bytes = 0
        self.qmm = qmm
        self.exl3 = exl3
        self.safe_open = safe_open
        self.mapping: dict[Any, Any] = {}
        for path in sorted(self.root.glob('*.safetensors')):
            with self.safe_open(path, framework='pt', device='cpu') as handle:
                for name in handle.keys():
                    if name in self.mapping:
                        raise ValueError('Checkpoint tensor has multiple owners: ' + name)
                    self.mapping[name] = path
        config_sha = hashlib.sha256((self.root / 'config.json').read_bytes()).hexdigest()
        identity = {'checkpoint': str(self.root), 'config_sha256': config_sha,
                    'dense': 'q4', 'head': 'fp8', 'kv_b': 'bf16', 'kv_cache': 'fp8'}
        identity_path = self.cache / 'identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError('Fixed-weight cache belongs to another checkpoint or runtime')
        identity_path.write_text(json.dumps(identity, indent=2) + '\n')

    def cache_path(self, name: Any, kind: Any) -> Any:
        digest = hashlib.sha256(name.encode()).hexdigest()
        return self.cache / (kind + '-' + digest + '.pt')

    def raw(self, name: Any) -> Any:
        with self.safe_open(self.mapping[name], framework='pt', device='cpu') as handle:
            tensor = handle.get_tensor(name)
        tensor._dream_name = name
        return tensor

    def remember(self, name: Any, tensor: Any) -> Any:
        """Bound fixed-matrix residency while preserving the same numerical bytes."""
        if tensor.requires_grad:
            raise ValueError('Fixed matrix cache requires frozen tensors')
        size = tensor.numel() * tensor.element_size()
        while self.gpu_cache and self.gpu_bytes + size > self.cache_bytes:
            _, old = self.gpu_cache.popitem(last=False)
            self.gpu_bytes -= old.numel() * old.element_size()
        tensor = tensor.to(self.device)
        if size <= self.cache_bytes:
            self.gpu_cache[name] = tensor
            self.gpu_bytes += size
        return tensor

    def __call__(self, name: Any) -> Any:
        transformed = name == 'lm_head.weight'
        projection = '.self_attn.' in name and '_proj.weight' in name
        shared = '.mlp.shared_experts.' in name and name.endswith('.weight')
        if 'kv_b_proj' in name:
            projection = False
        if not (transformed or projection or shared):
            return self.raw(name)
        key = 'runtime-' + name
        if key in self.gpu_cache:
            self.gpu_cache.move_to_end(key)
            return self.gpu_cache[key]
        path = self.cache_path(name, 'runtime')
        if path.exists():
            weight = torch.load(path, weights_only=True, map_location='cpu')
            return self.remember(key, weight)
        source = self.raw(name).to(self.device)
        if transformed:
            quantized = self.qmm.make_f8(source)
            scale = quantized.scale.repeat_interleave(128, dim=1)
            weight = quantized.weight.float() * scale
        else:
            words, scales, biases = self.qmm.quantize4_mse_raw(source)
            weight = self.qmm.dequantize(words, scales, biases)
        weight = weight.cpu()
        temporary = path.with_suffix('.tmp')
        torch.save(weight, temporary)
        temporary.replace(path)
        return self.remember(key, weight)

    def unpack(self, trellis: Any) -> Any:
        name = trellis._dream_name
        key = 'rotated-' + name
        if key in self.gpu_cache:
            self.gpu_cache.move_to_end(key)
            return self.gpu_cache[key]
        path = self.cache_path(name, 'rotated')
        if path.exists():
            weight = torch.load(path, weights_only=True, map_location='cpu')
            return self.remember(key, weight)
        weight = self.exl3.unpack(trellis)
        temporary = path.with_suffix('.tmp')
        torch.save(weight, temporary)
        temporary.replace(path)
        return self.remember(key, weight)
