"""Check bounded fixed-matrix residency without native runtime imports."""

from collections import OrderedDict

import pytest
import torch

from glm_rsi_dream.neurons.checkpoint import FrozenCheckpoint


def test_fixed_matrix_cache_evicts_without_changing_tensor_values():
    reader = object.__new__(FrozenCheckpoint)
    reader.device = 'cpu'
    reader.cache_bytes = 16
    reader.gpu_cache = OrderedDict()
    reader.gpu_bytes = 0
    first = torch.arange(4, dtype=torch.float32)
    second = torch.arange(4, dtype=torch.float32) + 8
    torch.testing.assert_close(reader.remember('first', first), first, rtol=0, atol=0)
    torch.testing.assert_close(reader.remember('second', second), second, rtol=0, atol=0)
    assert list(reader.gpu_cache) == ['second']
    assert reader.gpu_bytes == 16
    with pytest.raises(ValueError, match='frozen tensors'):
        reader.remember('trainable', torch.ones(4, requires_grad=True))
