"""Keep native GPU checks separate from the CPU suite."""

import pytest
from hypothesis import HealthCheck, settings

settings.register_profile(
    "ci",
    max_examples=60,
    derandomize=True,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
settings.load_profile("ci")


def pytest_collection_modifyitems(items):
    gpu_items = []
    for item in items:
        if item.get_closest_marker("gpu") is not None:
            gpu_items.append(item)
    if not gpu_items:
        return
    try:
        import torch
    except ImportError:
        available = False
    else:
        available = torch.cuda.is_available()
    if not available:
        skip = pytest.mark.skip(reason="CUDA is required for native GPU checks")
        for item in gpu_items:
            item.add_marker(skip)
