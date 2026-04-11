"""Shared test fixtures."""

import pytest
import torch


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def rng():
    """Seeded random generator for reproducibility."""
    return torch.Generator().manual_seed(42)
