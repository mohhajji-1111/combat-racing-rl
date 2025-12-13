"""
Test Configuration
=================

Pytest configuration and fixtures.

Author: Combat Racing RL Team
Date: 2024-2025
"""

import pytest
import numpy as np


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    yield
    np.random.seed(None)


@pytest.fixture
def sample_state():
    """Create sample state vector."""
    return np.random.randn(20)


@pytest.fixture
def sample_action():
    """Create sample action."""
    return 5
