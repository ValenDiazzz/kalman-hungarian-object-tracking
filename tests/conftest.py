"""Scientific tests generate their temporary artifacts only in ignored paths."""

import tempfile
from pathlib import Path

import pytest

from kalman_hungarian_object_tracking.config import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def output_tmp():
    root = Path("outputs/test-tmp")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as directory:
        yield Path(directory)
